import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import LineString, mapping
from shapely.ops import transform as shp_transform
from skimage.graph import route_through_array
from pyproj import CRS, Transformer, Geod


# Defaults
DEFAULT_COST_RASTER = (
    Path("data/processed") / "cost_Forest_height_2019_NAFR.tif"
)
DEFAULT_OUTPUT_DIR = Path("outputs")

# Approx coordinates (lon, lat) in WGS84
START_POINT = (-5.6060, 36.0130)
END_POINT = (-1.752427, 43.335370)

# Tunable defaults (easy to find)
DEFAULT_COARSE1_FACTOR = 4       # x4 downsampling for the coarse global pass
DEFAULT_REFINE_BUFFER_KM = 50.0   # 50 km corridor around coarse path for refinement
DEFAULT_FINE2_FACTOR = 1          # Native resolution for final pass (1=native, 2=half)
DEFAULT_MAX_NODES = 250_000_000    # Node budget to avoid OOM
DEFAULT_OUT_FORMAT = "gpkg"


class Spinner:
    """Simple console spinner to indicate progress while blocking ops run."""

    def __init__(self, message: str = "Working") -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def _run():
            frames = ["|", "/", "-", "\\"]
            i = 0
            while not self._stop.is_set():
                sys.stdout.write(f"\r{self.message} {frames[i % len(frames)]}")
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


# (Legacy helpers removed: compute_window_for_points, km_to_degree_buffer)


def read_cost_surface(
    src: rasterio.DatasetReader,
    window: Window | None = None,
    out_shape: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    """Read cost array and mask from rasterio source as float64.

    Masked cells are set to np.inf in the returned array.
    Returns (array, mask, out_transform)
    """
    if window is None:
        window = Window(0, 0, src.width, src.height)

    read_kwargs = {"window": window, "masked": True}
    if out_shape is not None:
        # out_shape is (rows, cols)
        read_kwargs.update(
            {
                "out_shape": (1, int(out_shape[0]), int(out_shape[1])),
                "resampling": Resampling.average,
            }
        )

    data = src.read(1, **read_kwargs)
    arr = data.filled(np.nan).astype(np.float64)

    # Determine nodata mask: masked or value equals src.nodata
    mask = np.zeros(arr.shape, dtype=bool)
    mask |= np.isnan(arr)
    nd = src.nodata
    if nd is not None:
        mask |= (arr == float(nd))

    # Replace masked with inf cost
    arr[mask] = np.inf

    # Good practice: ensure strictly non-negative
    arr[arr < 0] = 0.0

    # Transform for this window
    from rasterio.windows import transform as win_transform

    base_transform = win_transform(window, src.transform)
    if out_shape is None:
        out_transform = base_transform
    else:
        # Scale transform to the new out_shape
        scale_x = window.width / float(out_shape[1])
        scale_y = window.height / float(out_shape[0])
        out_transform = base_transform * rasterio.Affine.scale(scale_x, scale_y)
    return arr, mask, out_transform


def adjust_costs(
    arr: np.ndarray,
    eps: float = 1e-6,
    nonforest_weight: float = 1.0,
) -> np.ndarray:
    """Adjust costs to reduce zigzags while keeping forest preference.

    - Forest (0) becomes ~eps (small positive), so distance is weakly penalized.
    - Non-forest (1) becomes ~1+eps, keeping a strong penalty relative to forest.
    - np.inf remains a barrier.
    """
    out = arr.copy()
    finite = np.isfinite(out)
    out[finite] = eps + nonforest_weight * out[finite]
    return out


def buffer_line_geodesic(
    coords_lonlat: Sequence[Tuple[float, float]],
    buffer_km: float,
    dst_crs: CRS,
) -> "shapely.geometry.Polygon":
    """Create a geodesic-like buffer around a line and return polygon in dst_crs.

    Uses an Azimuthal Equidistant projection centered on the line midpoint to
    buffer in meters, then transforms back to WGS84 and into dst_crs.
    """
    if len(coords_lonlat) < 2:
        raise ValueError("Need at least two coordinates to buffer a line")

    line_ll = LineString(coords_lonlat)
    # Midpoint for projection center (approximate)
    mid = line_ll.interpolate(0.5, normalized=True)
    lon0, lat0 = float(mid.x), float(mid.y)

    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    to_aeqd = Transformer.from_crs(CRS.from_epsg(4326), aeqd, always_xy=True).transform
    to_ll = Transformer.from_crs(aeqd, CRS.from_epsg(4326), always_xy=True).transform

    line_m = shp_transform(to_aeqd, line_ll)
    poly_m = line_m.buffer(buffer_km * 1000.0)
    poly_ll = shp_transform(to_ll, poly_m)

    if dst_crs.to_epsg() == 4326 or dst_crs.is_geographic:
        return poly_ll
    # Transform to destination CRS
    to_dst = Transformer.from_crs(CRS.from_epsg(4326), dst_crs, always_xy=True).transform
    poly_dst = shp_transform(to_dst, poly_ll)
    return poly_dst


def nearest_valid_cell(
    arr: np.ndarray, mask: np.ndarray, rc: Tuple[int, int], max_radius: int = 50
) -> Tuple[int, int]:
    r, c = rc
    if (
        0 <= r < mask.shape[0]
        and 0 <= c < mask.shape[1]
        and not mask[r, c]
        and np.isfinite(arr[r, c])
    ):
        return rc
    for rad in range(1, max_radius + 1):
        r0 = max(0, r - rad)
        r1 = min(mask.shape[0], r + rad + 1)
        c0 = max(0, c - rad)
        c1 = min(mask.shape[1], c + rad + 1)
        sub = ~mask[r0:r1, c0:c1]
        if sub.any():
            rr, cc = np.argwhere(sub)[0]
            return (r0 + int(rr), c0 + int(cc))
    raise ValueError("Could not find a valid cell near the provided point.")


def segment_waypoints(
    start_xy: Tuple[float, float], end_xy: Tuple[float, float], segments: int
) -> List[Tuple[float, float]]:
    xs = np.linspace(start_xy[0], end_xy[0], segments + 1)
    ys = np.linspace(start_xy[1], end_xy[1], segments + 1)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def lcp_refined_segmented_along_path(
    src: rasterio.DatasetReader,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    coarse_path_ll: Sequence[Tuple[float, float]],
    raster_crs: CRS,
    refine_buffer_km: float,
    fine_factor: int,
    max_nodes: int,
    fully_connected: bool,
) -> Tuple[List[Tuple[float, float]], float]:
    """Refined LCP segmentation along the coarse path with a true corridor mask.

    Splits the coarse path into segments, builds a geodesic buffer (refine_buffer_km)
    around each segment, and computes LCP within each masked slice at the desired
    fine resolution. This stays close to the coarse route and avoids straight-line
    windows.
    """
    if len(coarse_path_ll) < 2:
        raise ValueError("Coarse path is too short for refinement.")

    # Helper to split indices into N segments
    def split_indices(n: int) -> List[Tuple[int, int]]:
        total = len(coarse_path_ll) - 1
        # guard
        n = max(1, min(n, total))
        idxs = [0]
        for i in range(1, n):
            idxs.append(int(round(i * total / n)))
        idxs.append(total)
        pairs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1) if idxs[i + 1] > idxs[i]]
        return pairs

    # Estimate nodes for a given segment count
    def estimate_nodes(n: int) -> Tuple[int, List[Window], List[Tuple[int, int]]]:
        pairs = split_indices(n)
        windows: List[Window] = []
        max_nodes_est = 0
        for a, b in pairs:
            sub_ll = coarse_path_ll[a : b + 1]
            poly = buffer_line_geodesic(sub_ll, refine_buffer_km, raster_crs)
            minx, miny, maxx, maxy = poly.bounds
            from rasterio.windows import from_bounds as win_from_bounds

            w = win_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            w = w.round_offsets().round_lengths()
            col_off = max(0, int(w.col_off))
            row_off = max(0, int(w.row_off))
            width = max(1, min(int(w.width), src.width - col_off))
            height = max(1, min(int(w.height), src.height - row_off))
            w = Window(col_off, row_off, width, height)
            windows.append(w)
            out_rows = max(1, int(height) // max(1, fine_factor))
            out_cols = max(1, int(width) // max(1, fine_factor))
            nodes = out_rows * out_cols
            max_nodes_est = max(max_nodes_est, nodes)
        return max_nodes_est, windows, pairs

    # Find a segment count that satisfies node budget
    n_segments = 6
    cap = 128
    max_nodes_est, windows, pairs = estimate_nodes(n_segments)
    while max_nodes_est > max_nodes and n_segments < cap:
        n_segments *= 2
        max_nodes_est, windows, pairs = estimate_nodes(n_segments)

    if max_nodes_est > max_nodes:
        print(
            f"Warning: even with {n_segments} segments, per-segment nodes (~{max_nodes_est}) exceed limit {max_nodes}. Proceeding anyway.",
            flush=True,
        )

    # Transformer for lon/lat to raster CRS
    ll2r = Transformer.from_crs(CRS.from_epsg(4326), raster_crs, always_xy=True)

    total_cost = 0.0
    full_coords: List[Tuple[float, float]] = []

    for i, ((a, b), w) in enumerate(zip(pairs, windows), start=1):
        sub_ll = coarse_path_ll[a : b + 1]
        poly = buffer_line_geodesic(sub_ll, refine_buffer_km, raster_crs)

        # Read array for this window at fine resolution
        h = int(w.height)
        wdt = int(w.width)
        if fine_factor > 1:
            out_rows = max(1, h // fine_factor)
            out_cols = max(1, wdt // fine_factor)
            arr, mask, tr = read_cost_surface(src, w, out_shape=(out_rows, out_cols))
        else:
            arr, mask, tr = read_cost_surface(src, w)

        # Apply small length penalty to reduce zigzags
        arr = adjust_costs(arr)

        # Rasterize corridor mask
        inside = rasterize(
            [(poly, 1)],
            out_shape=arr.shape,
            transform=tr,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        combined_mask = mask | (~inside)
        arr[~np.isfinite(arr)] = np.inf
        arr[~inside] = np.inf

        # Start and end points: from last path point or coarse segment endpoint
        if i == 1:
            p0 = start_xy
        else:
            p0 = full_coords[-1]
        end_ll = sub_ll[-1]
        p1 = ll2r.transform(*end_ll)

        sr, sc = rowcol(tr, p0[0], p0[1])
        er, ec = rowcol(tr, p1[0], p1[1])
        s_rc = nearest_valid_cell(arr, combined_mask, (int(sr), int(sc)))
        e_rc = nearest_valid_cell(arr, combined_mask, (int(er), int(ec)))

        print(
            f"Refine seg {i}/{len(pairs)}: grid {arr.shape[1]}x{arr.shape[0]}, start {s_rc}, end {e_rc}",
            flush=True,
        )
        sp = Spinner(f"Computing refined segment {i}/{len(pairs)}")
        sp.start()
        try:
            path_rc, seg_cost = compute_lcp(arr, s_rc, e_rc, fully_connected=fully_connected)
        finally:
            sp.stop()
        total_cost += seg_cost
        seg_coords = [index_to_xy(tr, r, c) for r, c in path_rc]
        if i > 1 and len(seg_coords) > 0:
            seg_coords = seg_coords[1:]
        full_coords.extend(seg_coords)

    return full_coords, total_cost


def index_to_xy(transform_aff: rasterio.Affine, row: int, col: int) -> Tuple[float, float]:
    x, y = transform_aff * (col + 0.5, row + 0.5)
    return x, y


def compute_lcp(
    arr: np.ndarray,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    fully_connected: bool = True,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute least-cost path using scikit-image's route_through_array.

    Returns (list of (row, col) indices, total cost)
    """
    path, cost = route_through_array(
        arr,
        start_rc,
        end_rc,
        fully_connected=fully_connected,
        geometric=True,
    )
    # route_through_array returns a list of (row, col) pairs
    return path, float(cost)


def geodesic_length_wgs84(coords_lonlat: Sequence[Tuple[float, float]]) -> float:
    geod = Geod(ellps="WGS84")
    length_m = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords_lonlat[:-1], coords_lonlat[1:]):
        _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
        length_m += dist
    return length_m


def write_geojson_line(
    coords_xy: Sequence[Tuple[float, float]],
    crs: CRS,
    out_path: Path,
) -> None:
    line = LineString(coords_xy)
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(line),
    }
    # RFC 7946 prefers WGS84, but we also save in raster CRS for inspection
    collection = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collection, f)


def write_vector_line(
    coords_xy: Sequence[Tuple[float, float]],
    crs: CRS,
    out_path: Path,
    driver: str | None = None,
) -> Path:
    """Write a line to a vector file. Tries Fiona drivers else falls back to GeoJSON.

    Returns the actual written path.
    """
    # Determine target driver from extension if not provided
    if driver is None:
        ext = out_path.suffix.lower()
        if ext == ".shp":
            driver = "ESRI Shapefile"
        elif ext == ".gpkg":
            driver = "GPKG"
        elif ext in (".geojson", ".json"):
            driver = "GeoJSON"
        else:
            driver = "GeoJSON"

    if driver == "GeoJSON":
        write_geojson_line(coords_xy, crs, out_path)
        return out_path

    try:
        import fiona  # type: ignore

        schema = {"geometry": "LineString", "properties": {"id": "int"}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Prefer crs_wkt for clarity across Fiona versions
        with fiona.open(
            out_path,
            "w",
            driver=driver,
            schema=schema,
            crs_wkt=crs.to_wkt(),
            layer="lcp" if driver == "GPKG" else None,
        ) as dst:
            dst.write(
                {"geometry": mapping(LineString(coords_xy)), "properties": {"id": 1}}
            )
        return out_path
    except Exception as e:
        # Fallback to GeoJSON
        alt = out_path.with_suffix(".geojson") if out_path.suffix.lower() != ".geojson" else out_path
        write_geojson_line(coords_xy, crs, alt)
        print(
            f"Vector driver '{driver}' unavailable; wrote GeoJSON instead: {alt} ({e})",
            flush=True,
        )
        return alt


def reproject_coords(
    coords_xy: Sequence[Tuple[float, float]],
    src_crs: CRS,
    dst_crs: CRS,
) -> List[Tuple[float, float]]:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs, ys = zip(*coords_xy)
    x2, y2 = transformer.transform(xs, ys)
    return list(zip(x2, y2))


def format_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def run_lcp(
    cost_raster: Path,
    start_lonlat: Tuple[float, float],
    end_lonlat: Tuple[float, float],
    out_dir: Path,
    fully_connected: bool = True,
    plot_png: bool = False,
    max_nodes: int = DEFAULT_MAX_NODES,
    out_format: str = DEFAULT_OUT_FORMAT,
    extras: bool = False,
    refine: bool = True,
    coarse1_factor: int = DEFAULT_COARSE1_FACTOR,
    refine_buffer_km: float = DEFAULT_REFINE_BUFFER_KM,
    fine2_factor: int = DEFAULT_FINE2_FACTOR,
) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Opening cost raster...", flush=True)
    with rasterio.open(cost_raster) as src:
        raster_crs = CRS.from_wkt(src.crs.to_wkt()) if src.crs else None
        if raster_crs is None:
            raise ValueError("Raster has no CRS")

        print(f"Raster size: {src.width} x {src.height}; CRS: {raster_crs.to_string()}")

        # Transform WGS84 lon/lat to raster CRS
        ll2r = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        start_xy = ll2r.transform(*start_lonlat)
        end_xy = ll2r.transform(*end_lonlat)

        # For optional plotting later
        arr_plot = None  # type: ignore
        path_plot_rc = None  # type: ignore

        if refine:
            # Stage 1: coarse full-raster run
            cf1 = max(1, int(coarse1_factor))
            print(f"Refine: Stage 1 coarse run with factor x{cf1} over full raster...", flush=True)
            full_rows = max(1, src.height // cf1)
            full_cols = max(1, src.width // cf1)
            arr1, mask1, tr1 = read_cost_surface(src, window=None, out_shape=(full_rows, full_cols))
            arr1 = adjust_costs(arr1)
            sr1, sc1 = rowcol(tr1, start_xy[0], start_xy[1])
            er1, ec1 = rowcol(tr1, end_xy[0], end_xy[1])
            s1 = nearest_valid_cell(arr1, mask1, (int(sr1), int(sc1)))
            e1 = nearest_valid_cell(arr1, mask1, (int(er1), int(ec1)))
            print(f"Stage 1 grid: {arr1.shape[1]}x{arr1.shape[0]}, start {s1}, end {e1}")
            sp1 = Spinner("Computing coarse global path")
            sp1.start()
            try:
                path1_rc, cost1 = compute_lcp(arr1, s1, e1, fully_connected=fully_connected)
            finally:
                sp1.stop()
            coords1_xy = [index_to_xy(tr1, r, c) for r, c in path1_rc]
            coords1_ll = reproject_coords(coords1_xy, raster_crs, CRS.from_epsg(4326))
            print(f"Stage 1 produced {len(coords1_ll)} vertices; cost ~{cost1:.3f}")

            # Corridor from coarse path
            print(f"Refine: Building {refine_buffer_km:.0f} km corridor around coarse path...", flush=True)
            corridor_poly = buffer_line_geodesic(coords1_ll, float(refine_buffer_km), raster_crs)
            minx, miny, maxx, maxy = corridor_poly.bounds
            from rasterio.windows import from_bounds as win_from_bounds
            window = win_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.round_offsets().round_lengths()
            col_off = max(0, int(window.col_off))
            row_off = max(0, int(window.row_off))
            width = max(0, min(int(window.width), src.width - col_off))
            height = max(0, min(int(window.height), src.height - row_off))
            window = Window(col_off, row_off, width, height)
            print(
                f"Refine window: col_off={int(window.col_off)}, row_off={int(window.row_off)}, "
                f"width={int(window.width)}, height={int(window.height)}",
                flush=True,
            )

            # Final run at native resolution (no downsampling per requirements)
            f2 = 1
            out_rows = max(1, height // f2)
            out_cols = max(1, width // f2)
            est_nodes2 = out_rows * out_cols
            if est_nodes2 > max_nodes:
                print(
                    f"Refine: corridor grid {out_cols}x{out_rows} (~{est_nodes2} nodes) exceeds limit {max_nodes}. "
                    f"Falling back to segmented LCP following coarse path with a {refine_buffer_km:.0f} km corridor.",
                    flush=True,
                )
                coords_xy, total_cost = lcp_refined_segmented_along_path(
                    src=src,
                    start_xy=start_xy,
                    end_xy=end_xy,
                    coarse_path_ll=coords1_ll,
                    raster_crs=raster_crs,
                    refine_buffer_km=float(refine_buffer_km),
                    fine_factor=f2,
                    max_nodes=max_nodes,
                    fully_connected=fully_connected,
                )
                # Proceed to output with coords_xy and skip the single-grid run
                path_rc = None  # placeholder to skip PNG plotting later
            else:
                if f2 > 1:
                    print(
                        f"Refine: reading corridor at x{f2} downsample to {out_cols}x{out_rows}...",
                        flush=True,
                    )
                    arr2, mask2, tr2 = read_cost_surface(src, window, out_shape=(out_rows, out_cols))
                else:
                    print("Refine: reading corridor at native resolution...", flush=True)
                    arr2, mask2, tr2 = read_cost_surface(src, window)
                arr2 = adjust_costs(arr2)
                inside = rasterize(
                    [(corridor_poly, 1)],
                    out_shape=arr2.shape,
                    transform=tr2,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)

                combined_mask = mask2 | (~inside)
                arr2[~np.isfinite(arr2)] = np.inf
                arr2[~inside] = np.inf

            if est_nodes2 <= max_nodes:
                sr2, sc2 = rowcol(tr2, start_xy[0], start_xy[1])
                er2, ec2 = rowcol(tr2, end_xy[0], end_xy[1])
                s2 = nearest_valid_cell(arr2, combined_mask, (int(sr2), int(sc2)))
                e2 = nearest_valid_cell(arr2, combined_mask, (int(er2), int(ec2)))
                print(f"Stage 2 grid: {arr2.shape[1]}x{arr2.shape[0]}, start {s2}, end {e2}")
                sp2 = Spinner("Computing refined path")
                sp2.start()
                try:
                    path_rc, total_cost = compute_lcp(arr2, s2, e2, fully_connected=fully_connected)
                finally:
                    sp2.stop()
                print(f"Refine: computed path with {len(path_rc)} pixels; cost ~{total_cost:.3f}")
                coords_xy = [index_to_xy(tr2, r, c) for r, c in path_rc]
                arr_plot = arr2
                path_plot_rc = path_rc
        else:
            # Compute processing window with buffer
            print(
                f"Computing window around path with buffer ~{buffer_km:.0f} km...",
                flush=True,
            )
            window = compute_window_for_points(src, start_xy, end_xy, buffer_km=buffer_km)
            print(
                f"Window: col_off={int(window.col_off)}, row_off={int(window.row_off)}, "
                f"width={int(window.width)}, height={int(window.height)}",
                flush=True,
            )

            # Determine coarse factor if needed to keep memory in check
            w, h = int(window.width), int(window.height)
            if coarse_factor is None:
                # Choose factor so that (w/f)*(h/f) <= max_nodes, but cap at 4 per user requirement
                f_est = int(math.ceil(math.sqrt((w * h) / float(max_nodes))))
                coarse_factor_eff = min(4, max(1, f_est))
            else:
                coarse_factor_eff = min(4, max(1, int(coarse_factor)))
                if int(coarse_factor) != coarse_factor_eff:
                    print(f"Capping coarse factor to {coarse_factor_eff} (max allowed is 4)")

            # Read cost array (possibly downsampled)
            # Check node budget; if too large even at cap factor 4, segment the path
            est_nodes = (w // coarse_factor_eff) * (h // coarse_factor_eff)
            if est_nodes > max_nodes:
                print(
                    f"Single window nodes {est_nodes} exceed limit {max_nodes} at factor x{coarse_factor_eff}; using segmented LCP.",
                    flush=True,
                )
                coords_xy, total_cost = lcp_segmented(
                    src,
                    start_xy,
                    end_xy,
                    buffer_km,
                    fully_connected,
                    coarse_factor_eff,
                    max_nodes,
                )
            else:
                # Read cost array (possibly downsampled)
                if coarse_factor_eff > 1:
                    out_rows = max(1, h // coarse_factor_eff)
                    out_cols = max(1, w // coarse_factor_eff)
                    print(
                        f"Reading cost surface downsampled by x{coarse_factor_eff} to {out_cols}x{out_rows}...",
                        flush=True,
                    )
                    arr, mask, out_transform = read_cost_surface(src, window, out_shape=(out_rows, out_cols))
                else:
                    print("Reading cost surface at full resolution...", flush=True)
                    arr, mask, out_transform = read_cost_surface(src, window)
                arr = adjust_costs(arr)

                # Compute start/end in window indices (possibly on downsampled grid)
                sr, sc = rowcol(out_transform, start_xy[0], start_xy[1])
                er, ec = rowcol(out_transform, end_xy[0], end_xy[1])
                start_rc = nearest_valid_cell(arr, mask, (int(sr), int(sc)))
                end_rc = nearest_valid_cell(arr, mask, (int(er), int(ec)))
                print(f"Start idx: {start_rc}; End idx: {end_rc}; Grid: {arr.shape[1]}x{arr.shape[0]}")

                spinner = Spinner("Computing least-cost path")
                spinner.start()
                try:
                    path_rc, total_cost = compute_lcp(arr, start_rc, end_rc, fully_connected=fully_connected)
                finally:
                    spinner.stop()
                print(f"Computed path with {len(path_rc)} pixels. Total cost: {total_cost:.3f}")

                # Convert indices to raster CRS coordinates
                coords_xy = [index_to_xy(out_transform, r, c) for r, c in path_rc]
                arr_plot = arr
                path_plot_rc = path_rc

        name = "lcp_tarifa_portbou"
        dst_ext = ".geojson" if out_format.lower() == "geojson" else (".shp" if out_format.lower() == "shp" else ".gpkg")

        # Reproject to WGS84 and save only the primary output unless extras are requested
        coords_lonlat = reproject_coords(coords_xy, raster_crs, CRS.from_epsg(4326))
        out_vec_ll = out_dir / f"{name}{dst_ext}"
        written_ll = write_vector_line(coords_lonlat, CRS.from_epsg(4326), out_vec_ll)
        print(f"Wrote path: {written_ll}")

        if extras:
            # Optional: also write raster CRS version
            out_vec_src = out_dir / f"{name}_raster_crs{dst_ext}"
            written_src = write_vector_line(coords_xy, raster_crs, out_vec_src)
            print(f"Wrote path (raster CRS): {written_src}")

        # Compute length
        if raster_crs.is_projected:
            # Euclidean length in projected units (assume meters)
            length_m = 0.0
            for (x1, y1), (x2, y2) in zip(coords_xy[:-1], coords_xy[1:]):
                dx = x2 - x1
                dy = y2 - y1
                length_m += math.hypot(dx, dy)
        else:
            length_m = geodesic_length_wgs84(coords_lonlat)

        # Always report final total cost
        try:
            print(f"Final total cost: {total_cost:.6f}")
        except Exception:
            pass

        # Optional summary only when extras=True
        if extras:
            summary_path = out_dir / f"{name}_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "start_lonlat": start_lonlat,
                        "end_lonlat": end_lonlat,
                        "pixels_in_path": len(coords_xy),
                        "total_cost": total_cost,
                        "approx_length_m": length_m,
                        "raster": str(cost_raster),
                        "window": {
                            "col_off": int(window.col_off),
                            "row_off": int(window.row_off),
                            "width": int(window.width),
                            "height": int(window.height),
                        },
                        "crs": raster_crs.to_string(),
                    "out_format": out_format,
                    "refine": refine,
                    "coarse1_factor": coarse1_factor,
                    "refine_buffer_km": refine_buffer_km,
                    "fine2_factor": fine2_factor,
                },
                f,
                indent=2,
            )
            print(f"Wrote summary: {summary_path}")

        # Optional plot
        if plot_png and extras and (arr_plot is not None) and (path_plot_rc is not None):
            try:
                import matplotlib.pyplot as plt

                # Downsample background for performance if huge
                bg = arr_plot.copy()
                # Replace inf with nan for plotting
                bg[~np.isfinite(bg)] = np.nan

                plt.figure(figsize=(10, 10))
                plt.imshow(bg, cmap="gray", interpolation="nearest")
                ys = [r for r, c in path_plot_rc]
                xs = [c for r, c in path_plot_rc]
                plt.plot(xs, ys, color="cyan", linewidth=1.5)
                # start/end scatter uses array indices when available; skip for simplicity
                plt.gca().invert_yaxis()
                plt.title("Least-cost path (array indices)")
                plt.tight_layout()
                out_png = out_dir / f"{name}.png"
                plt.savefig(out_png, dpi=200)
                plt.close()
                print(f"Wrote preview PNG: {out_png}")
            except Exception as e:
                print(f"Skipping PNG preview (Matplotlib not available or failed): {e}")

    t1 = time.time()
    print(f"Done in {format_hms(t1 - t0)}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute least-cost path between two points using a cost raster "
            "(Tarifa -> Portbou by default)."
        )
    )
    p.add_argument(
        "--cost-raster",
        default=str(DEFAULT_COST_RASTER),
        help="Path to cost raster (GeoTIFF).",
    )
    p.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        default=list(START_POINT),
        help="Start point lon lat in WGS84 (default from script).",
    )
    p.add_argument(
        "--end",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        default=list(END_POINT),
        help="End point lon lat in WGS84 (default from script).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for results (GeoJSON, PNG, summary).",
    )
    # Plotting is off by default; enable via --plot when --extras
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG preview (used when --extras).",
    )
    p.add_argument(
        "--four-neigh",
        action="store_true",
        help="Use 4-neighbour connectivity instead of 8.",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help="Node budget for memory control.",
    )
    p.add_argument(
        "--out-format",
        choices=["geojson", "shp", "gpkg"],
        default=DEFAULT_OUT_FORMAT,
        help="Vector output format (default: gpkg).",
    )
    p.add_argument(
        "--extras",
        action="store_true",
        help="Also write raster-CRS path, summary JSON, and optional PNG.",
    )
    # Refine (coarse-to-fine) options
    p.add_argument(
        "--refine",
        action="store_true",
        default=True,
        help="Two-stage LCP: coarse global pass then refined corridor pass (default on).",
    )
    p.add_argument(
        "--coarse1-factor",
        type=int,
        default=DEFAULT_COARSE1_FACTOR,
        help="Stage 1 global downsampling factor (default: 10).",
    )
    p.add_argument(
        "--refine-buffer-km",
        type=float,
        default=DEFAULT_REFINE_BUFFER_KM,
        help="Corridor buffer (km) around the coarse path (default: 40).",
    )
    p.add_argument(
        "--fine2-factor",
        type=int,
        choices=[1, 2],
        default=DEFAULT_FINE2_FACTOR,
        help="Stage 2 corridor downsampling factor (default 1=native).",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cost_raster = Path(args.cost_raster)
    if not cost_raster.exists():
        raise FileNotFoundError(f"Cost raster not found: {cost_raster}")

    out_dir = Path(args.output_dir)

    start = (float(args.start[0]), float(args.start[1]))
    end = (float(args.end[0]), float(args.end[1]))

    run_lcp(
        cost_raster=cost_raster,
        start_lonlat=start,
        end_lonlat=end,
        out_dir=out_dir,
        fully_connected=not args.four_neigh,
        plot_png=args.plot,
        max_nodes=args.max_nodes,
        out_format=args.out_format,
        extras=args.extras,
        refine=args.refine,
        coarse1_factor=args.coarse1_factor,
        refine_buffer_km=float(args.refine_buffer_km),
        fine2_factor=int(args.fine2_factor),
    )


if __name__ == "__main__":
    main()
