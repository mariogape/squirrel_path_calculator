import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds, transform as window_transform, Window
from rasterio.warp import transform_bounds

# Default paths (edit these if needed)
DEFAULT_INPUT_PATH = Path("data/raw/Forest_height_2019_NAFR.tif")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def reclassify_to_cost(
    input_raster: str,
    output_raster: str,
    threshold_m: int = 3,
    mask_no_data: bool = False,
    output_nodata_value: int = 255,
    compress: str = "LZW",
    bounds: tuple | None = None,
    bounds_crs: str = "EPSG:4326",
    use_full_extent: bool = False,
    show_progress: bool = True,
):
    """Reclassify GEDI/forest height raster into a cost raster.

    Rules:
    - Forest (cost 0): pixels with height >= threshold (and <= 60 when heights are encoded 0-60)
    - Non-forest (cost 1): pixels with height < threshold, water (101), snow/ice (102)
    - No data (103): by default treated as non-forest (1); can be masked if mask_no_data=True

    Parameters
    ----------
    input_raster : str
        Path to input GeoTIFF with encoded values: 0-60 heights, 101 water, 102 snow/ice, 103 no data.
    output_raster : str
        Path to write the output cost GeoTIFF (uint8).
    threshold_m : int, optional
        Height threshold in meters to consider forest, by default 3.
    mask_no_data : bool, optional
        If True, pixels with value 103 (and source NoData) are set to output_nodata_value and marked as NoData in output.
        If False, they will be treated as non-forest cost 1. Default False.
    output_nodata_value : int, optional
        NoData value to use in the output when mask_no_data=True. Default 255.
    compress : str, optional
        Compression for the output GeoTIFF. Default "LZW".
    """

    input_path = Path(input_raster)
    output_path = Path(output_raster)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env():
        with rasterio.open(input_path) as src:
            if src.count != 1:
                raise ValueError("Input raster must be single-band.")

            src_nodata = src.nodata
            profile = src.profile.copy()

            # Determine processing window (crop)
            if use_full_extent:
                crop_window = Window(0, 0, src.width, src.height)
            else:
                if bounds is None:
                    # Default: Iberian Peninsula in WGS84 (approx)
                    bounds = (-10.0, 35.0, 4.0, 44.5)
                    bounds_crs = "EPSG:4326"
                # Transform requested bounds into source CRS
                b_src = transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
                crop_window = window_from_bounds(*b_src, transform=src.transform)
                # Align window to pixel grid and clip to image
                crop_window = crop_window.round_offsets().round_lengths()
                # Clip to raster extent
                row_off = max(0, int(crop_window.row_off))
                col_off = max(0, int(crop_window.col_off))
                height = max(0, min(int(crop_window.height), src.height - row_off))
                width = max(0, min(int(crop_window.width), src.width - col_off))
                crop_window = Window(col_off, row_off, width, height)

                if width <= 0 or height <= 0:
                    raise ValueError("Crop window outside raster extent; check bounds/CRS.")

            # Output dimensions and transform for the crop
            out_transform = window_transform(crop_window, src.transform)
            out_width, out_height = int(crop_window.width), int(crop_window.height)

            # Choose safe tiled block sizes (multiples of 16) based on output size
            def _valid_block(size: int) -> int:
                target = min(256, size)
                block = max(16, (target // 16) * 16)
                return min(block, size)

            bx = _valid_block(out_width)
            by = _valid_block(out_height)
            use_tiled = bx >= 16 and by >= 16

            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress=compress,
                tiled=use_tiled,
                transform=out_transform,
                width=out_width,
                height=out_height,
            )
            if use_tiled:
                profile.update(blockxsize=bx, blockysize=by)

            if mask_no_data:
                profile.update(nodata=output_nodata_value)
            else:
                # Remove nodata from profile to avoid accidental masking
                profile.update(nodata=None)

            # Progress setup
            total_tiles_x = (out_width + bx - 1) // bx
            total_tiles_y = (out_height + by - 1) // by
            total_tiles = max(1, total_tiles_x * total_tiles_y)
            use_tqdm = False
            pbar = None
            if show_progress:
                try:
                    from tqdm import tqdm  # type: ignore

                    pbar = tqdm(total=total_tiles, desc="Reclassifying", unit="tile")
                    use_tqdm = True
                except Exception:
                    use_tqdm = False
            printed_pct = -1

            with rasterio.open(output_path, "w", **profile) as dst:
                # Iterate over tiles within the crop window
                src_row0 = int(crop_window.row_off)
                src_col0 = int(crop_window.col_off)

                tiles_done = 0
                for y in range(0, out_height, by):
                    tile_h = min(by, out_height - y)
                    for x in range(0, out_width, bx):
                        tile_w = min(bx, out_width - x)

                        # Source read window
                        read_window = Window(src_col0 + x, src_row0 + y, tile_w, tile_h)
                        data = src.read(1, window=read_window)

                        # Initialize as non-forest cost (1)
                        out = np.ones(data.shape, dtype=np.uint8)

                        # Forest where height >= threshold and <= 60
                        forest_mask = (data >= threshold_m) & (data <= 60)
                        out[forest_mask] = 0

                        # No data (103) handling and propagate source NoData
                        nodata_code_mask = data == 103
                        src_nodata_mask = np.zeros_like(nodata_code_mask)
                        if src_nodata is not None:
                            src_nodata_mask = data == src_nodata

                        if mask_no_data:
                            out[nodata_code_mask | src_nodata_mask] = output_nodata_value
                        else:
                            out[nodata_code_mask | src_nodata_mask] = 1

                        # Destination write window is tile position within output
                        write_window = Window(x, y, tile_w, tile_h)
                        dst.write(out, 1, window=write_window)

                        # Progress update
                        tiles_done += 1
                        if show_progress:
                            if use_tqdm and pbar is not None:
                                pbar.update(1)
                            else:
                                pct = int(tiles_done * 100 / total_tiles)
                                if pct != printed_pct and pct % 5 == 0:
                                    print(f"Progress: {pct}% ({tiles_done}/{total_tiles} tiles)")
                                    printed_pct = pct

            if use_tqdm and pbar is not None:
                pbar.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reclassify a GEDI/forest height raster into a cost map where forest=0 and non-forest=1."
        )
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Path to input GeoTIFF. Expected encoding: 0-60 heights (meters), 101 water, 102 snow/ice, 103 no data."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        help="Path to output cost GeoTIFF (uint8).",
    )
    p.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=3,
        help="Height threshold in meters to consider forest (default: 3).",
    )
    p.add_argument(
        "--mask-nodata",
        action="store_true",
        help=(
            "If set, output pixels with 103 (and source NoData) are set to output NoData value instead of cost=1."
        ),
    )
    p.add_argument(
        "--output-nodata",
        type=int,
        default=255,
        help="Output NoData value when --mask-nodata is used (default: 255).",
    )
    p.add_argument(
        "--compress",
        default="LZW",
        help="Output GeoTIFF compression (default: LZW).",
    )
    # Region of interest options
    p.add_argument(
        "--full",
        action="store_true",
        help="Process full raster extent (ignore default Iberia crop).",
    )
    p.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Optional bounding box to crop [in --bounds-crs]. Overrides --full."
        " Default CRS EPSG:4326.",
    )
    p.add_argument(
        "--bounds-crs",
        default="EPSG:4326",
        help="CRS of --bounds (default: EPSG:4326).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output.",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve input path: CLI value or default constant
    input_path = args.input if args.input else str(DEFAULT_INPUT_PATH)
    if args.output:
        output_path = args.output
    else:
        in_name = Path(input_path).stem
        output_path = str(DEFAULT_OUTPUT_DIR / f"cost_{in_name}.tif")

    reclassify_to_cost(
        input_raster=input_path,
        output_raster=output_path,
        threshold_m=args.threshold,
        mask_no_data=args.mask_nodata,
        output_nodata_value=args.output_nodata,
        compress=args.compress,
        bounds=tuple(args.bounds) if args.bounds is not None else None,
        bounds_crs=args.bounds_crs,
        use_full_extent=args.full,
        show_progress=not args.no_progress,
    )

    print(f"Wrote cost raster: {output_path}")


if __name__ == "__main__":
    main()
