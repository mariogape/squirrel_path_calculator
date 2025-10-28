# Squirrel Path Calculator

This project playfully estimates the “squirrel route” — a fictional path a squirrel could take to cross a landscape from tree to tree without touching the ground. It’s inspired by the old Spanish myth that a squirrel could traverse the Iberian Peninsula through continuous forest. Technically, the repository computes a least‑cost path (LCP) between two points over a forest/non‑forest cost map.

- Input cost raster example (uint8 GeoTIFF): `data/processed/cost_Forest_height_2019_NAFR.tif`
  - Convention: 0 = preferred (forest/trees), 1 = avoided (non‑forest/open). NoData/invalid are barriers.
- Output: the path as a single line in `outputs/lcp_tarifa_portbou.gpkg` (WGS84).

## What’s Inside
- `scripts/least_cost_path.py` — main LCP tool. Two‑stage pipeline (coarse → refine) designed to be fast and memory‑safe, with plug‑and‑play defaults.
- `scripts/reclassify_forest_cost.py` — helper to build a 0/1 cost raster from a forest height raster (with progress, cropping, and NoData handling).
- `data/processed/` — place your cost raster here.
- `outputs/` — results are written here.

## Beginner‑Friendly: How To Use
1) Install Python packages (example with pip):

```
pip install numpy rasterio shapely scikit-image pyproj tqdm matplotlib
```

2) Create or provide a cost raster

Important: If you are using the GLAD forest height dataset (2019), run the reclassify script first to generate the cost raster. In this project we built the cost raster from GLAD 2019 and removed all trees below 3 m (threshold = 3).

- Option A: Build from a forest height raster using the reclassify script.
  - Put your forest height GeoTIFF at `data/raw/Forest_height_2019_NAFR.tif`.
  - Run:

    ```
    python scripts/reclassify_forest_cost.py --full
    ```

  - What it does:
    - Encodes 0/1 costs: height >= threshold (default 3 m) -> forest (0); lower (and special codes water=101, snow/ice=102) -> non-forest (1). NoData (103) can be masked or treated as 1.
    - Cropping: default Iberia; use `--full` for full extent, or `--bounds MINX MINY MAXX MAXY` with `--bounds-crs`.
    - Writes to `data/processed/cost_Forest_height_2019_NAFR.tif` and shows progress.
    - Example used in this repo: GLAD forest height (2019) with threshold 3 m (trees below 3 m treated as non-forest).

- Option B: Use your own cost raster (0 = preferred, 1 = avoided, NoData = barrier). Place it under `data/processed/` and reference it with `--cost-raster` if needed.

3) Compute the squirrel path (no flags required):

```
python scripts/least_cost_path.py
```

Note: The LCP script can use any cost raster that follows the convention 0=preferred, 1=avoided, NoData=barrier. Pass a custom path with `--cost-raster`.

You’ll see progress logs. The script:
- Runs a coarse global LCP on a downsampled raster to capture the overall route.
- Builds a geodesic corridor around that coarse path.
- Runs a final LCP at native resolution inside the corridor. If that corridor is still too large for memory, it is processed in segments along the coarse route and stitched.
- Saves `outputs/lcp_tarifa_portbou.gpkg` and prints the final total cost.

## Adjusting the Route (Easy Tweaks)
Open `scripts/least_cost_path.py` and edit the tunables near the top:
- Start/end points: `START_POINT`, `END_POINT` (lon/lat WGS84). See `scripts/least_cost_path.py:30`.
- Coarse downsample: `DEFAULT_COARSE1_FACTOR` (lower = slower but more global fidelity). See `scripts/least_cost_path.py:34`.
- Refine corridor: `DEFAULT_REFINE_BUFFER_KM` in km. See `scripts/least_cost_path.py:35`.
- Final factor: `DEFAULT_FINE2_FACTOR` (1=native, 2=half) for the refined pass. See `scripts/least_cost_path.py:36`.
- Node budget: `DEFAULT_MAX_NODES` to control memory/segmentation. See `scripts/least_cost_path.py:37`.
- Output format: `DEFAULT_OUT_FORMAT` (gpkg|geojson|shp). See `scripts/least_cost_path.py:38`.

Prefer CLI? Every tunable has a flag (e.g., `--coarse1-factor`, `--refine-buffer-km`, `--fine2-factor`, `--max-nodes`, `--out-format`, `--start`, `--end`). Example:

```
python scripts/least_cost_path.py \
  --cost-raster data/processed/cost_Forest_height_2019_NAFR.tif \
  --start LON LAT --end LON LAT \
  --coarse1-factor 10 --refine-buffer-km 40 --fine2-factor 1 \
  --max-nodes 18000000 --out-format gpkg --extras --plot
```

## Reclassify Script Details
- File: `scripts/reclassify_forest_cost.py`
- Purpose: Convert a forest height raster into a binary cost map for LCP.
- Key options:
  - `input` (positional): path to forest height GeoTIFF (default expects `data/raw/Forest_height_2019_NAFR.tif`).
  - `-o/--output`: output path (default writes to `data/processed/cost_*.tif`).
  - `-t/--threshold`: forest threshold in meters (default 3 m).
  - `--mask-nodata`: mark 103 (and source NoData) as NoData in output; otherwise count them as cost 1.
  - `--compress`: output compression (default LZW).
  - Cropping: `--full` for full extent; or `--bounds MINX MINY MAXX MAXY` with `--bounds-crs` (default EPSG:4326).
  - `--no-progress`: disable progress output.
- Output encoding (uint8): 0 (forest), 1 (non‑forest), optional NoData.
 - Typical input used here: GLAD forest height dataset (2019), with all trees below 3 m cut (threshold = 3).

## Outputs
- Main: `outputs/lcp_tarifa_portbou.gpkg` — final path (WGS84), LineString, layer `lcp`.
- With `--extras` (debug):
  - `outputs/lcp_tarifa_portbou_raster_crs.*` — path in raster CRS
  - `outputs/lcp_tarifa_portbou_summary.json` — settings, total cost, length, corridor window
  - `outputs/lcp_tarifa_portbou.png` — quick preview

## How It Works (Under the Hood)
- Coarse LCP (full extent) → corridor (geodesic buffer) → refined LCP (native resolution within corridor).
- If the corridor won’t fit in memory, it’s split along the coarse path (powers of two, capped) and stitched — still inside the same corridor.
- A tiny length‑penalty epsilon is added to valid costs to reduce zigzags in uniform forest while preserving a strong 0 vs 1 preference.

## Tips & Troubleshooting
- Path too jagged: widen the corridor slightly or increase the epsilon (see `adjust_costs`).
- Missing a detour: increase `DEFAULT_REFINE_BUFFER_KM` and/or `DEFAULT_MAX_NODES`.
- Many segments (e.g., 256): native corridor is huge; raise `--max-nodes` or try `--fine2-factor 2`.
- Slow installs: prefer prebuilt wheels for `rasterio`/`shapely` on Windows/macOS.
