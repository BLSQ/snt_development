import re
import rasterio
import numpy as np
from pathlib import Path
from affine import Affine


def get_ou_level_from_string(adm2_name: str) -> int:
    """Extract the organizational unit level from a column name formatted as 'level_<NUMBER>_*'.

    Returns:
        int: The extracted organizational unit level.
    """
    match = re.match(r"level_(\d+)_", adm2_name)
    if not match:
        raise ValueError(f"Expected format 'level_<NUMBER>_*', received: {adm2_name}")

    return int(match.group(1))


def load_tiff_bands(
    tif_file: str | Path, band_names: list[str] | None = None
) -> tuple[dict[str, np.ndarray], Affine, str, float | None]:
    """Read a multi-band TIFF and return raster data along with metadata.

    Args:
        tif_file: Path to the TIFF file.
        band_names: Optional list of band names. If not provided, defaults to 'band_1', 'band_2', etc.

    Returns:
        Tuple containing:
            - bands: dict mapping band name -> 2D array
            - transform: affine transform of the raster
            - crs: coordinate reference system
            - nodata: nodata value
    """
    tif_path = Path(tif_file)
    if not tif_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tif_file}")

    bands = {}

    try:
        with rasterio.open(tif_path) as src:
            count = src.count
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            for idx in range(1, count + 1):
                if band_names and idx <= len(band_names):
                    name = band_names[idx - 1]
                else:
                    name = f"band_{idx}"
                bands[name] = src.read(idx)

    except rasterio.errors.RasterioIOError as e:
        raise RuntimeError(f"Failed to read TIFF file: {tif_file}") from e

    return bands, transform, crs, nodata


def parse_raster_filename_vars(raster_file: Path) -> dict:
    """Parse a raster filename into its components: category, version, region, indicator, year.

    Expected filename pattern:
        <Category>__<Version>_<Region>_<Indicator>_<Year>.tif
        Example: Malaria__202508_Global_Pf_Parasite_Rate_2020.tif

    Args:
        raster_file: Path object pointing to the raster file.

    Returns:
        A dictionary with keys: category, version, region, indicator, year.

    Raises:
        ValueError: if the filename does not match the expected pattern.
    """
    pattern = r"([A-Za-z]+)__([0-9]{6})_(Global|Africa)_(.+?)_(\d{4})\.tif$"
    m = re.search(pattern, raster_file.name)
    if not m:
        raise ValueError(f"Cannot parse raster filename: {raster_file.name}")

    return {
        "category": m.group(1),
        "version": m.group(2),
        "region": m.group(3),
        "indicator": m.group(4),
        "year": m.group(5),
    }
