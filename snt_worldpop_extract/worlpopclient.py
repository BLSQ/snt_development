import requests
from pathlib import Path
import rasterio
from rasterio.enums import Compression


class WorldPopClient:
    """Mini client for the WorldPop REST API.

    Source: https://www.worldpop.org/rest/data/
    """

    def __init__(
        self,
        url: str = "https://www.worldpop.org/rest/data",
        project_alias: str = "pop",
        subproject: str = "wpgp",
    ):
        """Initialize the client with a specific project alias and subproject.

        NOTE: For the moment we only point to population data /pop.

        Parameters
        ----------
        url : str
            The base URL for the WorldPop API.
        project_alias : str
            The main project alias (e.g., 'pop', 'births', 'pregnancies').
        subproject : str
            The subproject under the main project (e.g., 'wpgp').
        """
        self.base_url = f"{url}/{project_alias}/{subproject}"
        self._country_cache = {}  # Cache to store datasets per country

    @staticmethod
    def _matches(d: dict, keyword: str) -> bool:
        keyword_lower = keyword.lower()
        fields = [
            d.get("title", ""),
            d.get("description", ""),
        ]
        for field in fields:
            if isinstance(field, list):
                field_str = " ".join(field)
            else:
                field_str = field
            if keyword_lower in field_str.lower():
                return True
        return False

    def _fetch_country_datasets_metadata(self, country_iso3: str) -> list:
        """Fetch datasets for a given country ISO3 code and store in cache.

        Parameters
        ----------
        country_iso3 : str
            The ISO3 country code (e.g., 'COD', 'BFA').

        Returns
        -------
        list
            A list of datasets metadata for the specified country.
        """
        iso3_upper = country_iso3.upper()
        if iso3_upper not in self._country_cache:
            url = f"{self.base_url}?iso3={iso3_upper}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            self._country_cache[iso3_upper] = data.get("data", [])
        return self._country_cache[iso3_upper]

    def get_datasets_by_country(self, country_iso3: str) -> list:
        """Return datasets for a given country ISO3 code.

        Parameters
        ----------
        country_iso3 : str
            The ISO3 country code (e.g., 'COD', 'BFA').

        Returns
        -------
        list
            A list of datasets metadata for the specified country.
        """
        return self._fetch_country_datasets_metadata(country_iso3)

    def search_datasets(self, country_iso3: str, keyword: str) -> list:
        """Search datasets by ISO3 and keyword in title and description.

        Parameters
        ----------
        country_iso3 : str
            The ISO3 country code (e.g., 'COD', 'BFA').
        keyword : str
            Keyword to filter datasets by title.

        Returns
        -------
        list
            A list of datasets metadata that match the keyword.
        """
        datasets = self._fetch_country_datasets_metadata(country_iso3)

        if not keyword:
            return datasets

        return [d for d in datasets if self._matches(d, keyword)]

    def get_population_geotiff(self, country_iso3: str, year: int, output_dir: Path, fname: str) -> Path:
        """Download and save the WorldPop raster dataset (GeoTIFF) for a given country and year.

        This method retrieves a population raster dataset from the WorldPop API
        and saves the original GeoTIFF file to the specified directory.

        Parameters
        ----------
        country_iso3 : str
            ISO3 code of the country (e.g., "COD", "BFA").
        year : int
            Year to filter the dataset (e.g., 2020).
        output_dir : Path
            Directory to save the GeoTIFF file.
        fname : str
            Filename for the saved GeoTIFF file (e.g., "wpop_COD_pop_2020.tif").

        Returns
        -------
            str: Full path to the saved GeoTIFF file.

        Raises
        ------
            ValueError: If no matching dataset or file is found for the given country and year.
            requests.HTTPError: If the file download fails.
        """
        datasets = self._fetch_country_datasets_metadata(country_iso3)
        if not datasets:
            raise ValueError(f"No datasets found for country {country_iso3.upper()}.")

        year_str = str(year)
        # Filter for datasets mentioning the year
        matching = [d for d in datasets if self._matches(d, year_str)]

        if not matching:
            raise ValueError(f"No dataset for year {year} in country {country_iso3.upper()}.")

        files = matching[0].get("files", [])
        if not files:
            raise ValueError(f"No files available in the dataset for {country_iso3.upper()} {year}.")

        raster_url = files[0]
        response = requests.get(raster_url, stream=True)
        response.raise_for_status()

        tif_path = output_dir / fname

        try:
            with Path.open(tif_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        f.write(chunk)
        except OSError as e:
            raise OSError(f"Failed to write raster file to {tif_path}: {e}") from e

        return tif_path

    @staticmethod
    def compress_geotiff(
        src_path: Path,
        dst_path: Path,
        compression: Compression = Compression.deflate,
        predictor: int = 2,
        blockxsize: int = 512,
        blockysize: int = 512,
    ) -> Path:
        """Read a GeoTIFF and write out a compressed, tiled GeoTIFF (COG style).

        Parameters
        ----------
        src_path : Path
            Path to the source GeoTIFF file.
        dst_path : Path
            Path where the compressed GeoTIFF will be saved.
        compression : Compression, optional
            Compression algorithm (e.g., `Compression.deflate`, `Compression.lzw`, `Compression.zstd`),
            by default `Compression.deflate`.
        predictor : int, optional
            Predictor type (2 is best for continuous data), by default 2.
        blockxsize : int, optional
            Tile width in pixels, by default 512.
        blockysize : int, optional
            Tile height in pixels, by default 512.

        Returns
        -------
        Path
            The `dst_path` of the newly written compressed GeoTIFF.

        Raises
        ------
        RuntimeError
            If reading or writing fails.
        """
        profile = None

        try:
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
                profile.update(
                    compress=compression,
                    predictor=predictor,
                    tiled=True,
                    blockxsize=blockxsize,
                    blockysize=blockysize,
                )

                with rasterio.open(dst_path, "w", **profile) as dst:
                    for band in range(1, src.count + 1):
                        data = src.read(band)
                        dst.write(data, band)
        except Exception as e:
            raise RuntimeError(f"Failed to compress {src_path.name} → {dst_path.name}: {e}") from e

        return dst_path
