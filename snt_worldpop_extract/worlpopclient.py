from pathlib import Path

import requests


class WorldPopClient:
    """Mini client for the WorldPop REST API.

    Source: https://data.worldpop.org/GIS/Population
    """

    def __init__(self, url: str = "https://data.worldpop.org/GIS/Population/Global_2000_2020"):
        """Initialize the client with a specific project alias and subproject.

        NOTE: For the moment we only point to population data /pop.

        Parameters
        ----------
        url : str
            The base URL for the WorldPop data download.
        """
        self.base_url = url

    def download_data_for_country(
        self,
        country_iso3: str,
        output_dir: Path,
        fname: str | None = None,
        year: str = "2020",
        un_adj: bool = False,
    ) -> Path:
        """Download and save the WorldPop raster dataset for a given country and year.

        This operation is atomic. A partial download will not result in a corrupt
        final file.

        Parameters
        ----------
        country_iso3 : str
            3-letter ISO code of the country (e.g., "COD", "BFA").
        output_dir : Path
            Directory to save the GeoTIFF file.
        fname : str, optional
            Filename to save the raster data. If None, defaults to
            "{country_iso3}_worldpop_population_{year}.tif".
        year : str
            Year to filter the dataset (e.g., "2020").
        un_adj : bool
            Whether to download the "unadjuvanted" (constrained) version. Defaults to False.

        Returns
        -------
        Path
            Full path to the saved GeoTIFF file.

        Raises
        ------
        ValueError
            If the country_iso3 code is invalid.
        IOError
            If the file download or disk write fails.
        """
        if not (isinstance(country_iso3, str) and len(country_iso3) == 3):
            raise ValueError("country_iso3 must be a 3-letter string.")

        country_iso3 = country_iso3.upper()
        try:
            raster_url = self._build_url(country_iso3, year, un_adj)
            if fname is None:
                adj_suffix = "_UNadj" if un_adj else ""
                fname = f"{country_iso3.upper()}_worldpop_ppp_{year}{adj_suffix}.tif"
            destination_path = output_dir / fname
        except Exception as e:
            raise ValueError(f"Could not determine download details for {country_iso3} {year}: {e}") from e

        self._atomic_download(raster_url, destination_path)
        return destination_path

    def _build_url(self, country_iso3: str, year: str = 2020, un_adj: bool = False) -> str:
        """Build download URL.

        Parameters
        ----------
        country_iso3 : str
            Country ISO A3 code.
        year : int, optional
            Year of interest (2000--2020). Default=2020.
        un_adj : bool, optional
            Use UN adjusted population counts. Default=False.

        Returns
        -------
        url : str
            Download URL.
        """
        return (
            f"{self.base_url}/{year}/{country_iso3.upper()}/"
            f"{country_iso3.lower()}_ppp_{year}{'_UNadj' if un_adj else ''}.tif"
        )

    @staticmethod
    def _atomic_download(url: str, destination_path: Path, session: requests.Session | None = None) -> None:
        """Downloads a file from a URL to a destination path atomically.

        It downloads to a temporary file first and renames it upon success,
        preventing partial/corrupt files.

        Parameters
        ----------
        url : str
            The URL of the file to download.
        destination_path : Path
            The final path to save the file.
        session : requests.Session, optional
            An existing requests session to use for the download (in the case of reusing connections).

        Raises
        ------
        requests.HTTPError
            If the download fails with a non-200 status code.
        OSError
            If the file cannot be written to disk.
        """
        # Download to a temporary path
        temp_path = destination_path.with_suffix(destination_path.suffix + ".part")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        http_client = session or requests

        try:
            with http_client.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                with Path.open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                        f.write(chunk)
            # If download is successful, rename the temp file to the final destination
            temp_path.rename(destination_path)

        except (requests.RequestException, OSError) as e:
            raise OSError(f"Failed to download or write file from {url}: {e}") from e
        finally:
            if temp_path.exists():  # Clean up the partial file
                try:
                    temp_path.unlink()
                except OSError as e:
                    raise OSError(f"Error removing partial file {temp_path}: {e}") from e
