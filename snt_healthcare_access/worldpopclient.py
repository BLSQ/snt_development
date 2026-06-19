from pathlib import Path
import re
import requests
from openhexa.sdk import current_run
import logging


class WorldPopClient:
    """Mini client for the WorldPop REST API.

    Source: https://data.worldpop.org/GIS/Population
    """

    def __init__(
        self, url: str = "https://data.worldpop.org/GIS/Population", logger: logging.Logger | None = None
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        url : str
            The base URL for the WorldPop data download.
        logger : logging.Logger, optional
            A logger instance to use for logging messages. If None, a default logger will be created
        """
        self.base_url = url
        self.logger = logger or logging.getLogger(__name__)

    def download_data_for_country(
        self,
        country_iso3: str,
        year: int,
        output_dir: Path,
        overwrite: bool = False,
        filename: str | None = None,
    ) -> Path:
        """Download and save the WorldPop raster dataset for a given country and year.

        This operation is atomic. A partial download will not result in a corrupt
        final file.

        Parameters
        ----------
        country_iso3 : str
            3-letter ISO code of the country (e.g., "COD", "BFA").
        year : int
            Year to filter the dataset (e.g. 2020).
        output_dir : Path
            Directory to save the GeoTIFF file.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists. Defaults to False.
        filename : str, optional
            Filename to save the raster data. If None, defaults to
            "{country_iso3}_worldpop_population_{year}.tif".

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

        if year < 2015 or year > 2030:  # NOTE: We might want to change the url repo in the future.
            raise ValueError(
                f"WorldPop data not available for {year} "
                "(see: https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/)"
            )

        country_iso3 = country_iso3.upper()
        candidate_url = self._build_url(country_iso3, year)

        # Determine the filename to save as
        if filename:
            fname = filename
        else:
            fname = Path(candidate_url).name

        destination_path = output_dir / fname

        if not overwrite and destination_path.exists():
            self._log(f"File {destination_path.name} already exists. Skipping download.", level="info")
            return destination_path

        self._download_file(candidate_url, destination_path)
        return destination_path

    def _build_url(self, country_iso3: str, year: int) -> str:
        """Build download URL candidates.

        Parameters
        ----------
        country_iso3 : str
            Country ISO A3 code.
        year : int, optional
            Year of interest.

        Returns
        -------
        Path
            download URL candidate.
        """
        # select latest release available
        releases = self._list_remote_directories(url=f"{self.base_url}/Global_2015_2030/")
        if not releases:
            raise ValueError(f"No releases found at {self.base_url}/Global_2015_2030/")
        latest_release = releases[0]
        return (
            f"{self.base_url}/Global_2015_2030/{latest_release}/{year}/{country_iso3.upper()}/"
            f"v1/100m/constrained/{country_iso3.lower()}_pop_{year}_CN_100m_{latest_release}_v1.tif"
        )

    def _list_remote_directories(self, url: str) -> list[str]:
        """List folder names available at an HTTP directory listing URL.

        Returns
        -------
        list[str]
            Directory names found at the URL, sorted in reverse alphabetical order.
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return sorted(set(re.findall(r'href="([^/"]+)/"', response.text)), reverse=True)

    def _download_file(self, url: str, destination_path: Path) -> None:
        """Download a WorldPop raster from URL."""
        try:
            self._log(f"Download WorldPop raster data from URL: {url}")
            self._atomic_download(url, destination_path)
            return
        except OSError as err:
            raise OSError(f"WorldPop URL '{url}' failed to download. Details: {err}") from err

    def _atomic_download(
        self, url: str, destination_path: Path, session: requests.Session | None = None
    ) -> None:
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
                    self._log(f"Failed to remove partial file {temp_path}: {e}", level="warning")

    def _log(self, message: str, level: str = "info") -> None:
        """Log a message using the Python logger and/or the OpenHEXA current_run, if available."""
        if self.logger:
            log_method = getattr(self.logger, level, self.logger.info)
            log_method(message)
        if current_run is not None:
            if level == "info":
                current_run.log_info(message)
            elif level == "warning":
                current_run.log_warning(message)
            elif level == "error":
                current_run.log_error(message)
            elif level == "debug":
                current_run.log_debug(message)
            elif level == "critical":
                current_run.log_critical(message)