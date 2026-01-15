import re
import requests
import logging
import geopandas as gpd
from pathlib import Path
from owslib.wcs import WebCoverageService
import xml.etree.ElementTree as ET
from openhexa.sdk import current_run


class MAPExtractorError(RuntimeError):
    """Custom exception for errors raised by the MAP raster extractor."""

    pass


class MAPRasterExtractor:
    """Raster extractor for Malaria Atlas Project (MAP) datasets via WCS."""

    SUPPORTED_CATEGORIES = ("Malaria", "Interventions")

    def __init__(
        self,
        base_url: str = "https://data.malariaatlas.org/geoserver/",
        category: str = "Malaria",
        logger: logging.Logger | None = None,
    ):
        """Initialize the MAPRasterExtractor."""
        if category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(f"Supported categories: {self.SUPPORTED_CATEGORIES}.")
        self.logger = logger
        self.base_url = base_url
        self.category = category
        self.coverage_ids: list[str] = self._list_coverage_ids_for_category()

    def _list_coverage_ids_for_category(self, category_name: str | None = None) -> list[str]:
        """Private: Fetch coverage IDs from the WCS service.

        Returns:
            List of coverage IDs.
        """
        if category_name is None:
            category_name = self.category
        url = f"{self.base_url}/{category_name}/ows"
        wcs = WebCoverageService(url, version="2.0.1")
        return list(wcs.contents.keys())

    def list_coverage_ids_for_category(self, category: str | None = None) -> list[str]:
        """Log a user-safe message to OpenHexa UI and full details to the file logger.

        Args:
            category: Category to fetch ('Malaria' or 'Interventions').
                      If None, uses the instance's category.

        Returns:
            List of coverage IDs.
        """
        if category is None:
            category = self.category
        return self._list_coverage_ids_for_category(category)

    def _log_message(self, message: str, level: str = "info", exc: Exception | None = None) -> None:
        """Log a message using self.logger and/or current_run."""
        if not level or not message:
            return

        level = level.lower()
        logger_methods = {
            "info": "info",
            "warning": "warning",
            "error": "error",
            "debug": "debug",
        }
        run_methods = {
            "info": "log_info",
            "warning": "log_warning",
            "error": "log_error",
            "debug": "log_debug",
        }

        if level not in logger_methods:
            raise ValueError(f"Unsupported logging level: {level}")

        # Log to standard logger
        if self.logger:
            if exc:
                self.logger.error(message, exc_info=exc)
            else:
                getattr(self.logger, logger_methods[level])(message)

        # current_run to avoid exception details
        try:
            run_fn = getattr(current_run, run_methods[level], None)
            if run_fn:
                run_fn(message)
        except Exception:
            # Never let UI logging break the pipeline
            pass

    def _get_time_positions_for_coverage(self, coverage_id: str, timeout: int = 10) -> dict[str, str]:
        """Parse the WCS DescribeCoverage XML to extract available time positions for a coverage.

        Returns:
            A Dict {year: ISO time strings} (usually one per available year).
        """
        url = f"https://data.malariaatlas.org/geoserver/{self.category}/ows"
        params = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "DescribeCoverage",
            "coverageId": coverage_id,
        }

        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            msg = f"Failed to fetch coverage '{coverage_id}' from {url}"
            self._log_message(msg, level="error", exc=e)
            raise MAPExtractorError(msg) from e

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            msg = f"Failed to parse XML for coverage '{coverage_id}'"
            self._log_message(msg, level="error", exc=e)
            raise MAPExtractorError(msg) from e

        # Namespace declarations commonly found in DescribeCoverage responses
        ns = {
            "gml": "http://www.opengis.net/gml/3.2",
            "wcs": "http://www.opengis.net/wcs/2.0",
        }

        # Look for recurring time positions
        # For a WCS 2.0 service, these often appear as <gml:timePosition> elements
        time_positions = [elem.text for elem in root.findall(".//gml:timePosition", namespaces=ns)]

        # If no individual time positions are found, an overall TimePeriod may exist
        if not time_positions:
            periods = root.findall(".//gml:TimePeriod", namespaces=ns)
            for period in periods:
                begin = period.find("gml:beginPosition", namespaces=ns)
                end = period.find("gml:endPosition", namespaces=ns)
                if begin is not None and end is not None:
                    time_positions.append(f"{begin.text}/{end.text}")

        return {t[:4]: t for t in sorted(time_positions)}

    def _latest_version_for_indicator(self, target_indicator: str) -> str | None:
        """Get the coverage ID with the latest version for a given indicator.

        Given a list of MAP coverage IDs and a target indicator string
        (like "Pv_Incidence_Rate"), return the dataset with the latest version.

        Returns:
            The coverage ID string with the latest version for the target indicator,
            or None if not found.
        """
        # pattern assumes: Malaria__<version>_Global_<indicator>
        version_pattern = re.compile(rf"{self.category}__([0-9]{{6}})_")
        matches = []
        for ds in self.coverage_ids:
            if target_indicator in ds:
                # extract the version part (e.g. 202208, 202508 etc.)
                m = re.search(version_pattern, ds)
                if m:
                    version = m.group(1)
                    matches.append((version, ds))

        if not matches:
            return None

        matches.sort(key=lambda x: x[0])
        return matches[-1][1]  # return latest

    def _get_band_names(self, coverage_id: str | None, category: str | None) -> list[str]:
        """Retrieve the band names (mean, mask, LCI, UCI) of a WCS coverage via DescribeCoverage.

        Args:
            coverage_id: Coverage ID to query. If None, raises ValueError.
            category: Category name ('Malaria' or 'Interventions').

        Returns:
            List of band names/layers available for the coverage id.
        """
        if coverage_id is None:
            raise ValueError("coverage_id must be provided.")
        if category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(f"Supported categories: {self.SUPPORTED_CATEGORIES}.")
        if category:
            url = f"{self.base_url}/{category}/ows"
        else:
            url = f"{self.base_url}/{self.category}/ows"

        params = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "DescribeCoverage",
            "coverageId": coverage_id,
        }
        resp = requests.get(url, params=params)
        root = ET.fromstring(resp.content)

        ns = {"swe": "http://www.opengis.net/swe/2.0"}
        return [field.attrib.get("name") for field in root.findall(".//swe:field", ns)]

    def _build_raster_query(self, coverage_id: str, bbox: list, time_position: str) -> dict:
        """Build the WCS GetCoverage query parameters.

        Returns:
            Dictionary of query parameters.
        """
        return {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "coverageId": coverage_id,
            "format": "image/tiff",
            "subset": [
                f"Long({bbox[0]},{bbox[2]})",
                f"Lat({bbox[1]},{bbox[3]})",
                f'time("{time_position}")',
            ],
        }

    def _download_raster(
        self, coverage_id: str, bbox: list, time_position: str, output_fname: Path, timeout: int = 30
    ) -> Path:
        """Download the raster for the given coverage ID, bounding box, and year.

        Returns:
            Path to the downloaded raster file.
        """
        if output_fname is None:
            raise ValueError("output_fname must be provided.")
        if not output_fname.parent.exists():
            raise ValueError("Provided output_fname's parent directory does not exist.")

        year = time_position.split("-", maxsplit=1)[0]
        params = self._build_raster_query(coverage_id, bbox, time_position)
        url = f"{self.base_url}/{self.category}/wcs"

        try:
            with requests.get(url, params=params, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with output_fname.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.RequestException as e:
            msg = f"Failed to download raster '{coverage_id}' for year {year}"
            self.logger.error(f"{msg} : Error {e}")
            raise MAPExtractorError(msg) from e
        except OSError as e:
            msg = f"Failed to write raster to '{output_fname}'"
            self.logger.error(f"{msg} : Error {e}")
            raise MAPExtractorError(msg) from e

        return output_fname

    def get_band_names(self, coverage_id: str | None, category: str | None = None) -> list[str]:
        """Public: Retrieve the band names (mean, mask, LCI, UCI) of a WCS coverage via DescribeCoverage.

        Args:
            coverage_id: Coverage ID to query. If None, raises ValueError.
            category: Category name ('Malaria' or 'Interventions').

        Returns:
            List of band names/layers available for the coverage id.
        """
        if coverage_id is None:
            raise ValueError("coverage_id must be provided.")
        if category is None:
            category = self.category
        return self._get_band_names(coverage_id=coverage_id, category=category)

    def download_indicator_raster(
        self,
        category: str | None = None,
        indicator: str | None = None,
        target_year: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        shapes: gpd.GeoDataFrame | None = None,
        output_path: Path | None = None,
        replace_file: bool = False,
    ) -> Path:
        """Get the raster dataset ID for a given indicator and target year.

        Args:
            indicator: Indicator name (e.g. 'Pf_Incidence_Rate').
            target_year: Target year as string (e.g. '2022'). If None, uses latest available.
            category: Category name ('Malaria' or 'Interventions').
            bbox: Bounding box definition to download.
              [min longitude(minx), min latitude (miny), max longitude (maxx), max latitude (maxy)].
            shapes: (Preferred) GeoDataFrame defining the area of interest. If provided, bbox is derived.
            output_path: Path where to save the raster.
            replace_file: If True, replaces the file if it already exists.

        Returns:
            Path to the downloaded raster file.
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")

        if category and category != self.category:
            if category not in self.SUPPORTED_CATEGORIES:
                raise ValueError(f"Supported categories: {self.SUPPORTED_CATEGORIES}.")
            self._log_message(f"Switching category from '{self.category}' to '{category}'")
            self.category = category
            self.coverage_ids = self._list_coverage_ids_for_category()

        latest_coverage_id = self._latest_version_for_indicator(indicator)
        if latest_coverage_id is None:
            raise ValueError(f"No coverage found for indicator '{indicator}' in category '{category}'.")
        self._log_message(f"Latest coverage ID for indicator '{indicator}': {latest_coverage_id}")

        available_times: dict = self._get_time_positions_for_coverage(latest_coverage_id)
        if target_year not in available_times:
            raise ValueError(
                f"Year {target_year} is not available for indicator '{indicator}'."
                f" Available years: {available_times if available_times else 'No years available!'}"
            )

        if shapes is not None:
            # derive bbox from shapes
            minx, miny, maxx, maxy = shapes.total_bounds
            bbox = [minx, miny, maxx, maxy]
        if bbox is None:
            raise ValueError("Either bbox or shapes must be provided to define the area of interest.")

        try:
            # Avoid downloading the same file in the provided folder
            raster_fname = output_path / f"{latest_coverage_id}_{target_year}.tif"
            if raster_fname.exists():
                if replace_file:
                    raster_fname.unlink()  # delete existing file
                    self._log_message(
                        f"Raster exists, deleting and re-downloading: {raster_fname.name}", level="warning"
                    )
                else:
                    self._log_message(f"Raster already exists: {raster_fname.name}, skipping download.")
                    return raster_fname

            raster_path = self._download_raster(
                coverage_id=latest_coverage_id,
                bbox=bbox,
                time_position=available_times[target_year],
                output_fname=raster_fname,
            )
            self._log_message(f"Raster downloaded successfully: {raster_path.name}")
        except MAPExtractorError as e:
            msg = f"Download failed: {raster_fname.name}"
            self._log_message(msg, level="error", exc=e)
            raise MAPExtractorError(msg) from e
        except Exception as e:
            msg = f"An unexpected error occurred while downloading raster: {raster_fname.name}"
            self._log_message(msg, level="error", exc=e)
            raise MAPExtractorError(msg) from e

        return raster_path
