import hashlib
import io
import json
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import NoReturn
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
from openhexa.sdk import File, current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets.dataset import Dataset, DatasetVersion
from snt_lib.snt_pipeline_utils import load_configuration_snt

OUTPUT_DATASET_NAME = "SNT_METADATA"
OUTPUT_DATASET_SLUG = "snt-metadata"
OUTPUT_DATASET_DESCRIPTION = (
    "SNT Explorer data-layer catalogue, trimmed to the layers available in this workspace."
)
TRIMMED_FILENAME = "SNT_metadata_trimmed.json"
REPORT_FILENAME = "SNT_metadata_trimmed_report.json"

# Where the master list is fetched from. "latest" resolves to the latest published GitHub release.
REPOSITORY = "BLSQ/snt_development"
MASTER_REPO_PATH = "metadata/SNT_metadata_master.json"
LATEST_RELEASE = "latest"
HTTP_TIMEOUT_SECONDS = 30


class MasterNotFoundError(Exception):
    """The requested ref has no master list: no release yet, unknown ref, or ref older than the file."""


# Drop reasons, in the order the checks run. Anything other than COLUMN_NOT_FOUND means the column
# could not even be looked for.
MALFORMED_LAYER = "MALFORMED_LAYER"  # SOURCE_DATA is missing a required field
CONFIG_KEY_MISSING = "CONFIG_KEY_MISSING"  # DATASET.NAME is not a key of SNT_DATASET_IDENTIFIERS
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"  # the configured dataset does not exist in the workspace
DATASET_NO_VERSION = "DATASET_NO_VERSION"  # the dataset exists but has never been published to
VERSION_NOT_FOUND = "VERSION_NOT_FOUND"  # a pinned VERSION matches no version name or id
FILE_NOT_FOUND = "FILE_NOT_FOUND"  # FILENAME is not in the resolved version
UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"  # neither .csv nor .parquet
CHECK_FAILED = "CHECK_FAILED"  # API error, or the file could not be read
COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"  # the file exists but does not contain COLUMN


@dataclass
class SourceCheck:
    """Outcome of resolving one (dataset key, version, filename) triple, shared by all its layers."""

    dataset_key: str
    version_requested: str
    filename: str
    dataset_id: str | None = None
    version_resolved: str | None = None
    columns: list[str] = field(default_factory=list)
    reason: str | None = None  # None means the file was found and its columns read
    detail: str = ""


@pipeline("snt_metadata_json_trim")
@parameter(
    "repository_ref",
    name="Master list version (release tag or branch)",
    type=str,
    default="latest",  # must be a literal: the OpenHEXA CLI parses defaults via AST (= LATEST_RELEASE)
    required=True,
    help=(
        "Which version of SNT_metadata_master.json to fetch from the snt_development repository: "
        "'latest' (the latest release), a release tag, or a branch such as 'main'. "
        "Ignored when a file is given below."
    ),
)
@parameter(
    "metadata_file",
    name="Master list file (optional override)",
    type=File,
    required=False,
    default=None,
    help=(
        "Use this file instead of fetching the master list from the repository, e.g. to test an edited "
        "copy. Leave empty for normal runs."
    ),
)
def snt_metadata_json_trim(repository_ref: str, metadata_file: File | None = None) -> None:
    """Trim the SNT metadata master list to the layers whose source column exists in this workspace.

    Fails the run whenever no usable output is published: configuration or master list unavailable,
    no layer kept, or the outputs could not be written or published. Layers that cannot be checked
    are dropped and logged without failing the run.
    """
    root_path = Path(workspace.files_path)
    output_path = root_path / "pipelines" / "snt_metadata_json_trim" / "output"

    try:
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
        dataset_identifiers = snt_config.get("SNT_DATASET_IDENTIFIERS", {})
    except Exception as e:
        fail(
            f"Could not load configuration/SNT_config.json or read COUNTRY_CODE from it: {e}. "
            "Nothing was produced.",
            e,
        )

    try:
        content, master = load_master(repository_ref, metadata_file)
    except Exception as e:
        fail(f"Could not get the master list: {e}. Nothing was produced.", e)

    try:
        metadata = parse_metadata(content)
    except Exception as e:
        fail(f"Could not parse the master list {master['PATH']}: {e}. Nothing was produced.", e)

    current_run.log_info(
        f"Checking {len(metadata)} layers from {master['PATH']} (sha256 {master['SHA256'][:12]}) "
        f"for country {country_code}."
    )

    sources = check_sources(metadata, country_code, dataset_identifiers)
    trimmed, dropped = trim_metadata(metadata, sources, country_code)
    log_summary(metadata, trimmed, dropped)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        trimmed_path = write_json(trimmed, output_path / TRIMMED_FILENAME)
        report_path = write_json(
            build_report(master, country_code, metadata, trimmed, dropped, sources),
            output_path / REPORT_FILENAME,
        )
        current_run.log_info(f"Trimmed metadata written to {trimmed_path}.")
    except Exception as e:
        fail(f"Could not write the output files to {output_path}: {e}. Nothing was published.", e)

    if not trimmed:
        fail(
            "No layer was kept. Either no upstream pipeline has published its outputs yet in this "
            "workspace, or the master list does not match it (dataset keys, filenames, column names). "
            f"Nothing was published to {OUTPUT_DATASET_NAME}, so its previous version stays the latest. "
            f"The reason for every dropped layer is logged above and in {report_path}."
        )

    try:
        publish_to_dataset([trimmed_path, report_path], country_code)
    except Exception as e:
        fail(
            f"Could not publish to the {OUTPUT_DATASET_NAME} dataset: {e}. "
            f"The output files are still available in {output_path}.",
            e,
        )


class PipelineFailedError(Exception):
    """Ends the run as failed; its message has already been logged at error level."""


def fail(message: str, cause: Exception | None = None) -> NoReturn:
    """Log an error and fail the run.

    Parameters
    ----------
    message : str
        What went wrong and what was (not) produced.
    cause : Exception | None
        The underlying exception, chained for the traceback.

    Raises
    ------
    PipelineFailedError
        Always.
    """
    current_run.log_error(message)
    raise PipelineFailedError(message) from cause


def load_master(repository_ref: str, metadata_file: File | None) -> tuple[bytes, dict]:
    """Get the master list content, from the override file if given, else from the repository.

    Parameters
    ----------
    repository_ref : str
        "latest", a release tag, or a branch name.
    metadata_file : File | None
        Optional override file.

    Returns
    -------
    tuple[bytes, dict]
        The raw content, and the MASTER provenance block for the report.
    """
    if metadata_file is not None:
        current_run.log_info(f"Using the master list file {metadata_file.path}; repository_ref is ignored.")
        content = Path(metadata_file.path).read_bytes()
        source, ref_requested, ref_resolved, path = "file", None, None, metadata_file.path
    else:
        ref_requested = (repository_ref or LATEST_RELEASE).strip()
        ref_resolved = resolve_ref(ref_requested)
        content, path = fetch_master(ref_resolved)
        source = "repository"
        current_run.log_info(f"Fetched the master list from {REPOSITORY} at '{ref_resolved}'.")

    return content, {
        "SOURCE": source,
        "REF_REQUESTED": ref_requested,
        "REF_RESOLVED": ref_resolved,
        "PATH": path,
        "SHA256": hashlib.sha256(content).hexdigest(),
    }


def http_get(url: str) -> bytes:
    """Fetch a URL from GitHub, unauthenticated (the repository is public).

    Parameters
    ----------
    url : str
        The URL.

    Returns
    -------
    bytes
        The response body.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "openhexa-snt-metadata-json-trim"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def resolve_ref(ref: str) -> str:
    """Turn "latest" into the tag of the latest published release; return any other ref unchanged.

    Parameters
    ----------
    ref : str
        "latest", a release tag, or a branch name.

    Returns
    -------
    str
        The ref to fetch from.

    Raises
    ------
    MasterNotFoundError
        If ref is "latest" and the repository has no published release.
    """
    if ref != LATEST_RELEASE:
        return ref
    try:
        release = json.loads(http_get(f"https://api.github.com/repos/{REPOSITORY}/releases/latest"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise MasterNotFoundError(
                f"no release has been published yet in {REPOSITORY}. Set repository_ref to a branch "
                "such as 'main'"
            ) from e
        raise
    return release["tag_name"]


def fetch_master(ref: str) -> tuple[bytes, str]:
    """Download the master list at a given ref.

    Parameters
    ----------
    ref : str
        A release tag or branch name.

    Returns
    -------
    tuple[bytes, str]
        The file content, and the URL it was fetched from.

    Raises
    ------
    MasterNotFoundError
        If the ref does not exist, or exists but does not contain the master list.
    """
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{quote(ref, safe='/')}/{MASTER_REPO_PATH}"
    try:
        return http_get(url), url
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    # A 404 means either the ref or the file is missing: ask the API which one.
    try:
        http_get(f"https://api.github.com/repos/{REPOSITORY}/commits/{quote(ref, safe='/')}")
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            raise MasterNotFoundError(
                f"'{ref}' is not a release tag or branch of {REPOSITORY}. Check the spelling, or use "
                "'latest', a release tag or a branch such as 'main'"
            ) from e
        raise
    raise MasterNotFoundError(
        f"'{ref}' exists in {REPOSITORY} but has no {MASTER_REPO_PATH} (it predates the master list). "
        "Use a newer release tag, or a branch that contains the file"
    )


def parse_metadata(content: bytes) -> dict:
    """Parse the master list and check it is a JSON object of layers.

    Parameters
    ----------
    content : bytes
        The raw file content.

    Returns
    -------
    dict
        The master list, keyed by layer id, in file order.

    Raises
    ------
    ValueError
        If the content is not a JSON object.
    """
    metadata = json.loads(content.decode("utf-8-sig"))
    if not isinstance(metadata, dict):
        raise ValueError("expected a JSON object keyed by layer id")
    return metadata


def source_of(layer: dict, country_code: str) -> tuple[str, str, str, str]:
    """Extract the source pointer of a layer, with {COUNTRY_CODE} resolved.

    Parameters
    ----------
    layer : dict
        One entry of the metadata catalogue.
    country_code : str
        Country code substituted into FILENAME.

    Returns
    -------
    tuple[str, str, str, str]
        (dataset key, requested version, filename, column).

    Raises
    ------
    KeyError
        If SOURCE_DATA is missing a required field.
    """
    source = layer["SOURCE_DATA"]
    dataset = source["DATASET"]
    return (
        dataset["NAME"],
        str(dataset.get("VERSION", "latest")),
        source["FILENAME"].replace("{COUNTRY_CODE}", country_code),
        source["COLUMN"],
    )


def check_sources(metadata: dict, country_code: str, dataset_identifiers: dict) -> dict[tuple, SourceCheck]:
    """Resolve each distinct source file once and read its column names.

    Parameters
    ----------
    metadata : dict
        The metadata catalogue.
    country_code : str
        Country code substituted into FILENAME.
    dataset_identifiers : dict
        SNT_DATASET_IDENTIFIERS from SNT_config.json.

    Returns
    -------
    dict[tuple, SourceCheck]
        One check per (dataset key, version, filename).
    """
    keys = set()
    for layer in metadata.values():
        try:
            keys.add(source_of(layer, country_code)[:3])
        except (KeyError, TypeError, AttributeError):
            continue  # reported per layer as MALFORMED_LAYER

    return {key: check_source(*key, dataset_identifiers) for key in sorted(keys)}


def check_source(dataset_key: str, version: str, filename: str, dataset_identifiers: dict) -> SourceCheck:
    """Locate one file in its dataset version and read its column names.

    Parameters
    ----------
    dataset_key : str
        DATASET.NAME, a key of SNT_DATASET_IDENTIFIERS.
    version : str
        DATASET.VERSION: "latest", or a version name or id.
    filename : str
        FILENAME with {COUNTRY_CODE} resolved.
    dataset_identifiers : dict
        SNT_DATASET_IDENTIFIERS from SNT_config.json.

    Returns
    -------
    SourceCheck
        The columns found, or the reason they could not be read.
    """
    check = SourceCheck(dataset_key=dataset_key, version_requested=version, filename=filename)

    check.dataset_id = dataset_identifiers.get(dataset_key)
    if not check.dataset_id:
        check.reason = CONFIG_KEY_MISSING
        check.detail = f"'{dataset_key}' is not a key of SNT_DATASET_IDENTIFIERS in SNT_config.json"
        return log_source(check)

    try:
        dataset = workspace.get_dataset(check.dataset_id)
    except ValueError:
        check.reason = DATASET_NOT_FOUND
        check.detail = f"dataset '{check.dataset_id}' does not exist in this workspace"
        return log_source(check)
    except Exception as e:
        check.reason = CHECK_FAILED
        check.detail = f"could not look up dataset '{check.dataset_id}': {e}"
        return log_source(check)

    try:
        dataset_version = resolve_version(dataset, version)
    except Exception as e:
        check.reason = CHECK_FAILED
        check.detail = f"could not list the versions of dataset '{check.dataset_id}': {e}"
        return log_source(check)
    if dataset_version is None:
        if version == "latest":
            check.reason = DATASET_NO_VERSION
            check.detail = f"dataset '{check.dataset_id}' has no version yet"
        else:
            check.reason = VERSION_NOT_FOUND
            check.detail = f"dataset '{check.dataset_id}' has no version named or with id '{version}'"
        return log_source(check)
    check.version_resolved = dataset_version.name

    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        check.reason = UNSUPPORTED_FILE_TYPE
        check.detail = f"cannot read columns from a '{suffix}' file"
        return log_source(check)

    try:
        dataset_file = dataset_version.get_file(filename)
    except FileExistsError:  # what the SDK raises for a missing file
        check.reason = FILE_NOT_FOUND
        check.detail = f"'{filename}' is not in version '{dataset_version.name}' of '{check.dataset_id}'"
        return log_source(check)
    except Exception as e:
        check.reason = CHECK_FAILED
        check.detail = f"could not look up '{filename}' in '{check.dataset_id}': {e}"
        return log_source(check)

    try:
        check.columns = read_columns(dataset_file.read(), suffix)
    except Exception as e:
        check.reason = CHECK_FAILED
        check.detail = f"could not read the columns of '{filename}' in '{check.dataset_id}': {e}"

    return log_source(check)


def resolve_version(dataset: Dataset, version: str) -> DatasetVersion | None:
    """Return the dataset version a layer asks for.

    "latest" means the most recent version, whatever its name. Any other value is matched against
    version names, then ids.

    Parameters
    ----------
    dataset : Dataset
        The dataset to look in.
    version : str
        DATASET.VERSION from the layer.

    Returns
    -------
    DatasetVersion | None
        The matching version, or None if there is none.
    """
    if version == "latest":
        return dataset.latest_version
    for candidate in dataset.versions:
        if version in (candidate.name, candidate.id):
            return candidate
    return None


def read_columns(content: bytes, suffix: str) -> list[str]:
    """Read only the column names of a CSV or parquet file.

    Parameters
    ----------
    content : bytes
        The file content.
    suffix : str
        ".csv" or ".parquet".

    Returns
    -------
    list[str]
        The column names, exactly as stored.
    """
    if suffix == ".parquet":
        return pq.read_schema(io.BytesIO(content)).names
    return list(pd.read_csv(io.BytesIO(content), nrows=0).columns)


def log_source(check: SourceCheck) -> SourceCheck:
    """Log the outcome of one source check.

    Parameters
    ----------
    check : SourceCheck
        The completed check.

    Returns
    -------
    SourceCheck
        The same check, so callers can log and return in one statement.
    """
    where = f"{check.dataset_key} / {check.filename}"
    if check.reason is None:
        current_run.log_info(
            f"Found {where} (version {check.version_resolved}, {len(check.columns)} columns)."
        )
    elif check.reason == CHECK_FAILED:
        current_run.log_error(f"[{check.reason}] {where}: {check.detail}")
    else:
        current_run.log_warning(f"[{check.reason}] {where}: {check.detail}")
    return check


def trim_metadata(
    metadata: dict, sources: dict[tuple, SourceCheck], country_code: str
) -> tuple[dict, list[dict]]:
    """Keep the layers whose column exists in their source file.

    Parameters
    ----------
    metadata : dict
        The metadata catalogue.
    sources : dict[tuple, SourceCheck]
        The source checks from `check_sources`.
    country_code : str
        Country code substituted into FILENAME.

    Returns
    -------
    tuple[dict, list[dict]]
        The kept layers, unchanged and in file order; and one record per dropped layer.
    """
    trimmed = {}
    dropped = []
    for layer_id, layer in metadata.items():
        try:
            dataset_key, version, filename, column = source_of(layer, country_code)
        except (KeyError, TypeError, AttributeError) as e:
            dropped.append(drop_record(layer_id, MALFORMED_LAYER, f"SOURCE_DATA is missing {e}"))
            continue

        check = sources[dataset_key, version, filename]
        if check.reason is not None:
            dropped.append(drop_record(layer_id, check.reason, check.detail, dataset_key, filename, column))
        elif column not in check.columns:
            dropped.append(
                drop_record(
                    layer_id,
                    COLUMN_NOT_FOUND,
                    f"'{column}' is not a column of '{filename}'",
                    dataset_key,
                    filename,
                    column,
                )
            )
        else:
            trimmed[layer_id] = layer

    return trimmed, dropped


def drop_record(
    layer_id: str,
    reason: str,
    detail: str,
    dataset_key: str | None = None,
    filename: str | None = None,
    column: str | None = None,
) -> dict:
    """Build the report entry for a dropped layer.

    Parameters
    ----------
    layer_id : str
        The layer id.
    reason : str
        One of the drop-reason constants.
    detail : str
        Human-readable explanation.
    dataset_key : str | None
        DATASET.NAME, if it could be read.
    filename : str | None
        Resolved FILENAME, if it could be read.
    column : str | None
        COLUMN, if it could be read.

    Returns
    -------
    dict
        The report entry.
    """
    return {
        "LAYER_ID": layer_id,
        "REASON": reason,
        "DETAIL": detail,
        "DATASET_NAME": dataset_key,
        "FILENAME": filename,
        "COLUMN": column,
    }


def log_summary(metadata: dict, trimmed: dict, dropped: list[dict]) -> None:
    """Log how many layers were kept, and the dropped ones grouped by reason.

    Parameters
    ----------
    metadata : dict
        The input catalogue.
    trimmed : dict
        The kept layers.
    dropped : list[dict]
        The dropped-layer records.
    """
    by_reason = defaultdict(list)
    for record in dropped:
        by_reason[record["REASON"]].append(record["LAYER_ID"])
    for reason, layer_ids in by_reason.items():
        log = current_run.log_error if reason == CHECK_FAILED else current_run.log_warning
        log(f"Dropped {len(layer_ids)} layer(s) [{reason}]: {', '.join(layer_ids)}")

    current_run.log_info(f"Kept {len(trimmed)} of {len(metadata)} layers, dropped {len(dropped)}.")

    if by_reason.get(CHECK_FAILED):
        current_run.log_error(
            f"{len(by_reason[CHECK_FAILED])} layer(s) were dropped because a check failed, not because "
            "the data is absent. Re-run once the error above is resolved."
        )


def build_report(
    master: dict,
    country_code: str,
    metadata: dict,
    trimmed: dict,
    dropped: list[dict],
    sources: dict[tuple, SourceCheck],
) -> dict:
    """Build the run report published beside the trimmed catalogue.

    Parameters
    ----------
    master : dict
        Provenance of the master list, from `load_master`.
    country_code : str
        The country code.
    metadata : dict
        The input catalogue.
    trimmed : dict
        The kept layers.
    dropped : list[dict]
        The dropped-layer records.
    sources : dict[tuple, SourceCheck]
        The source checks, recording which dataset version each file was checked in.

    Returns
    -------
    dict
        The report.
    """
    return {
        "RUN_AT": datetime.now(UTC).isoformat(timespec="seconds"),
        "COUNTRY_CODE": country_code,
        "MASTER": master,
        "N_LAYERS_INPUT": len(metadata),
        "N_LAYERS_KEPT": len(trimmed),
        "N_LAYERS_DROPPED": len(dropped),
        "KEPT": list(trimmed),
        "DROPPED": dropped,
        "SOURCES": [
            {
                "DATASET_NAME": check.dataset_key,
                "DATASET_ID": check.dataset_id,
                "VERSION_REQUESTED": check.version_requested,
                "VERSION_RESOLVED": check.version_resolved,
                "FILENAME": check.filename,
                "STATUS": check.reason or "FOUND",
                "DETAIL": check.detail,
            }
            for check in sources.values()
        ],
    }


def write_json(content: dict, path: Path) -> Path:
    """Write JSON in the same format as metadata/SNT_metadata_master.json (4-space indent, UTF-8).

    Parameters
    ----------
    content : dict
        The content to write.
    path : Path
        Destination file.

    Returns
    -------
    Path
        The path written.
    """
    path.write_text(json.dumps(content, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def get_or_create_output_dataset() -> Dataset:
    """Return the SNT_METADATA dataset, creating it on first run.

    Looks it up by slug first, then by name, so a dataset created under a different slug is still
    found rather than duplicated.

    Returns
    -------
    Dataset
        The output dataset.
    """
    try:
        return workspace.get_dataset(OUTPUT_DATASET_SLUG)
    except ValueError:
        pass

    for dataset in workspace.list_datasets():
        if dataset.name == OUTPUT_DATASET_NAME:
            return dataset

    dataset = workspace.create_dataset(OUTPUT_DATASET_NAME, OUTPUT_DATASET_DESCRIPTION)
    current_run.log_info(f"Created dataset {OUTPUT_DATASET_NAME} (slug '{dataset.slug}').")
    return dataset


def publish_to_dataset(file_paths: list[Path], country_code: str) -> None:
    """Publish the output files as a new version of the SNT_METADATA dataset.

    Parameters
    ----------
    file_paths : list[Path]
        Files to add to the new version.
    country_code : str
        Used in the version name, matching the other SNT datasets (SNT_<CC>_<YYYYMMDD_HHMM>).
    """
    dataset = get_or_create_output_dataset()
    now = datetime.now(UTC)
    try:
        version = dataset.create_version(f"SNT_{country_code}_{now:%Y%m%d_%H%M}")
    except ValueError:  # a version with this name already exists: two runs in the same minute
        version = dataset.create_version(f"SNT_{country_code}_{now:%Y%m%d_%H%M%S}")

    for file_path in file_paths:
        version.add_file(file_path, filename=file_path.name)

    published = ", ".join(p.name for p in file_paths)
    current_run.log_info(f"Published {published} to {OUTPUT_DATASET_NAME} (version {version.name}).")


if __name__ == "__main__":
    snt_metadata_json_trim()
