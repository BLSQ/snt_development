import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

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
    "metadata_file",
    name="SNT metadata JSON (all layers)",
    type=File,
    required=True,
    help=(
        "The SNT_metadata_all_layers.json catalogue to trim. Each layer is kept only if the column "
        "named in its SOURCE_DATA exists in the referenced dataset file of this workspace."
    ),
)
def snt_metadata_json_trim(metadata_file: File) -> None:
    """Trim the SNT metadata catalogue to the layers whose source column exists in this workspace.

    Never fails the run: every problem is logged, and the run ends early (with nothing published)
    only when no output can be produced at all.
    """
    root_path = Path(workspace.files_path)
    output_path = root_path / "pipelines" / "snt_metadata_json_trim" / "output"

    try:
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
        dataset_identifiers = snt_config.get("SNT_DATASET_IDENTIFIERS", {})
    except Exception as e:
        current_run.log_error(
            f"Could not load configuration/SNT_config.json or read COUNTRY_CODE from it: {e}. "
            "Nothing was produced."
        )
        return

    try:
        metadata = read_metadata(Path(metadata_file.path))
    except Exception as e:
        current_run.log_error(
            f"Could not read the metadata file {metadata_file.path}: {e}. Nothing was produced."
        )
        return

    current_run.log_info(
        f"Checking {len(metadata)} layers from {metadata_file.name} for country {country_code}."
    )

    sources = check_sources(metadata, country_code, dataset_identifiers)
    trimmed, dropped = trim_metadata(metadata, sources, country_code)
    log_summary(metadata, trimmed, dropped)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        trimmed_path = write_json(trimmed, output_path / TRIMMED_FILENAME)
        report_path = write_json(
            build_report(metadata_file.path, country_code, metadata, trimmed, dropped, sources),
            output_path / REPORT_FILENAME,
        )
        current_run.log_info(f"Trimmed metadata written to {trimmed_path}.")
    except Exception as e:
        current_run.log_error(
            f"Could not write the output files to {output_path}: {e}. Nothing was published."
        )
        return

    try:
        publish_to_dataset([trimmed_path, report_path], country_code)
    except Exception as e:
        current_run.log_error(
            f"Could not publish to the {OUTPUT_DATASET_NAME} dataset: {e}. "
            f"The output files are still available in {output_path}."
        )


def read_metadata(path: Path) -> dict:
    """Read the metadata catalogue and check it is a JSON object of layers.

    Parameters
    ----------
    path : Path
        Path to the metadata JSON file.

    Returns
    -------
    dict
        The catalogue, keyed by layer id, in file order.

    Raises
    ------
    ValueError
        If the file is not a JSON object.
    """
    with path.open(encoding="utf-8-sig") as f:
        metadata = json.load(f)
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

    if metadata and not trimmed:
        current_run.log_error(
            "No layer was kept. Either no upstream pipeline has published its outputs yet in this "
            "workspace, or the metadata file does not match this workspace (dataset keys, filenames, "
            "column names). See the reasons above and SNT_metadata_trimmed_report.json."
        )
    if by_reason.get(CHECK_FAILED):
        current_run.log_error(
            f"{len(by_reason[CHECK_FAILED])} layer(s) were dropped because a check failed, not because "
            "the data is absent. Re-run once the error above is resolved."
        )


def build_report(
    input_path: str,
    country_code: str,
    metadata: dict,
    trimmed: dict,
    dropped: list[dict],
    sources: dict[tuple, SourceCheck],
) -> dict:
    """Build the run report published beside the trimmed catalogue.

    Parameters
    ----------
    input_path : str
        Path of the input metadata file.
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
        "INPUT_FILE": input_path,
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
    """Write JSON in the same format as build_snt_metadata_all_layers.py.

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
