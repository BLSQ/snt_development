"""Deploy a pinned GitHub release of the SNT codebase into this OpenHEXA workspace.

Both halves of the codebase move together, on one release tag, replacing the two
uncoordinated update paths described in docs/wip/release_strategy.md:

    R analytics   pipelines/**/*.ipynb, pipelines/**/utils/*.r, code/**/*.r
                  -> copied into the workspace filesystem, where notebooks source() them.
                  Replaces the per-pipeline `Pull scripts` toggle.

    Python half   <name>/pipeline.py
                  -> registered as a new pipeline VERSION through the OpenHEXA API.
                  Replaces the Template auto-update subscription.

The Python half is deliberately NOT copied into the workspace filesystem. OpenHEXA runs
each pipeline from the zip stored on its registered version, never from the bucket, so a
pipeline.py sitting in workspace/files would look authoritative and never execute.

Credentials: a pipeline run's own HEXA_TOKEN can read the API but is refused
(PERMISSION_DENIED) on uploadPipeline, so deployment reads a workspace API token from a
CUSTOM connection instead - see the `api_connection` parameter.
"""

import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.pipelines.pipeline import Pipeline
from openhexa.sdk.pipelines.runtime import get_pipeline

# The suffixes `openhexa pipelines push` puts in a pipeline version's zip. Kept identical
# so a version deployed from here is byte-comparable with one pushed by CI.
ZIPPED_SUFFIXES = (".py", ".ipynb", ".txt", ".md", ".r", ".sql")

GITHUB_HEADERS = {"User-Agent": "snt-workspace-manager"}


@pipeline("snt_workspace_manager")
@parameter(
    "github_repo",
    name="GitHub repository",
    help="owner/repo to pull the release from (e.g. BLSQ/snt_development_sandbox)",
    type=str,
    default="BLSQ/snt_development_sandbox",
    required=True,
)
@parameter(
    "release_tag",
    name="Release tag",
    help="GitHub release tag to deploy (e.g. v0.0.1-test). Becomes the pipeline version name.",
    type=str,
    default=None,
    required=True,
)
@parameter(
    "api_connection",
    name="OpenHEXA API connection",
    help=(
        "CUSTOM connection holding a workspace API token in a secret field named 'token'. "
        "A run's own credentials cannot deploy pipelines. Only needed if 'Deploy pipelines' is on."
    ),
    type=str,
    default="oh",
    required=False,
)
@parameter(
    "sync_analytics",
    name="Sync R analytics to the filesystem",
    help="Copy the release's notebooks and .r helpers into the workspace filesystem",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "deploy_pipelines",
    name="Deploy pipelines",
    help="Register each release pipeline.py as a new version of the matching OpenHEXA pipeline",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "only_pipelines",
    name="Only these pipelines",
    help=(
        "Comma-separated pipeline names to deploy, e.g. 'snt_dhis2_extract, snt_map_extracts'. "
        "Leave empty to deploy every pipeline in the release. Does not affect the analytics sync."
    ),
    type=str,
    default=None,
    required=False,
)
@parameter(
    "create_missing",
    name="Create pipelines that do not exist yet",
    help=(
        "Bootstrap an empty workspace by creating any pipeline the release defines but the "
        "workspace lacks. Off by default so a typo cannot silently create a duplicate."
    ),
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "backup_existing",
    name="Backup existing files",
    help="Move any existing tracked file to workspace/archive/<release_tag>/ before overwriting it",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "dry_run",
    name="Dry run",
    help="Report what would change without writing any file or registering any version",
    type=bool,
    default=False,
    required=False,
)
def snt_workspace_manager(
    github_repo: str,
    release_tag: str,
    api_connection: str,
    sync_analytics: bool,
    deploy_pipelines: bool,
    only_pipelines: str | None,
    create_missing: bool,
    backup_existing: bool,
    dry_run: bool,
) -> None:
    """Deploy one GitHub release's R analytics and Python orchestration into this workspace.

    Orchestration only: resolves the release, downloads its manifest and source tarball,
    then delegates the file sync and the pipeline deployment to plain helper functions.
    """
    snt_root_path = Path(workspace.files_path)
    if dry_run:
        current_run.log_info("DRY RUN - nothing will be written or registered.")

    token = get_api_token(api_connection) if deploy_pipelines else None

    release = get_release(github_repo, release_tag)
    manifest = download_manifest(release)
    analytics_files, pipeline_dirs = split_manifest(manifest["files"], manifest.get("pipelines"))
    current_run.log_info(
        f"Release {github_repo}@{release_tag} tracks {len(manifest['files'])} files: "
        f"{len(analytics_files)} analytics file(s) and {len(pipeline_dirs)} pipeline(s)."
    )

    failures = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tarball_root = download_and_extract_tarball(release["tarball_url"], Path(tmp_dir))

        if sync_analytics:
            archive_dir = snt_root_path / "archive" / release_tag if backup_existing else None
            copied, missing = sync_files(analytics_files, tarball_root, snt_root_path, archive_dir, dry_run)
            verb = "would be synced" if dry_run else "synced"
            current_run.log_info(f"Analytics: {len(copied)}/{len(analytics_files)} file(s) {verb}.")
            if missing:
                current_run.log_warning(
                    f"{len(missing)} manifest entries were not found in the release tarball: {missing}"
                )
        else:
            current_run.log_info("Analytics sync skipped by parameter.")

        if deploy_pipelines:
            failures = deploy_all(
                tarball_root,
                filter_pipelines(pipeline_dirs, only_pipelines),
                release,
                token,
                create_missing,
                dry_run,
            )
        else:
            current_run.log_info("Pipeline deployment skipped by parameter.")

    if not dry_run:
        write_release_marker(snt_root_path, release_tag)

    if failures:
        raise RuntimeError(
            f"{len(failures)} pipeline(s) failed to deploy: {failures}. The workspace is now "
            "partially updated - fix the cause and re-run to converge on the release."
        )


def get_api_token(connection_slug: str) -> str:
    """Read the workspace API token used to deploy pipelines.

    A run's own HEXA_TOKEN is refused with PERMISSION_DENIED on uploadPipeline, so
    deployment needs a workspace API key supplied through a CUSTOM connection.

    Returns
    -------
    str
        The bearer token held in the connection's `token` field.
    """
    try:
        token = workspace.custom_connection(connection_slug).token
    except Exception as exception:
        raise ValueError(
            f"Could not read a token from the custom connection '{connection_slug}': {exception}. "
            "Create a CUSTOM connection with that slug and a secret field named 'token' holding a "
            "workspace API token, or turn off 'Deploy pipelines'."
        ) from exception

    current_run.log_info(f"Using the API token from connection '{connection_slug}' to deploy.")
    return token


def get_release(github_repo: str, release_tag: str) -> dict:
    """Fetch a GitHub release's metadata by tag.

    Returns
    -------
    dict
        The GitHub API release object (tag_name, tarball_url, html_url, assets, ...).
    """
    url = f"https://api.github.com/repos/{github_repo}/releases/tags/{release_tag}"
    response = requests.get(url, headers=GITHUB_HEADERS, timeout=30)
    if response.status_code == 404:
        raise ValueError(f"Release '{release_tag}' not found in {github_repo}.")
    response.raise_for_status()
    return response.json()


def download_manifest(release: dict) -> dict:
    """Download and parse release_manifest.json from a release's assets.

    Returns
    -------
    dict
        The parsed manifest: {"version": ..., "files": {path: sha256}}.
    """
    asset = next((a for a in release["assets"] if a["name"] == "release_manifest.json"), None)
    if asset is None:
        raise ValueError(
            f"No release_manifest.json asset found on release {release['tag_name']}. "
            "Was the 'Generate Release Manifest' workflow run for this release?"
        )
    response = requests.get(asset["browser_download_url"], headers=GITHUB_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def split_manifest(tracked_files: dict, pipelines: dict | None = None) -> tuple[dict, list[str]]:
    """Separate the manifest into filesystem-synced analytics and API-deployed pipelines.

    Everything under a pipeline directory is deliberately excluded from the filesystem
    sync, because OpenHEXA runs pipelines from their registered version's zip and never
    from the workspace bucket. A copy in the bucket is inert and actively misleading.

    Two manifest generations are handled. Releases from 2026-09-21 onward carry a
    `pipelines` block naming the directories outright. Older manifests tracked only
    `<name>/pipeline.py`, so the directories are recovered from those entries instead;
    the exclusion is by directory either way, which is what keeps this correct if an old
    manifest is ever read alongside a new one.

    Parameters
    ----------
    tracked_files : dict
        The manifest's `files` map, `{repository path: sha256}`.
    pipelines : dict | None
        The manifest's `pipelines` block, if the release has one.

    Returns
    -------
    tuple[dict, list[str]]
        (the analytics files to copy, keyed by path; the pipeline directory names to deploy).
    """
    if pipelines:
        pipeline_dirs = set(pipelines)
    else:
        pipeline_dirs = {
            Path(p).parts[0]
            for p in tracked_files
            if len(Path(p).parts) == 2 and Path(p).parts[1] == "pipeline.py"
        }

    analytics = {
        rel_path: checksum
        for rel_path, checksum in tracked_files.items()
        if Path(rel_path).parts[0] not in pipeline_dirs
    }
    return analytics, sorted(pipeline_dirs)


def filter_pipelines(pipeline_dirs: list[str], only_pipelines: str | None) -> list[str]:
    """Restrict deployment to an explicitly named subset of the release's pipelines.

    An unknown name is an error rather than a silent no-op: a typo would otherwise look
    like a successful run that deployed nothing.

    Returns
    -------
    list[str]
        The selected pipeline directory names, or all of them if no subset was given.
    """
    if not only_pipelines or not only_pipelines.strip():
        return pipeline_dirs

    wanted = {name.strip().replace("-", "_") for name in only_pipelines.split(",") if name.strip()}
    unknown = sorted(wanted - set(pipeline_dirs))
    if unknown:
        raise ValueError(
            f"'Only these pipelines' names {unknown}, which the release does not define. "
            f"Available: {pipeline_dirs}"
        )

    selected = [dir_name for dir_name in pipeline_dirs if dir_name in wanted]
    current_run.log_info(f"Restricted to {len(selected)} of {len(pipeline_dirs)} pipeline(s): {selected}")
    return selected


def download_and_extract_tarball(tarball_url: str, extract_to: Path) -> Path:
    """Download a GitHub source tarball and extract it.

    One request for the whole repository: a per-file Contents API fetch would need one
    call per tracked file, against an unauthenticated limit of 60 per hour.

    Returns
    -------
    Path
        The path to the extracted repository root (GitHub tarballs contain one top-level
        directory named "<owner>-<repo>-<short_sha>").
    """
    response = requests.get(tarball_url, headers=GITHUB_HEADERS, timeout=120, stream=True)
    response.raise_for_status()

    tarball_path = extract_to / "release.tar.gz"
    with tarball_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    with tarfile.open(tarball_path) as tar:
        tar.extractall(path=extract_to)

    subdirs = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly one extracted directory, found: {subdirs}")
    return subdirs[0]


def sync_files(
    tracked_files: dict, source_root: Path, dest_root: Path, archive_dir: Path | None, dry_run: bool
) -> tuple[list[str], list[str]]:
    """Copy every tracked analytics file from the extracted tarball into the workspace.

    Existing files are moved under `archive_dir` before being overwritten, if one is given.

    Returns
    -------
    tuple[list[str], list[str]]
        (paths successfully copied, paths listed in the manifest but missing from the tarball).
    """
    copied, missing = [], []
    for rel_path in tracked_files:
        source_path = source_root / rel_path
        if not source_path.exists():
            missing.append(rel_path)
            current_run.log_warning(f"Manifest entry not found in tarball, skipping: {rel_path}")
            continue

        dest_path = dest_root / rel_path
        if dry_run:
            current_run.log_info(f"DRY RUN would sync: {rel_path}")
            copied.append(rel_path)
            continue

        if dest_path.exists() and archive_dir is not None:
            backup_path = archive_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest_path), str(backup_path))

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied.append(rel_path)

    return copied, missing


def deploy_all(
    tarball_root: Path,
    pipeline_dirs: list[str],
    release: dict,
    token: str,
    create_missing: bool,
    dry_run: bool,
) -> list[str]:
    """Register every release pipeline as a new version of the matching OpenHEXA pipeline.

    One pipeline's failure does not stop the others: a partial deployment is reported in
    full and re-running converges, whereas aborting halfway hides which ones still need it.

    Returns
    -------
    list[str]
        The directory names of the pipelines that could not be deployed.
    """
    failures = []
    for dir_name in pipeline_dirs:
        try:
            deploy_one(tarball_root / dir_name, dir_name, release, token, create_missing, dry_run)
        except Exception as exception:  # one bad pipeline must not stop the other nineteen
            current_run.log_error(f"[ERROR] {dir_name}: deployment failed: {exception}")
            failures.append(dir_name)

    if dry_run:
        current_run.log_info(f"Pipelines: {len(pipeline_dirs)} inspected, none registered (dry run).")
    else:
        current_run.log_info(
            f"Pipelines: {len(pipeline_dirs) - len(failures)}/{len(pipeline_dirs)} deployed."
        )
    return failures


def deploy_one(
    pipeline_dir: Path, dir_name: str, release: dict, token: str, create_missing: bool, dry_run: bool
) -> None:
    """Deploy a single pipeline directory as a new version, creating the pipeline if allowed.

    The OpenHEXA pipeline code is the directory name with underscores replaced by hyphens -
    the same slug this repo's CI passes to `openhexa pipelines push --code`.
    """
    if not (pipeline_dir / "pipeline.py").exists():
        raise ValueError(f"No pipeline.py in the release tarball at {dir_name}/.")

    code = dir_name.replace("_", "-")
    parsed = get_pipeline(pipeline_dir)
    existing = get_pipeline_by_code(token, code)

    if dry_run:
        action = "update" if existing else ("create" if create_missing else "SKIP (does not exist)")
        current_run.log_info(
            f"DRY RUN would {action}: {code} with {len(parsed.parameters)} parameter(s)."
        )
        return

    version_input = build_version_input(pipeline_dir, parsed, release)

    if existing:
        registered = upload_version(token, code, version_input)
        current_run.log_info(
            f"{code}: updated from version {existing['currentVersion']['versionNumber']} "
            f"to {registered['versionNumber']} ('{registered['versionName']}')."
        )
        return

    if not create_missing:
        raise ValueError(
            f"Pipeline '{code}' does not exist in workspace '{workspace.slug}'. Re-run with "
            "'Create pipelines that do not exist yet' enabled to bootstrap it."
        )

    created = create_pipeline_with_version(token, parsed.name, version_input)
    if created["code"] != code:
        raise ValueError(
            f"Created pipeline got code '{created['code']}', not the expected '{code}'. "
            f"OpenHEXA derives the code from the name '{parsed.name}'; delete the pipeline it "
            "just made and create it by hand in the UI with the right code."
        )
    current_run.log_info(f"{code}: created and seeded with version {version_input['name']}.")


def build_version_input(pipeline_dir: Path, parsed: Pipeline, release: dict) -> dict:
    """Build the GraphQL version input for a pipeline directory, the way the CLI does.

    The whole directory is zipped, so `requirements.txt` and `readme.md` travel with the
    code. Since 2026-09-21 the release manifest describes all of them, and lists them per
    pipeline in its `pipelines` block, so what ships here is verifiable after the fact.

    Returns
    -------
    dict
        The name, description, zipfile, parameters and timeout fields of the version input.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(pipeline_dir.glob("**/*")):
            if path.suffix.lower() in ZIPPED_SUFFIXES:
                archive.write(path, path.relative_to(pipeline_dir))
    buffer.seek(0)

    version_input = {
        "name": release["tag_name"],
        "description": f"Deployed by snt_workspace_manager from release {release['tag_name']}.",
        "externalLink": release["html_url"],
        "zipfile": base64.b64encode(buffer.read()).decode("ascii"),
        "parameters": [p.to_dict() for p in parsed.parameters],
        "timeout": parsed.timeout,
    }
    if parsed.functional_type:
        version_input["functionalType"] = parsed.functional_type
    return version_input


def get_pipeline_by_code(token: str, code: str) -> dict | None:
    """Look up a pipeline in this workspace by its code.

    Returns
    -------
    dict | None
        The pipeline with its current version number, or None if the workspace has no
        pipeline with that code.
    """
    return call_graphql(
        token,
        "query ($slug: String!, $code: String!) { pipelineByCode(workspaceSlug: $slug, code: $code)"
        " { id code currentVersion { versionNumber } } }",
        {"slug": workspace.slug, "code": code},
    )["pipelineByCode"]


def upload_version(token: str, code: str, version_input: dict) -> dict:
    """Register a new version of an existing pipeline.

    Returns
    -------
    dict
        The registered pipeline version (versionNumber, versionName).
    """
    data = call_graphql(
        token,
        "mutation ($input: UploadPipelineInput!) { uploadPipeline(input: $input)"
        " { success errors pipelineVersion { id versionName versionNumber } } }",
        {"input": {"workspaceSlug": workspace.slug, "code": code, **version_input}},
    )["uploadPipeline"]

    if not data["success"]:
        raise RuntimeError(f"uploadPipeline refused for '{code}': {data['errors']}")
    return data["pipelineVersion"]


def create_pipeline_with_version(token: str, pipeline_name: str, version_input: dict) -> dict:
    """Create a pipeline and seed it with its first version in one atomic call.

    Returns
    -------
    dict
        The created pipeline (id, code).
    """
    data = call_graphql(
        token,
        "mutation ($input: CreatePipelineInput!) { createPipeline(input: $input)"
        " { success errors pipeline { id code } } }",
        {"input": {"workspaceSlug": workspace.slug, "name": pipeline_name, "version": version_input}},
    )["createPipeline"]

    if not data["success"]:
        raise RuntimeError(f"createPipeline refused for '{pipeline_name}': {data['errors']}")
    return data["pipeline"]


def call_graphql(token: str, operation: str, variables: dict) -> dict:
    """Call the OpenHEXA GraphQL API with an explicit bearer token, reporting errors usefully.

    The SDK's own `graphql()` helper raises a bare HTTPError on a 4xx and discards the
    response body, which is where GraphQL puts the actual reason.

    Returns
    -------
    dict
        The `data` object of the GraphQL response.
    """
    response = requests.post(
        f"{os.environ['HEXA_SERVER_URL'].rstrip('/')}/graphql/",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": operation, "variables": variables},
        timeout=120,
    )
    if response.status_code != 200:
        current_run.log_error(f"HTTP {response.status_code} from the OpenHEXA API: {response.text[:2000]}")
        response.raise_for_status()

    body = response.json()
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body["data"]


def write_release_marker(snt_root_path: Path, release_tag: str) -> None:
    """Record the currently-deployed release tag in a hidden workspace-root file."""
    marker_path = snt_root_path / ".snt_release"
    marker_path.write_text(json.dumps({"snt_release": release_tag}, indent=2))
    current_run.log_info(f"Wrote release marker: {marker_path}")


if __name__ == "__main__":
    snt_workspace_manager()
