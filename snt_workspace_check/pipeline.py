"""Check what is actually deployed in this OpenHEXA workspace against one release manifest.

Read-only. This pipeline writes its own report and nothing else - it never deletes, moves,
overwrites, deploys or archives. `snt_workspace_manager` is the only component that changes
workspace state (docs/wip/PRODUCT_SPEC.md section 5.4).

Phase 1 of the build plan (PRODUCT_SPEC.md section 6): verification mode against a single
target release, both sources hashed, four statuses, report written. Deliberately not here:

    mismatch_known / position   need every release's manifest - blocked on section 7.2
    untracked / not_covered     need a filesystem walk rather than manifest-driven iteration
    removed_in_target           needs a second manifest
    attribution mode            phase 3

The iteration is driven by the target manifest, not by what is on disk. That is what keeps
phase 1 to one manifest fetch, and it is why no status about unknown *paths* can appear yet.

Two sources are hashed, and each tracked path is routed to exactly ONE of them:

    filesystem        files under workspace.files_path, at their repository-relative paths
    pipeline_version  the files inside each pipeline's CURRENT registered version zip

Anything under a pipeline directory is read from the zip only. A copy of such a file on the
workspace filesystem is inert - OpenHEXA runs pipelines from the registered version, never
from the bucket - so consulting it would report a file as fine on the strength of bytes that
never execute. Reporting those inert copies is phase 2's job, not a reason to read them here.

Credentials: none. A run's own HEXA_TOKEN reads `currentVersion.zipfile` in full, established
by the snt-token-probe run of 2026-09-22 (docs/wip/HISTORY.md section 2.4). The checker can
therefore run unattended in a country workspace holding no connection at all.
"""

import base64
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests
from openhexa.sdk import current_run, parameter, pipeline, workspace

# Bumped only when the report shape changes in a way a consumer must notice. Frozen at
# phase 4; until then the shape is provisional (PRODUCT_SPEC.md section 5.5).
SCHEMA_VERSION = 1

REPORT_DIR_NAME = "snt_status"
LATEST_REPORT_NAME = "status_latest.json"
RELEASE_MARKER_NAME = ".snt_release"
MANIFEST_ASSET_NAME = "release_manifest.json"

GITHUB_HEADERS = {"User-Agent": "snt-workspace-check"}

# OpenHEXA does not return a version's name as it was submitted: it appends the version
# number, so a version deployed as "v0.1.0-test" reads back as "v0.1.0-test [v1]". Observed
# in the first real run, 2026-09-22, where an exact comparison silently disabled the whole
# name-versus-content check (HISTORY.md section 1). The tag is recovered by stripping the
# suffix; a version whose name never carried a tag (a plain CLI push reads back as "v3")
# simply fails to match any tag, which is the right answer.
VERSION_NUMBER_SUFFIX = re.compile(r"\s*\[v\d+\]\s*$")

# Human-readable display text, never parsed by a consumer (PRODUCT_SPEC.md section 5.5).
REMEDIATION = {
    "match": None,
    "unknown_content": (
        "This copy is not any version this release ever shipped - it was edited in place, or it "
        "is corrupt. Run snt_workspace_manager at the target release to restore it; the current "
        "copy is archived first."
    ),
    "missing": (
        "The target release ships this file and the workspace does not have it. Run "
        "snt_workspace_manager at the target release to install it."
    ),
    "unreadable": (
        "The file could not be read, so nothing is known about its contents. Check permissions "
        "on the filesystem, or the API error recorded in this report's `errors` list."
    ),
}


@pipeline("snt_workspace_check")
@parameter(
    "github_repo",
    name="GitHub repository",
    help="owner/repo holding the releases to check against (e.g. BLSQ/snt_development_sandbox)",
    type=str,
    default="BLSQ/snt_development_sandbox",
    required=True,
)
@parameter(
    "release_tag",
    name="Target release tag",
    help=(
        "Release to check this workspace against (e.g. v0.2.1-test). Leave empty to use the tag "
        "recorded in .snt_release by the last snt_workspace_manager run."
    ),
    type=str,
    default=None,
    required=False,
)
def snt_workspace_check(github_repo: str, release_tag: str | None) -> None:
    """Hash this workspace against one release manifest and write a status report.

    Orchestration only: resolves the target release, downloads its manifest, delegates the
    hashing of each source to a helper, and writes the report.
    """
    snt_root_path = Path(workspace.files_path)

    declared_tag = read_release_marker(snt_root_path)
    target_tag, resolved_from = resolve_target(release_tag, declared_tag)
    current_run.log_info(
        f"Checking workspace '{workspace.slug}' against {github_repo}@{target_tag} "
        f"(target resolved from the {resolved_from})."
    )
    if declared_tag and declared_tag != target_tag:
        current_run.log_warning(
            f"The workspace declares release '{declared_tag}' but is being checked against "
            f"'{target_tag}'. The declaration records intent, not verified fact."
        )

    release = get_release(github_repo, target_tag)
    manifest = download_manifest(release)
    tracked_files = manifest["files"]
    pipeline_specs = read_pipelines_block(manifest, target_tag)

    analytics_files = split_filesystem_files(tracked_files, pipeline_specs)
    current_run.log_info(
        f"Release {target_tag} tracks {len(tracked_files)} file(s): {len(analytics_files)} on the "
        f"filesystem and {len(tracked_files) - len(analytics_files)} inside "
        f"{len(pipeline_specs)} pipeline version zip(s)."
    )

    entries = check_filesystem(analytics_files, snt_root_path)

    token = get_run_token()
    zip_entries, pipeline_reports, errors = check_pipeline_versions(
        pipeline_specs, tracked_files, target_tag, token
    )
    entries.extend(zip_entries)

    report = build_report(
        entries=entries,
        pipeline_reports=pipeline_reports,
        errors=errors,
        github_repo=github_repo,
        release=release,
        resolved_from=resolved_from,
        declared_tag=declared_tag,
    )
    report_path = write_report(snt_root_path, report)

    log_summary(report, report_path)


def resolve_target(release_tag: str | None, declared_tag: str | None) -> tuple[str, str]:
    """Decide which release this run checks against, and record where that came from.

    The order is fixed by PRODUCT_SPEC.md section 4.2: the parameter, then the .snt_release
    marker. The third case - neither given - is attribution mode, which is phase 3, so here
    it stops with an explanation rather than silently checking against nothing.

    Returns
    -------
    tuple[str, str]
        (the target release tag, the source it was resolved from: "parameter" or "marker").
    """
    if release_tag and release_tag.strip():
        return release_tag.strip(), "parameter"
    if declared_tag:
        return declared_tag, "marker"
    raise ValueError(
        f"No target release: the 'Target release tag' parameter is empty and this workspace has "
        f"no {RELEASE_MARKER_NAME} marker. Attribution mode - assessing a workspace with no target "
        "at all - is not built yet (PRODUCT_SPEC.md phase 3). Name a release tag to continue."
    )


def read_release_marker(snt_root_path: Path) -> str | None:
    """Read the release tag the last snt_workspace_manager run claims to have deployed.

    A workspace with no marker is the normal starting state for every country workspace
    today, so its absence is reported, never treated as an error.

    Returns
    -------
    str | None
        The declared release tag, or None if there is no readable marker.
    """
    marker_path = snt_root_path / RELEASE_MARKER_NAME
    if not marker_path.exists():
        current_run.log_info(f"No {RELEASE_MARKER_NAME} marker - this workspace declares no release.")
        return None

    try:
        declared = json.loads(marker_path.read_text())["snt_release"]
    except (OSError, ValueError, KeyError) as exception:
        current_run.log_warning(
            f"[WARNING] {RELEASE_MARKER_NAME} exists but could not be read: {exception}. "
            "Treating the workspace as declaring no release."
        )
        return None

    current_run.log_info(f"Workspace declares release '{declared}' in {RELEASE_MARKER_NAME}.")
    return declared


def get_run_token() -> str:
    """Read this run's own OpenHEXA API token from the environment.

    Deployment needs a workspace-scoped token from a connection, but reading does not: a
    run's own HEXA_TOKEN returns `currentVersion.zipfile` in full (HISTORY.md section 2.4).
    Keeping the checker credential-free is what lets it run unattended anywhere.

    Returns
    -------
    str
        The bearer token for the OpenHEXA GraphQL API.
    """
    token = os.environ.get("HEXA_TOKEN")
    if not token:
        raise RuntimeError(
            "HEXA_TOKEN is not set in this run's environment, so the pipeline version zips "
            "cannot be read. This variable is provided by the OpenHEXA runner."
        )
    return token


def get_release(github_repo: str, release_tag: str) -> dict:
    """Fetch a GitHub release's metadata by tag.

    Copied from snt_workspace_manager: pipelines are deployed as independent zips and cannot
    share a module, so the two carry the same helper by design.

    Returns
    -------
    dict
        The GitHub API release object (tag_name, published_at, assets, ...).
    """
    url = f"https://api.github.com/repos/{github_repo}/releases/tags/{release_tag}"
    response = requests.get(url, headers=GITHUB_HEADERS, timeout=30)
    if response.status_code == 404:
        raise ValueError(f"Release '{release_tag}' not found in {github_repo}.")
    response.raise_for_status()
    return response.json()


def download_manifest(release: dict) -> dict:
    """Download and parse release_manifest.json from a release's assets.

    Copied from snt_workspace_manager - see get_release for why.

    Returns
    -------
    dict
        The parsed manifest: {"version": ..., "files": {path: sha256}, "pipelines": {...}}.
    """
    asset = next((a for a in release["assets"] if a["name"] == MANIFEST_ASSET_NAME), None)
    if asset is None:
        raise ValueError(
            f"Release {release['tag_name']} has no {MANIFEST_ASSET_NAME} asset, so there is "
            "nothing to check against. Was the 'Generate Release Manifest' workflow run for it? "
            "A target release with no manifest cannot be checked; pick another tag."
        )
    response = requests.get(asset["browser_download_url"], headers=GITHUB_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def read_pipelines_block(manifest: dict, target_tag: str) -> dict:
    """Return the manifest's `pipelines` block, refusing a manifest that has none.

    The block says which tracked paths ship inside which pipeline zip, and under what name
    once inside it (PRODUCT_SPEC.md section 2.1). Without it a consumer has to re-derive the
    split by reapplying the generator's rule, and one that gets it wrong reports `missing`
    for files that are deployed and correct.

    snt_workspace_manager carries a fallback for pre-phase-0 manifests. It is deliberately
    not copied here: no live release lacks the block, so a fallback would be untested code
    for a case that can no longer occur (PRODUCT_SPEC.md section 2.1, still open).

    Returns
    -------
    dict
        {pipeline directory: {"code": <openhexa slug>, "zip_files": [<path in zip>, ...]}}.
    """
    pipelines = manifest.get("pipelines")
    if not pipelines:
        raise ValueError(
            f"The manifest for {target_tag} has no 'pipelines' block, so it predates the phase-0 "
            "generator (2026-09-21) and does not describe what each pipeline zip ships. This "
            "checker deliberately has no fallback for that - check against a later release."
        )
    return pipelines


def split_filesystem_files(tracked_files: dict, pipeline_specs: dict) -> dict:
    """Select the manifest entries that are expected on the workspace filesystem.

    The split is by directory, not by suffix: everything under a pipeline directory belongs
    to that pipeline's version zip, whatever it is called.

    Returns
    -------
    dict
        The filesystem-sourced subset of the manifest, {repository path: sha256}.
    """
    pipeline_dirs = set(pipeline_specs)
    return {
        rel_path: checksum
        for rel_path, checksum in tracked_files.items()
        if Path(rel_path).parts[0] not in pipeline_dirs
    }


def make_entry(
    path: str,
    source: str,
    pipeline_name: str | None,
    status: str,
    observed_sha256: str | None,
    target_sha256: str | None,
) -> dict:
    """Build one report entry, with every phase-2/3 field present as null.

    Fields are never omitted, so a consumer can read them unconditionally
    (PRODUCT_SPEC.md section 5.5).

    Returns
    -------
    dict
        One entry of the report's `entries` list.
    """
    return {
        "path": path,
        "source": source,
        "pipeline": pipeline_name,
        "status": status,
        "observed_sha256": observed_sha256,
        "target_sha256": target_sha256,
        "matching_releases": None,  # phase 3 - needs every release's manifest
        "position": None,  # phase 3 - needs release ordering
        "remediation": REMEDIATION[status],
    }


def sha256_bytes(payload: bytes) -> str:
    """Hash raw bytes, with no normalisation of any kind.

    Notebooks are hashed as they are on purpose: whether to normalise them is decision D9,
    to be taken after measuring real drift in phase 2 rather than guessed at now.

    Returns
    -------
    str
        The lowercase hex sha256 digest.
    """
    return hashlib.sha256(payload).hexdigest()


def check_filesystem(tracked_files: dict, snt_root_path: Path) -> list[dict]:
    """Hash every filesystem-sourced manifest entry and classify it against the target.

    Returns
    -------
    list[dict]
        One report entry per tracked path, in manifest path order.
    """
    entries = []
    for rel_path, expected in sorted(tracked_files.items()):
        file_path = snt_root_path / rel_path

        if not file_path.is_file():
            entries.append(make_entry(rel_path, "filesystem", None, "missing", None, expected))
            continue

        try:
            observed = sha256_bytes(file_path.read_bytes())
        except OSError as exception:
            current_run.log_warning(f"[WARNING] Could not read {rel_path}: {exception}")
            entries.append(make_entry(rel_path, "filesystem", None, "unreadable", None, expected))
            continue

        status = "match" if observed == expected else "unknown_content"
        entries.append(make_entry(rel_path, "filesystem", None, status, observed, expected))

    return entries


def check_pipeline_versions(
    pipeline_specs: dict, tracked_files: dict, target_tag: str, token: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Hash the contents of every pipeline's current registered version zip.

    Only the current version is read. Older versions are history - they are not what would
    run, and reporting on them would drown the report in files nobody can act on
    (PRODUCT_SPEC.md section 3.1).

    Returns
    -------
    tuple[list[dict], list[dict], list[dict]]
        (report entries, one per-pipeline summary block each, errors encountered).
    """
    entries: list[dict] = []
    pipeline_reports: list[dict] = []
    errors: list[dict] = []

    for dir_name, spec in sorted(pipeline_specs.items()):
        code = spec["code"]
        expected = {
            member: tracked_files[f"{dir_name}/{member}"]
            for member in spec["zip_files"]
            if f"{dir_name}/{member}" in tracked_files
        }

        try:
            version = fetch_current_version(token, code)
        except Exception as exception:  # one unreachable pipeline must not hide the other 21
            message = f"Could not read the current version of pipeline '{code}': {exception}"
            current_run.log_error(f"[ERROR] {message}")
            errors.append({"scope": f"pipeline_version:{code}", "message": message})
            entries.extend(
                make_entry(f"{dir_name}/{m}", "pipeline_version", dir_name, "unreadable", None, sha)
                for m, sha in sorted(expected.items())
            )
            pipeline_reports.append(pipeline_report(dir_name, code, None, None, 0))
            continue

        if version is None:
            current_run.log_warning(
                f"[WARNING] {code}: the target release defines this pipeline but the workspace has "
                "no deployed version of it. All of its files are reported missing."
            )
            entries.extend(
                make_entry(f"{dir_name}/{m}", "pipeline_version", dir_name, "missing", None, sha)
                for m, sha in sorted(expected.items())
            )
            pipeline_reports.append(pipeline_report(dir_name, code, None, None, 0))
            continue

        members = read_zip_members(version["zipfile"])
        pipeline_entries = [
            classify_zip_member(dir_name, member, sha, members) for member, sha in sorted(expected.items())
        ]
        entries.extend(pipeline_entries)

        # Files in the zip that no manifest entry describes. In phase 2 these become
        # `untracked` or `not_covered` entries; here they are counted so the omission is
        # visible rather than silent.
        undescribed = sorted(set(members) - set(expected))
        pipeline_reports.append(
            pipeline_report(
                dir_name,
                code,
                version["versionName"],
                name_matches_content(version["versionName"], target_tag, pipeline_entries),
                len(undescribed),
            )
        )
        if undescribed:
            current_run.log_info(
                f"{code}: {len(undescribed)} file(s) in the deployed zip are not described by the "
                f"manifest and are not yet reported per file (phase 2): {undescribed}"
            )

    return entries, pipeline_reports, errors


def fetch_current_version(token: str, code: str) -> dict | None:
    """Read a pipeline's current registered version, including its zipped source.

    Returns
    -------
    dict | None
        {"versionNumber", "versionName", "zipfile"} for the current version, or None if the
        workspace has no pipeline with that code or no version registered against it.
    """
    data = call_graphql(
        token,
        "query ($slug: String!, $code: String!) { pipelineByCode(workspaceSlug: $slug, code: $code)"
        " { id code currentVersion { versionNumber versionName zipfile } } }",
        {"slug": workspace.slug, "code": code},
    )["pipelineByCode"]

    if data is None:
        return None
    return data["currentVersion"]


def call_graphql(token: str, operation: str, variables: dict) -> dict:
    """Call the OpenHEXA GraphQL API with an explicit bearer token, reporting errors usefully.

    Copied from snt_workspace_manager - see get_release for why. The SDK's own `graphql()`
    helper raises a bare HTTPError on a 4xx and discards the response body, which is where
    GraphQL puts the actual reason.

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


def read_zip_members(encoded_zipfile: str) -> dict:
    """Decode a pipeline version's base64 zip and hash every file inside it.

    A 200 OK is not evidence on its own - GraphQL field-level denial returns a null field
    inside a successful response - so a zip that does not decode is an error, loudly.

    Returns
    -------
    dict
        {path inside the zip: sha256 of its contents}.
    """
    if not encoded_zipfile:
        raise ValueError(
            "The API returned an empty zipfile field for this version. That is how field-level "
            "permission denial presents itself; check the token before anything else."
        )

    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded_zipfile)))
    return {
        info.filename: sha256_bytes(archive.read(info.filename))
        for info in archive.infolist()
        if not info.is_dir()
    }


def classify_zip_member(dir_name: str, member: str, expected: str, members: dict) -> dict:
    """Classify one manifest-described file against what the deployed zip actually holds.

    Returns
    -------
    dict
        One report entry, sourced from `pipeline_version`.
    """
    path = f"{dir_name}/{member}"
    observed = members.get(member)
    if observed is None:
        return make_entry(path, "pipeline_version", dir_name, "missing", None, expected)

    status = "match" if observed == expected else "unknown_content"
    return make_entry(path, "pipeline_version", dir_name, status, observed, expected)


def claimed_release_tag(version_name: str) -> str:
    """Recover the release tag a pipeline version's name claims, ignoring OpenHEXA's suffix.

    Returns
    -------
    str
        The version name with a trailing " [v<number>]" removed, if it had one.
    """
    return VERSION_NUMBER_SUFFIX.sub("", version_name).strip()


def name_matches_content(version_name: str, target_tag: str, entries: list[dict]) -> bool | None:
    """Say whether a version's name is borne out by the bytes it actually contains.

    A version name is free text somebody typed; the hash is evidence. When they disagree the
    hash wins and the name is reported as misleading - a workspace whose version labels have
    stopped meaning anything looks perfectly healthy in the OpenHEXA UI, which shows only the
    names (PRODUCT_SPEC.md section 3.1).

    Phase 1 holds one manifest, so this can only be judged for a version that claims to BE
    the target. A version named after some other release is not evidence of anything until
    attribution mode can fetch that release's manifest, and is reported as null rather than
    guessed at.

    Returns
    -------
    bool | None
        True or False when the version claims the target release, None otherwise.
    """
    if claimed_release_tag(version_name) != target_tag:
        return None
    return all(entry["status"] == "match" for entry in entries)


def pipeline_report(
    dir_name: str,
    code: str,
    version_name: str | None,
    matches_content: bool | None,
    undescribed_count: int,
) -> dict:
    """Build the per-pipeline block that sits alongside the per-file entries.

    The name/content disagreement is a property of the deployment, not of any one file
    inside the zip, so it is reported here rather than as a per-file status.

    Returns
    -------
    dict
        One entry of the report's `pipelines` list.
    """
    return {
        "pipeline": dir_name,
        "code": code,
        # The raw API value, kept verbatim as the evidence it is, alongside the tag parsed
        # out of it - the two differ because OpenHEXA appends the version number.
        "current_version_name": version_name,
        "current_version_claims_tag": claimed_release_tag(version_name) if version_name else None,
        "version_name_matches_content": matches_content,
        "files_in_zip_not_in_manifest": undescribed_count,
    }


def build_report(
    entries: list[dict],
    pipeline_reports: list[dict],
    errors: list[dict],
    github_repo: str,
    release: dict,
    resolved_from: str,
    declared_tag: str | None,
) -> dict:
    """Assemble the report document from what each source produced.

    Returns
    -------
    dict
        The full report, in the PRODUCT_SPEC.md section 5.5 shape.
    """
    by_status: dict[str, int] = {}
    for entry in entries:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1

    incomplete = bool(errors) or any(entry["status"] == "unreadable" for entry in entries)
    target_tag = release["tag_name"]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        # Frozen at phase 4. A run cannot read the name of the pipeline version executing it,
        # so this stays null until the checker is deployed from a release tag by the manager.
        "checker_version": None,
        "workspace": workspace.slug,
        "repo": github_repo,
        "mode": "verification",
        "target_release": {
            "tag": target_tag,
            "resolved_from": resolved_from,
            "published_at": release.get("published_at"),
        },
        "declared_release": ({"tag": declared_tag, "source": RELEASE_MARKER_NAME} if declared_tag else None),
        "releases_considered": [
            {"tag": target_tag, "published_at": release.get("published_at"), "manifest_available": True}
        ],
        "incomplete": incomplete,
        "summary": {
            "by_status": by_status,
            "attribution": None,  # phase 3 - needs every release's manifest
        },
        "pipelines": pipeline_reports,
        "entries": entries,
        "errors": errors,
        # Named so the report never reads as a clean bill of health for things it never
        # looked at (PRODUCT_SPEC.md sections 1.2 and 5.4).
        "blind_spots": [
            "Files at paths no manifest describes are not reported yet (phase 2).",
            "Only the target release is considered, so a file that matches a different release "
            "is reported as unknown_content rather than mismatch_known (phase 3).",
            "Files removed in the target release are not reported (phase 2).",
            "OpenHEXA web apps, configuration/ and data/ are out of scope by decision.",
            "Country-specific notebook variants are not recognised as deliberate overrides.",
        ],
    }


def write_report(snt_root_path: Path, report: dict) -> Path:
    """Write the report twice: timestamped for history, and at the stable latest path.

    Returns
    -------
    Path
        The path of the timestamped copy.
    """
    report_dir = snt_root_path / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    # The colons of an ISO timestamp are legal in an object key but hostile in a filename on
    # a machine that later syncs this bucket, so they are replaced here and nowhere else -
    # `generated_at` inside the document keeps the exact ISO form.
    stamp = report["generated_at"].replace(":", "-")
    report_path = report_dir / f"status_{stamp}.json"

    payload = json.dumps(report, indent=2)
    report_path.write_text(payload)
    (report_dir / LATEST_REPORT_NAME).write_text(payload)

    return report_path


def log_summary(report: dict, report_path: Path) -> None:
    """Log the shape of the result, leaving the per-file detail to the JSON.

    With 100+ tracked files, per-file logging would be 300 unreadable lines; the requirement
    is a summary in the log and the full detail in the report (PRODUCT_SPEC.md section 5.4).
    """
    by_status = report["summary"]["by_status"]
    total = sum(by_status.values())
    current_run.log_info(
        f"Checked {total} tracked file(s) against {report['target_release']['tag']}: "
        + ", ".join(f"{count} {status}" for status, count in sorted(by_status.items()))
    )

    for entry in report["entries"]:
        if entry["status"] != "match":
            current_run.log_warning(f"[WARNING] {entry['status']}: {entry['path']} ({entry['source']})")

    for block in report["pipelines"]:
        if block["version_name_matches_content"] is False:
            current_run.log_warning(
                f"[WARNING] {block['code']}: its current version is named "
                f"'{block['current_version_name']}' but its contents are not that release's. "
                "The name is metadata someone typed; the hash is evidence."
            )

    if report["incomplete"]:
        current_run.log_warning(
            "[WARNING] This report is INCOMPLETE - at least one source could not be read. "
            "It is not a clean bill of health; see the 'errors' list."
        )

    current_run.log_info(f"Report written: {report_path} (and {REPORT_DIR_NAME}/{LATEST_REPORT_NAME})")


if __name__ == "__main__":
    snt_workspace_check()
