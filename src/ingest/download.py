"""Download and extract a dated snapshot of the Service-Public.gouv.fr "Particuliers" XML feed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

FLUX_VERSION = "3.5"
ARCHIVE_URL = (
    f"https://lecomarquage.service-public.gouv.fr/vdd/{FLUX_VERSION}/part/zip/vosdroits-latest.zip"
)
SCHEMA_URL = f"https://echanges.dila.gouv.fr/OPENDATA/SERVICE-PUBLIC_DTD/schema_{FLUX_VERSION}.zip"
USER_AGENT = "rag-demarches-snapshot/0.1"
RAW_DIR = Path("data/raw")


def fetch(url: str, dest: Path) -> dict[str, str | int | None]:
    """Stream `url` to `dest` and return HTTP metadata useful for provenance."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
        return {
            "url": url,
            "final_url": response.geturl(),
            "last_modified": response.headers.get("Last-Modified"),
            "size_bytes": dest.stat().st_size,
        }


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path, meta: dict, n_files: int) -> dict:
    """Merge HTTP metadata with local file name, hash and member count."""
    return {**meta, "file": path.name, "sha256": sha256(path), "n_files": n_files}


def safe_extract(archive: Path, target: Path) -> int:
    """Extract a zip archive, refusing members that would land outside `target`."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)
    return sum(1 for m in members if not m.is_dir())


def download_snapshot(raw_dir: Path = RAW_DIR, date: str | None = None) -> Path:
    """Download, extract and describe one snapshot; skip if it already exists."""
    date = date or datetime.now(UTC).strftime("%Y%m%d")
    snapshot = raw_dir / date
    manifest_path = snapshot / "manifest.json"
    if manifest_path.exists():
        print(f"Snapshot already present: {snapshot}")
        return snapshot

    snapshot.mkdir(parents=True, exist_ok=True)
    archive = snapshot / f"vosdroits-{FLUX_VERSION}-{date}.zip"
    schema = snapshot / f"schema_{FLUX_VERSION}.zip"

    archive_meta = fetch(ARCHIVE_URL, archive)
    schema_meta = fetch(SCHEMA_URL, schema)
    n_xml = safe_extract(archive, snapshot / "xml")
    n_schema = safe_extract(schema, snapshot / "schema")

    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "flux_version_in_url": FLUX_VERSION,
        "archive": describe_file(archive, archive_meta, n_xml),
        "schema": describe_file(schema, schema_meta, n_schema),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Snapshot written to {snapshot} ({n_xml} files)")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--date", help="Snapshot folder name (default: today, UTC, YYYYMMDD)")
    args = parser.parse_args()
    download_snapshot(args.raw_dir, args.date)


if __name__ == "__main__":
    main()
