from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_REPO_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face"
DATA_DIR_URL = f"{DATASET_REPO_URL}/resolve/main/data"
DEFAULT_FILES = (
    "WIDER_train.zip",
    "WIDER_val.zip",
    "WIDER_test.zip",
    "wider_face_split.zip",
)
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the WIDER FACE train, validation, test, and split archives."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where WIDER FACE zip files are stored. Defaults to data/.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract each archive into --output-dir after downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when the target archive already exists.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def build_file_url(filename: str) -> str:
    return f"{DATA_DIR_URL}/{filename}"


def download_file(url: str, target: Path, token: str | None = None, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"skip existing {target}")
        return target

    part_path = target.with_suffix(target.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = Request(url, headers=headers)
    mode = "ab" if resume_from else "wb"
    with urlopen(request) as response, part_path.open(mode) as file:
        if resume_from and getattr(response, "status", 200) != 206:
            file.seek(0)
            file.truncate()
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file.write(chunk)

    part_path.replace(target)
    print(f"downloaded {target}")
    return target


def safe_extract_zip(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_root = output_dir.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            destination = (destination_root / member.filename).resolve()
            if destination != destination_root and destination_root not in destination.parents:
                raise ValueError(f"Unsafe zip member in {archive}: {member.filename}")
        zip_file.extractall(destination_root)
    print(f"extracted {archive} -> {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for filename in DEFAULT_FILES:
        archive = download_file(
            build_file_url(filename),
            output_dir / filename,
            token=args.token,
            force=args.force,
        )
        if args.extract:
            safe_extract_zip(archive, output_dir)


if __name__ == "__main__":
    main()
