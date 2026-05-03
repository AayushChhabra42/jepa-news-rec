"""Download MIND dataset from Azure Blob Storage."""

import os
import zipfile
import requests
from tqdm import tqdm


MIND_URLS = {
    "mind-small": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    },
    "mind-large": {
        "train": "https://mind201910large.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910large.blob.core.windows.net/release/MINDlarge_dev.zip",
        "test": "https://mind201910large.blob.core.windows.net/release/MINDlarge_test.zip",
    },
}


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_mind(dataset: str = "mind-small", raw_dir: str = "data/raw") -> dict[str, str]:
    """Download and extract MIND dataset.

    Args:
        dataset: "mind-small" or "mind-large".
        raw_dir: Directory to store raw data.

    Returns:
        Dict mapping split name to extracted directory path.
    """
    urls = MIND_URLS[dataset]
    extracted_dirs = {}

    for split, url in urls.items():
        zip_path = os.path.join(raw_dir, f"{dataset}_{split}.zip")
        extract_dir = os.path.join(raw_dir, dataset, split)

        # Skip if already extracted
        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"[✓] {split} already extracted at {extract_dir}")
            extracted_dirs[split] = extract_dir
            continue

        # Download
        if not os.path.isfile(zip_path):
            print(f"[↓] Downloading {split} from {url}")
            download_file(url, zip_path)
        else:
            print(f"[✓] {split} zip already exists at {zip_path}")

        # Extract
        print(f"[⤓] Extracting {split} to {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_dirs[split] = extract_dir
        print(f"[✓] {split} extracted to {extract_dir}")

    return extracted_dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MIND dataset")
    parser.add_argument("--dataset", default="mind-small", choices=["mind-small", "mind-large"])
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    dirs = download_mind(args.dataset, args.raw_dir)
    print(f"\nDownloaded splits: {list(dirs.keys())}")
    for split, path in dirs.items():
        print(f"  {split}: {path}")
