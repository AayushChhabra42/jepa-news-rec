"""Download MIND dataset via Kaggle API (Azure Blob URLs are no longer public).

The dataset is mirrored at: https://www.kaggle.com/datasets/arashnic/mind-news-dataset

Prerequisites (run once in Colab):
    !pip install kaggle
    # Upload your kaggle.json or set env vars:
    #   import os
    #   os.environ['KAGGLE_USERNAME'] = 'your_username'
    #   os.environ['KAGGLE_KEY'] = 'your_api_key'
"""

import os
import glob
import shutil
import zipfile
import subprocess
import sys


KAGGLE_DATASET = "arashnic/mind-news-dataset"


def _ensure_kaggle():
    """Ensure kaggle package is available."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[!] Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])


def download_mind_kaggle(raw_dir: str = "data/raw", dataset: str = "mind-small") -> dict[str, str]:
    """Download MIND-small from Kaggle and organise into train/dev splits.

    The Kaggle mirror (arashnic/mind-news-dataset) contains both
    MINDsmall_train.zip and MINDsmall_dev.zip inside a single download.

    Uses the kaggle Python API directly (not the CLI, which requires
    __main__.py that the kaggle package doesn't provide).

    Args:
        raw_dir: Root directory for raw data.
        dataset: "mind-small" (only small is on this Kaggle mirror).

    Returns:
        Dict mapping split name → extracted directory path.
    """
    _ensure_kaggle()

    train_dir = os.path.join(raw_dir, dataset, "train")
    dev_dir = os.path.join(raw_dir, dataset, "dev")

    # Skip if already downloaded and extracted
    if (os.path.isdir(train_dir) and os.listdir(train_dir) and
            os.path.isdir(dev_dir) and os.listdir(dev_dir)):
        print(f"[✓] Dataset already extracted:")
        print(f"    train: {train_dir}")
        print(f"    dev:   {dev_dir}")
        return {"train": train_dir, "dev": dev_dir}

    # Download from Kaggle using the Python API
    kaggle_dl_dir = os.path.join(raw_dir, "kaggle_download")
    os.makedirs(kaggle_dl_dir, exist_ok=True)

    print(f"[↓] Downloading MIND dataset from Kaggle ({KAGGLE_DATASET})...")
    print("    (Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, or ~/.kaggle/kaggle.json exists)")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=kaggle_dl_dir, unzip=False)

    # Find the downloaded zip(s)
    zips = glob.glob(os.path.join(kaggle_dl_dir, "*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip files found in {kaggle_dl_dir} after download.")

    # Extract the outer zip (Kaggle wraps everything in one zip)
    print(f"[⤓] Extracting outer archive...")
    staging_dir = os.path.join(raw_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    for z in zips:
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(staging_dir)

    # Now look for MINDsmall_train.zip and MINDsmall_dev.zip inside staging
    # The Kaggle dataset may have them directly or inside a subfolder
    inner_zips = {}
    for root, dirs, files in os.walk(staging_dir):
        for f in files:
            fl = f.lower()
            if "train" in fl and fl.endswith(".zip"):
                inner_zips["train"] = os.path.join(root, f)
            elif ("dev" in fl or "valid" in fl) and fl.endswith(".zip"):
                inner_zips["dev"] = os.path.join(root, f)

    # Also check if the TSV files are directly in staging (no inner zips)
    tsv_files = glob.glob(os.path.join(staging_dir, "**", "*.tsv"), recursive=True)

    extracted_dirs = {}

    if inner_zips:
        # Extract inner zips
        for split, zip_path in inner_zips.items():
            extract_dir = os.path.join(raw_dir, dataset, split)
            os.makedirs(extract_dir, exist_ok=True)
            print(f"[⤓] Extracting {split} from {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            extracted_dirs[split] = extract_dir
            print(f"[✓] {split} → {extract_dir}")
    elif tsv_files:
        # TSVs are directly available — figure out the split structure
        print(f"[i] Found {len(tsv_files)} TSV files directly in staging.")
        # Try to find train/dev structure
        for tsv in tsv_files:
            rel = os.path.relpath(tsv, staging_dir).lower()
            if "train" in rel:
                dest_dir = train_dir
                split = "train"
            elif "dev" in rel or "valid" in rel:
                dest_dir = dev_dir
                split = "dev"
            else:
                continue
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(tsv))
            shutil.copy2(tsv, dest)
            extracted_dirs[split] = dest_dir

        if not extracted_dirs:
            # Fallback: just copy everything to train
            print("[!] Could not determine split structure. Copying all to train/")
            os.makedirs(train_dir, exist_ok=True)
            for tsv in tsv_files:
                shutil.copy2(tsv, train_dir)
            extracted_dirs["train"] = train_dir
    else:
        raise FileNotFoundError(
            f"Could not find MIND data files in {staging_dir}. "
            f"Contents: {os.listdir(staging_dir)}"
        )

    # Clean up staging
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\n[✓] MIND dataset ready:")
    for split, path in extracted_dirs.items():
        num_files = len(os.listdir(path))
        print(f"    {split}: {path} ({num_files} files)")

    return extracted_dirs


def download_mind_manual(raw_dir: str = "data/raw", dataset: str = "mind-small") -> dict[str, str]:
    """Fallback: guide user to manually download if Kaggle auth fails.

    Also supports direct URL download if Azure comes back online.
    """
    train_dir = os.path.join(raw_dir, dataset, "train")
    dev_dir = os.path.join(raw_dir, dataset, "dev")

    print("=" * 60)
    print("  MIND Dataset — Manual Download Instructions")
    print("=" * 60)
    print()
    print("The Azure Blob URLs are no longer publicly accessible.")
    print("Please download the dataset from one of these sources:")
    print()
    print("  Option 1: Kaggle (recommended)")
    print("    https://www.kaggle.com/datasets/arashnic/mind-news-dataset")
    print("    → Download, then extract MINDsmall_train.zip and MINDsmall_dev.zip")
    print()
    print("  Option 2: Set up Kaggle API credentials and re-run with --source kaggle")
    print("    !pip install kaggle")
    print('    os.environ["KAGGLE_USERNAME"] = "your_username"')
    print('    os.environ["KAGGLE_KEY"] = "your_api_key"')
    print()
    print(f"  After downloading, place the extracted files in:")
    print(f"    Train: {os.path.abspath(train_dir)}/")
    print(f"    Dev:   {os.path.abspath(dev_dir)}/")
    print()
    print("  Each directory should contain: news.tsv, behaviors.tsv,")
    print("  entity_embedding.vec, relation_embedding.vec")
    print("=" * 60)

    return {}


def download_mind(
    dataset: str = "mind-small",
    raw_dir: str = "data/raw",
    source: str = "kaggle",
) -> dict[str, str]:
    """Download MIND dataset.

    Args:
        dataset: "mind-small" or "mind-large".
        raw_dir: Directory to store raw data.
        source: "kaggle" or "manual".

    Returns:
        Dict mapping split name to extracted directory path.
    """
    if source == "kaggle":
        try:
            return download_mind_kaggle(raw_dir, dataset)
        except Exception as e:
            print(f"\n[!] Kaggle download failed: {e}")
            print("[!] Falling back to manual instructions.\n")
            return download_mind_manual(raw_dir, dataset)
    else:
        return download_mind_manual(raw_dir, dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MIND dataset")
    parser.add_argument("--dataset", default="mind-small", choices=["mind-small", "mind-large"])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--source", default="kaggle", choices=["kaggle", "manual"])
    args = parser.parse_args()

    dirs = download_mind(args.dataset, args.raw_dir, args.source)
    if dirs:
        print(f"\nDownloaded splits: {list(dirs.keys())}")
        for split, path in dirs.items():
            print(f"  {split}: {path}")
