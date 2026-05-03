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
    MINDsmall_train and MINDsmall_dev data.

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
    if (os.path.isdir(train_dir) and os.path.isfile(os.path.join(train_dir, "news.tsv")) and
            os.path.isdir(dev_dir) and os.path.isfile(os.path.join(dev_dir, "news.tsv"))):
        print(f"[✓] Dataset already extracted:")
        print(f"    train: {train_dir}")
        print(f"    dev:   {dev_dir}")
        return {"train": train_dir, "dev": dev_dir}

    # Download from Kaggle using the Python API — unzip=True to extract all
    kaggle_dl_dir = os.path.join(raw_dir, "kaggle_download")
    os.makedirs(kaggle_dl_dir, exist_ok=True)

    print(f"[↓] Downloading MIND dataset from Kaggle ({KAGGLE_DATASET})...")
    print("    (Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, or ~/.kaggle/kaggle.json exists)")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=kaggle_dl_dir, unzip=True)

    # Debug: show everything that was extracted
    print(f"\n[i] Contents of {kaggle_dl_dir}:")
    all_files = []
    for root, dirs, files in os.walk(kaggle_dl_dir):
        for f in files:
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, kaggle_dl_dir)
            all_files.append((rel, fpath))
            print(f"    {rel}")

    # --- Strategy 1: Look for directories containing news.tsv ---
    extracted_dirs = {}
    for rel, fpath in all_files:
        if os.path.basename(fpath).lower() == "news.tsv":
            parent = os.path.dirname(fpath)
            parent_name = os.path.basename(parent).lower()
            # Also check grandparent in case structure is like MINDsmall_train/news.tsv
            grandparent_name = os.path.basename(os.path.dirname(parent)).lower()
            combined = f"{grandparent_name}/{parent_name}"

            if "train" in parent_name or "train" in grandparent_name:
                split = "train"
            elif "dev" in combined or "valid" in combined or "test" in combined:
                split = "dev"
            else:
                # Unknown — check the relative path
                if "train" in rel.lower():
                    split = "train"
                elif "dev" in rel.lower() or "valid" in rel.lower():
                    split = "dev"
                else:
                    continue

            dest_dir = train_dir if split == "train" else dev_dir
            if dest_dir not in [v for v in extracted_dirs.values()]:
                # Copy entire directory contents
                os.makedirs(dest_dir, exist_ok=True)
                for item in os.listdir(parent):
                    src = os.path.join(parent, item)
                    dst = os.path.join(dest_dir, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                extracted_dirs[split] = dest_dir
                print(f"[✓] {split} → {dest_dir} (from {parent})")

    # --- Strategy 2: Look for inner zip files if Strategy 1 found nothing ---
    if not extracted_dirs:
        print("[i] No news.tsv found directly. Looking for inner zip files...")
        inner_zips = {}
        for rel, fpath in all_files:
            fl = os.path.basename(fpath).lower()
            if fl.endswith(".zip"):
                if "train" in fl:
                    inner_zips["train"] = fpath
                elif "dev" in fl or "valid" in fl:
                    inner_zips["dev"] = fpath

        for split, zip_path in inner_zips.items():
            dest = train_dir if split == "train" else dev_dir
            os.makedirs(dest, exist_ok=True)
            print(f"[⤓] Extracting {split} from {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            extracted_dirs[split] = dest
            print(f"[✓] {split} → {dest}")

    if not extracted_dirs:
        raise FileNotFoundError(
            f"Could not find MIND data files (news.tsv) anywhere in {kaggle_dl_dir}. "
            f"Files found: {[r for r, _ in all_files]}"
        )

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
