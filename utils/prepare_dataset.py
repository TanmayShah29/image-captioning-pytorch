"""
Dataset Preparation Pipeline for Flickr8k
==========================================
SINGLE ENTRY POINT for all dataset handling.

Call prepare_dataset(data_root) and it will either:
  ✅  Fully prepare & validate the dataset  →  returns stats dict
  ❌  Crash with a clear, actionable error  →  never silent

CANONICAL OUTPUT CONTRACT
─────────────────────────
After successful preparation the repo uses ONLY:

    data/
    ├── images/          ← every .jpg in one flat folder
    └── captions.txt     ← image_name<TAB>caption  (no header)

All downstream code (dataset.py, train.py, verify.py) relies on this.
"""

import csv
import glob
import os
import shutil
from collections import defaultdict


# ======================================================================
# PUBLIC API  —  the only function callers need
# ======================================================================

def prepare_dataset(data_root="data"):
    """Detect, normalize, validate, and guarantee the Flickr8k dataset.

    Steps
    ─────
    1.  Locate images  (any folder layout)
    2.  Locate captions (any known format)
    3.  Normalize images  →  data/images/
    4.  Normalize captions → data/captions.txt
    5.  Cross-validate images ↔ captions
    6.  Print confirmation or crash with clear error

    Parameters
    ----------
    data_root : str
        Root directory to scan. Default ``"data"``.

    Returns
    -------
    dict
        ``{"images_dir", "captions_file", "num_images",
           "num_captions", "num_unique_images"}``

    Raises
    ------
    DatasetError
        With a human-readable explanation of WHAT failed, WHY, and HOW to fix it.
    """

    _print_banner("DATASET PREPARATION")

    # ── Check root exists ────────────────────────────────────────────
    if not os.path.isdir(data_root):
        _fail(
            what=f"Data root directory not found: '{data_root}'",
            why="The directory you specified does not exist.",
            fix=f"Create the directory and place Flickr8k files inside:\n"
                f"    mkdir -p {data_root}\n"
                f"    # then unzip Flickr8k into {data_root}/",
        )

    canonical_img_dir = os.path.join(data_root, "images")
    canonical_cap_file = os.path.join(data_root, "captions.txt")

    # ── Fast path: already prepared ──────────────────────────────────
    if _is_already_prepared(canonical_img_dir, canonical_cap_file):
        print("  ⚡ Dataset already in canonical format — skipping preparation.")
        stats = _validate_canonical(canonical_img_dir, canonical_cap_file)
        _print_success(stats)
        return stats

    # ── 1. Find images ───────────────────────────────────────────────
    print("\n── Step 1/5: Locating images ──")
    source_images = _find_all_images(data_root, exclude_dir=canonical_img_dir)
    if not source_images:
        _fail(
            what="No .jpg images found anywhere under '{}'".format(data_root),
            why="The dataset images are missing or in an unexpected location.",
            fix="Download Flickr8k and unzip it so that .jpg files are\n"
                "somewhere inside the '{}/' directory.\n"
                "  Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k".format(
                    data_root),
        )
    print(f"  Found {len(source_images)} .jpg images")

    # ── 2. Find captions ─────────────────────────────────────────────
    print("\n── Step 2/5: Locating captions ──")
    raw_caption_file = _find_caption_file(data_root, exclude_file=canonical_cap_file)
    if raw_caption_file is None:
        _fail(
            what="No caption file found under '{}'".format(data_root),
            why="Expected a .txt or .csv file containing image–caption pairs.",
            fix="Place one of these files inside '{}':\n"
                "  • Flickr8k.token.txt\n"
                "  • captions.txt  /  captions.csv\n"
                "  • Any .txt or .csv with image↔caption data".format(data_root),
        )
    print(f"  Found caption file: {raw_caption_file}")

    # ── 3. Parse captions ─────────────────────────────────────────────
    print("\n── Step 3/5: Parsing & normalizing captions ──")
    raw_pairs = _parse_any_caption_format(raw_caption_file)
    print(f"  Parsed {len(raw_pairs)} raw image–caption pairs")
    if not raw_pairs:
        _fail(
            what="Zero valid captions parsed from '{}'".format(raw_caption_file),
            why="The file exists but no lines matched any known caption format.\n"
                "  Supported formats:\n"
                "    1. image.jpg#0<TAB>caption      (Flickr8k token format)\n"
                "    2. image,caption                 (CSV with/without header)\n"
                "    3. image<TAB>caption             (TSV)",
            fix="Check the file content and ensure it matches one of the\n"
                "formats above. The file must be UTF-8 encoded.",
        )

    # ── 4. Copy images into canonical dir ─────────────────────────────
    print("\n── Step 4/5: Normalizing images → data/images/ ──")
    os.makedirs(canonical_img_dir, exist_ok=True)

    # Build a lookup: filename → full source path
    source_lookup = {}
    for path in source_images:
        name = os.path.basename(path)
        # If duplicate name, prefer the one NOT already in canonical dir
        if name not in source_lookup:
            source_lookup[name] = path

    # Copy missing images
    already_present = set(os.listdir(canonical_img_dir)) if os.path.isdir(canonical_img_dir) else set()
    copied = 0
    for name, src_path in source_lookup.items():
        if name not in already_present:
            dst = os.path.join(canonical_img_dir, name)
            try:
                shutil.copy2(src_path, dst)
                copied += 1
            except Exception as e:
                print(f"    ⚠ Could not copy {name}: {e}")

    total_in_dir = len([f for f in os.listdir(canonical_img_dir)
                        if f.lower().endswith((".jpg", ".jpeg"))])
    print(f"  Copied {copied} new images ({total_in_dir} total in data/images/)")

    if total_in_dir == 0:
        _fail(
            what="data/images/ is empty after normalization",
            why="Image copy failed — source images may be corrupted or permissions denied.",
            fix="Manually check the image files and directory permissions.",
        )

    # ── 5. Write canonical captions.txt ───────────────────────────────
    print("\n── Step 5/5: Writing canonical captions.txt ──")
    available_images = set(
        f for f in os.listdir(canonical_img_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )

    # Cross-validate: keep only pairs whose image exists
    before_count = len(raw_pairs)
    valid_pairs = [(img, cap) for img, cap in raw_pairs if img in available_images]
    after_count = len(valid_pairs)
    orphan_count = before_count - after_count

    print(f"  Captions before cross-validation: {before_count}")
    print(f"  Captions after  cross-validation: {after_count}")
    if orphan_count > 0:
        # Show a few orphan image names for debugging
        orphans = set(img for img, _ in raw_pairs if img not in available_images)
        sample = list(orphans)[:5]
        print(f"  Removed {orphan_count} captions for {len(orphans)} missing images")
        print(f"    Sample missing: {sample}")

    if not valid_pairs:
        _fail(
            what="Zero valid image–caption pairs after cross-validation",
            why="None of the images referenced in the caption file exist on disk.",
            fix="Ensure the image filenames in the caption file match the\n"
                "actual .jpg filenames in your dataset.\n"
                "  Caption references → image names like: {}\n"
                "  Available images   → names like: {}".format(
                    [p[0] for p in raw_pairs[:3]],
                    list(available_images)[:3]),
        )

    # Write
    with open(canonical_cap_file, "w", encoding="utf-8") as f:
        for img, cap in valid_pairs:
            f.write(f"{img}\t{cap}\n")

    unique_imgs = len(set(img for img, _ in valid_pairs))
    print(f"  Written {after_count} lines to {canonical_cap_file}")

    # ── Done ──────────────────────────────────────────────────────────
    stats = {
        "images_dir": canonical_img_dir,
        "captions_file": canonical_cap_file,
        "num_images": unique_imgs,
        "num_captions": after_count,
        "num_unique_images": unique_imgs,
    }
    _print_success(stats)
    return stats


# ======================================================================
# INTERNAL — already-prepared check
# ======================================================================

def _is_already_prepared(img_dir, cap_file):
    """Return True if canonical format exists and looks healthy."""
    if not os.path.isdir(img_dir):
        return False
    if not os.path.isfile(cap_file):
        return False
    # Quick check: at least 1 image and 1 caption line
    jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    if len(jpgs) == 0:
        return False
    with open(cap_file, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first or "\t" not in first:
        return False
    return True


def _validate_canonical(img_dir, cap_file):
    """Validate an already-prepared canonical dataset."""
    images = set(f for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".jpeg")))
    pairs = []
    with open(cap_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0] in images:
                pairs.append((parts[0], parts[1]))

    if not pairs:
        _fail(
            what="Canonical captions.txt exists but contains no valid pairs",
            why="Image–caption cross-validation failed.",
            fix="Delete data/captions.txt and re-run to trigger re-preparation.",
        )

    unique = len(set(p[0] for p in pairs))
    return {
        "images_dir": img_dir,
        "captions_file": cap_file,
        "num_images": unique,
        "num_captions": len(pairs),
        "num_unique_images": unique,
    }


# ======================================================================
# INTERNAL — find images
# ======================================================================

def _find_all_images(root, exclude_dir=None):
    """Recursively find all .jpg/.jpeg files under *root*."""
    results = []
    exclude = os.path.abspath(exclude_dir) if exclude_dir else None
    for dirpath, _dirnames, filenames in os.walk(root):
        # Skip the canonical output directory to avoid counting already-copied files
        if exclude and os.path.abspath(dirpath).startswith(exclude):
            continue
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg")):
                results.append(os.path.join(dirpath, f))
    return results


# ======================================================================
# INTERNAL — find caption file
# ======================================================================

_CAPTION_CANDIDATES = [
    "Flickr8k.token.txt",
    "Flickr8k_text/Flickr8k.token.txt",
    "captions.csv",
    "captions_csv.txt",
    "results.csv",
]


def _find_caption_file(root, exclude_file=None):
    """Return the path to the most likely caption file."""
    exclude = os.path.abspath(exclude_file) if exclude_file else None

    # 1. Check known names
    for name in _CAPTION_CANDIDATES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            if exclude and os.path.abspath(path) == exclude:
                continue
            return path

    # 2. Scan for any .txt or .csv up to 2 levels deep
    for depth_limit in range(3):
        for dirpath, _dirs, files in os.walk(root):
            rel_depth = dirpath.replace(root, "").count(os.sep)
            if rel_depth > depth_limit:
                continue
            for f in sorted(files):
                full = os.path.join(dirpath, f)
                if exclude and os.path.abspath(full) == exclude:
                    continue
                low = f.lower()
                if low.endswith((".txt", ".csv")):
                    # Heuristic: file likely contains captions if it has many lines
                    try:
                        size = os.path.getsize(full)
                        if size > 1000:  # > 1 KB likely has real data
                            return full
                    except OSError:
                        pass
    return None


# ======================================================================
# INTERNAL — parse ANY caption format
# ======================================================================

def _parse_any_caption_format(filepath):
    """Parse captions from ANY known Flickr8k format.

    Returns list of (image_filename, caption_string) tuples.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Detect format from first few non-empty lines
    sample_lines = [l.strip() for l in lines[:20] if l.strip()]
    if not sample_lines:
        return []

    fmt = _detect_format(sample_lines, filepath)
    print(f"  Detected format: {fmt}")

    if fmt == "token":
        return _parse_token_format(lines)
    elif fmt == "csv":
        return _parse_csv_format(filepath)
    elif fmt == "tsv_plain":
        return _parse_tsv_plain(lines)
    else:
        # Try all parsers, return whichever produces results
        print("  Format unclear — trying all parsers...")
        for name, parser in [("token", lambda: _parse_token_format(lines)),
                              ("csv", lambda: _parse_csv_format(filepath)),
                              ("tsv", lambda: _parse_tsv_plain(lines))]:
            try:
                result = parser()
                if result:
                    print(f"    → '{name}' parser succeeded with {len(result)} pairs")
                    return result
            except Exception:
                pass
        return []


def _detect_format(sample_lines, filepath):
    """Heuristic format detection from sample lines."""
    first = sample_lines[0]

    # Token format: contains '#' and TAB  (e.g.  image.jpg#0\tcaption)
    token_score = sum(1 for l in sample_lines if "#" in l and "\t" in l)
    if token_score >= len(sample_lines) * 0.5:
        return "token"

    # CSV with header: first line has 'image' and 'caption' keywords
    first_low = first.lower()
    if ("image" in first_low and "caption" in first_low) or filepath.lower().endswith(".csv"):
        return "csv"

    # Plain TSV: TAB-separated, first field looks like a filename
    tsv_score = sum(1 for l in sample_lines if "\t" in l and "." in l.split("\t")[0])
    if tsv_score >= len(sample_lines) * 0.5:
        return "tsv_plain"

    return "unknown"


def _parse_token_format(lines):
    """Parse Flickr8k .token.txt: ``image.jpg#N<TAB>caption``"""
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        img_field = parts[0].strip()
        caption = parts[1].strip()
        if not caption:
            continue
        # Strip #0, #1, etc.
        img_name = img_field.split("#")[0].strip()
        if not img_name:
            continue
        # Ensure it looks like a filename
        if "." not in img_name:
            continue
        pairs.append((img_name, caption))
    return pairs


def _parse_csv_format(filepath):
    """Parse CSV (with or without header)."""
    pairs = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # Try to sniff the dialect
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        except csv.Error:
            dialect = "excel"

        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        return []

    # Detect header row
    header = rows[0]
    header_low = [h.strip().lower() for h in header]
    has_header = any(kw in header_low for kw in ("image", "caption", "filename", "text", "comment"))

    if has_header:
        # Find column indices
        img_col = _find_col(header_low, ("image", "image_name", "filename", "image_id", "img"))
        cap_col = _find_col(header_low, ("caption", "captions", "text", "comment"))
        data_rows = rows[1:]
    else:
        img_col = 0
        cap_col = 1 if len(header) > 1 else 0
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(img_col, cap_col):
            continue
        img = row[img_col].strip()
        cap = row[cap_col].strip()
        if not img or not cap:
            continue
        # Strip #N suffix if present
        img = img.split("#")[0].strip()
        if "." not in img:
            continue
        pairs.append((img, cap))

    return pairs


def _parse_tsv_plain(lines):
    """Parse plain TSV: ``image.jpg<TAB>caption``"""
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        img = parts[0].strip().split("#")[0].strip()
        cap = parts[1].strip()
        if img and cap and "." in img:
            pairs.append((img, cap))
    return pairs


def _find_col(header_low, keywords):
    """Find column index matching any keyword."""
    for i, h in enumerate(header_low):
        if h in keywords:
            return i
    return 0


# ======================================================================
# INTERNAL — error & display helpers
# ======================================================================

class DatasetError(RuntimeError):
    """Raised when dataset preparation fails unrecoverably."""
    pass


def _fail(what, why, fix):
    """Raise DatasetError with structured message."""
    msg = (
        f"\n{'='*64}\n"
        f"❌  DATASET ERROR\n"
        f"{'='*64}\n\n"
        f"WHAT:  {what}\n\n"
        f"WHY:   {why}\n\n"
        f"FIX:   {fix}\n"
        f"\n{'='*64}\n"
    )
    raise DatasetError(msg)


def _print_banner(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def _print_success(stats):
    print(f"\n{'='*64}")
    print(f"  ✅ Dataset ready")
    print(f"     Images  : {stats['num_images']}")
    print(f"     Captions: {stats['num_captions']}")
    print(f"     Path    : {stats['images_dir']}")
    print(f"     Captions: {stats['captions_file']}")
    print(f"{'='*64}\n")


# ======================================================================
# CLI — can be run standalone
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Prepare Flickr8k dataset into canonical format")
    parser.add_argument("--data_root", default="data",
                        help="Root directory containing raw Flickr8k data")
    args = parser.parse_args()

    try:
        stats = prepare_dataset(args.data_root)
        print("Done. You can now run:  python train.py")
    except DatasetError as e:
        print(str(e))
        exit(1)
