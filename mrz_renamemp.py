#!/usr/bin/env python3

# MODIFICATION: Refined file naming logic:
# 1. Handles missing MRZ data by renaming the file with an "UNIDENTIFIABLE_MRZ_" prefix.
# 2. Improved duplicate handling to ensure clean incrementing (_1, _2, etc.).
# 3. Keeps the original file suffix (e.g., .jpeg).
# 4. Applies gamma correction (0.7) before binarization for better text contrast.
# 5. NEW: Moved tuning constants to a dedicated configuration block for maintenance.

import re
import warnings
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from fastmrz import FastMRZ
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import math
import os

warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION ---
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Preprocessing Tuning Parameters (Adjust these values for better MRZ detection)
GAMMA_CORRECTION_VALUE = 0.7  # Lower value (e.g., 0.5) darkens text more.
BINARIZATION_THRESHOLD = (
    157  # Pixels darker than this become black (0). Default is often 120-150.
)
UNIDENTIFIABLE_MRZ_PREFIX = "UNIDENTIFIABLE_MRZ"  # Fallback name prefix
# ----------------------------


def preprocess_image(path: Path) -> Path:
    """
    Preprocesses the image for better MRZ detection (Grayscale, Gamma Correction,
    Autocontrast, Binarization).
    """
    image = Image.open(path).convert("L")

    # --- Gamma Correction (Uses global constant) ---
    gamma = GAMMA_CORRECTION_VALUE
    # Build a gamma lookup table (256 entries)
    gamma_table = [int((i / 255.0) ** gamma * 255) for i in range(256)]
    image = image.point(gamma_table)
    # -----------------------------------------------

    image = ImageOps.autocontrast(image)
    # Binarization: Convert pixels below the threshold to black (0), others to white (255)
    image = image.point(lambda x: 0 if x < BINARIZATION_THRESHOLD else 255).convert("RGB")  # type: ignore[reportOperatorIssue]

    tmp_path = path.with_name(f"tmp_{path.name}")
    image.save(tmp_path)
    return tmp_path


def rename_file_with_mrz(file_path: Path):
    """
    Worker function to preprocess an image, extract MRZ, rename the file,
    and clean up the temporary image. Handles naming fallback for unidentifiable MRZ.
    """
    fast_mrz = FastMRZ()
    tmp_img = None

    try:
        tmp_img = preprocess_image(file_path)
        mrz_raw = fast_mrz.get_details(str(tmp_img), ignore_parse=True)
        tmp_img.unlink(missing_ok=True)

        base_name = ""

        if not isinstance(mrz_raw, str) or not mrz_raw.strip():
            # Handle unidentifiable/empty MRZ using global constant
            base_name = UNIDENTIFIABLE_MRZ_PREFIX
        else:
            # Sanitize the MRZ string for use as a filename
            name = re.sub(
                r"[^A-Z0-9_]", "", mrz_raw.replace("\n", "_").replace("<", "_").upper()
            )
            base_name = re.sub(r"_+", "_", name).strip("_")

        # Robust Duplicate Handling

        # 1. Set the initial proposed new file path
        new_name_stem = base_name
        new_name = f"{new_name_stem}{file_path.suffix.lower()}"
        new_path = file_path.with_name(new_name)

        # 2. Handle collisions
        counter = 1
        while new_path.exists():
            new_name_stem = f"{base_name}_{counter}"
            new_name = f"{new_name_stem}{file_path.suffix.lower()}"
            new_path = file_path.with_name(new_name)
            counter += 1

        file_path.rename(new_path)

        if base_name == UNIDENTIFIABLE_MRZ_PREFIX:
            return f"[?] Unidentifiable MRZ, renamed: {file_path.name} → {new_name}"
        else:
            return f"[✓] {file_path.name} → {new_name}"

    except Exception as e:
        if tmp_img:
            tmp_img.unlink(missing_ok=True)

        return f"[!] Error: {file_path.name} → {type(e).__name__}: {e}"


def main():
    """Main function to find image files and parallelize the MRZ processing."""

    # Set TESSDATA_PREFIX to the user's home directory. Tesseract will look in $HOME/tessdata
    tessdata_path = os.path.expanduser("~/.tessdata")
    os.environ["TESSDATA_PREFIX"] = tessdata_path

    files = [
        f
        for f in Path.cwd().iterdir()
        if f.suffix.lower() in SUPPORTED_EXT and f.is_file()
    ]
    if not files:
        print("No supported image files found.")
        return

    # THREADING MODIFICATION: Limit threads to 80% of CPU count
    total_cpus = cpu_count()
    max_workers = max(1, math.ceil(total_cpus * 0.80))

    print(f"Total CPUs available: {total_cpus}")
    print(f"Using {max_workers} threads (80% limit) to process {len(files)} files.")

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for msg in pool.map(rename_file_with_mrz, files):
            print(msg)


if __name__ == "__main__":
    main()
