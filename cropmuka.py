#!/usr/bin/env python3

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from rembg import new_session, remove  # Ensure rembg is installed: pip install rembg

# --- Configuration ---
# Supported image extensions (case-insensitive check)
SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}

# Initialize rembg session once
try:
    # Use "u2net" for general objects or "u2net_human_seg" for people
    REMBG_SESSION = new_session("u2net_human_seg")
except Exception as e:
    print(f"Error initializing rembg session: {e}", file=sys.stderr)
    print(
        "Please ensure rembg is installed correctly (pip install rembg) and models are available."
    )
    sys.exit(1)


def get_subject_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Finds the bounding box of the main subject in the image using rembg's mask.
    Returns (x0, y0, x1, y1) or None if no subject is found.
    """
    # 1. Generate mask
    result = remove(img, session=REMBG_SESSION, only_mask=True)

    # 2. Convert mask to numpy array
    mask_np = np.array(result.convert("L"))

    # 3. Find coordinates where the mask is active (subject area)
    binary = mask_np > 10  # Threshold for the mask to be considered valid
    coords = np.argwhere(binary)

    if coords.size == 0:
        return None  # No subject found

    # 4. Determine BBox (min/max coordinates)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # PIL crop expects (left, top, right, bottom) which is (x0, y0, x1, y1)
    return (x0, y0, x1, y1)


def crop_subject_and_save(input_path: Path, output_path: Path):
    """
    Opens an image, finds the subject bounding box, crops the image,
    and saves it to the target path, preserving the original extension.
    """
    try:
        # 1. Open and convert to RGB for consistent processing
        img = Image.open(input_path).convert("RGB")

        # 2. Get BBox
        bbox = get_subject_bbox(img)

        if bbox:
            # 3. Crop
            cropped = img.crop(bbox)

            # 4. Determine save format based on original extension
            output_extension = input_path.suffix.upper()[1:]  # e.g., PNG, JPEG

            # Use 'JPEG' format for both .JPG and .JPEG extensions
            output_format = (
                "JPEG" if output_extension in ("JPG", "JPEG") else output_extension
            )

            # 5. Save the cropped image
            # Note: We save using the output_path which retains the original extension
            cropped.save(output_path, format=output_format)

            print(
                f"Cropped and saved: {input_path.relative_to(Path.cwd())} → {output_path.name}"
            )
        else:
            print(f"No subject found in {input_path.name}, skipped.")

    except Exception as e:
        print(f"🛑 Failed to process {input_path.name}: {e}")


def process_directory(input_dir: Path, output_dir: Path):
    """
    Recursively scans the input directory, processes supported images,
    and saves them to the output directory while maintaining the relative structure.
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {input_dir}")
    print(f"Output directory:   {output_dir}")
    print("-" * 50)

    # Use rglob to find all files recursively
    for source_path in input_dir.rglob("*"):

        # --- FIX: Skip anything (files or directories) that is inside the output folder ---
        if source_path.is_relative_to(output_dir):
            continue

        # 1. Skip directories (we only want to process files)
        if source_path.is_dir():
            continue

        # 2. Check if the file is a supported image type
        if source_path.suffix.lower() in SUPPORTED_EXT:

            # 3. Preserve directory structure
            # Get the path relative to the input_dir
            rel_path = source_path.parent.relative_to(input_dir)

            # Create the corresponding target directory path
            target_dir = output_dir / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            # The target file path keeps the original filename and extension
            target_file_path = target_dir / source_path.name

            # 4. Process the image
            crop_subject_and_save(source_path, target_file_path)


# --- Run ---
if __name__ == "__main__":
    # Define directories relative to the current working directory
    input_directory = Path.cwd()
    output_directory = input_directory / "cropped_subject_original_bg"

    process_directory(input_directory, output_directory)

    print("-" * 50)
    print("✨ Processing complete.")
