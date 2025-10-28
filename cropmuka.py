#!/usr/bin/env python3

# DeepFace-based script for 3:4 portrait cropping centered on the face.
# Uses RetinaFace backend for robust detection.

from pathlib import Path
from PIL import Image
import numpy as np
from deepface import DeepFace  # New dependency for robust face detection

# --- Configuration ---
SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
# Set the desired detector backend for DeepFace
DEEPFACE_DETECTOR = (
    "retinaface"  # Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
)

# --- PORTRAIT TUNING CONFIGURATION ---
# These constants control the final crop composition. Adjust them for fine-tuning.
PORTRAIT_ASPECT_RATIO = 3 / 4  # The target aspect ratio (3/4 = 0.75 for portrait)
VERTICAL_EXPANSION_FACTOR = (
    1.8  # Controls vertical context: 1.8x face height for overall crop height
)
FACE_HORIZONTAL_PADDING = (
    1.1  # Controls min horizontal padding: 1.1 = 10% total padding around face
)
FACE_VERTICAL_ANCHOR = (
    0.23  # Controls head space: 0.23 means face starts 23% down from the top edge
)
# -------------------------------------

print(f"DeepFace initialized with detector backend: {DEEPFACE_DETECTOR}")


# ----------------------------------------------------------------------
# NEW FUNCTION: Face Detection using DeepFace
# ----------------------------------------------------------------------
def get_face_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Detects the largest face in the image using DeepFace with the RetinaFace backend.
    Returns (x0, y0, x1, y1) in PIL format (left, top, right, bottom).
    """
    # Convert PIL Image to NumPy array (DeepFace handles internal format conversion)
    img_np = np.array(img.convert("RGB"))

    try:
        # DeepFace detection call
        detected_faces = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,  # Set to False to prevent errors if no face is found
            align=False,  # We don't need alignment for simple bounding box extraction
        )

        # Filter out cases where detection returned an empty list or None
        if not detected_faces or not any("facial_area" in d for d in detected_faces):
            return None

        # Select the largest face by area
        largest_face_data = max(
            detected_faces, key=lambda d: d["facial_area"]["w"] * d["facial_area"]["h"]
        )

        area = largest_face_data["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        # Convert DeepFace/OpenCV format (x, y, w, h) to PIL format (x0, y0, x1, y1)
        return (x, y, x + w, y + h)

    except Exception as e:
        # DeepFace can sometimes raise exceptions even with enforce_detection=False
        print(f"DeepFace detection failed on image: {e}")
        return None


# ----------------------------------------------------------------------
# ASPECT RATIO CROP FUNCTION (SIMPLIFIED)
# ----------------------------------------------------------------------
def calculate_3_4_crop(
    face_bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """
    Calculates a 3:4 aspect ratio bounding box based on the face_bbox,
    using global tuning constants for portrait composition.
    """
    x0_face, y0_face, x1_face, y1_face = face_bbox
    w_face = x1_face - x0_face
    h_face = y1_face - y0_face

    # 1. Determine the ideal crop height (H_crop) based on the face and global factor
    H_crop_initial = int(h_face * VERTICAL_EXPANSION_FACTOR)

    # 2. Determine the required crop width (W_crop) based on H_crop and global ratio
    W_crop = int(H_crop_initial * PORTRAIT_ASPECT_RATIO)

    # If the required width is too small to contain the face (based on padding factor), adjust W_crop and H_crop
    if W_crop < w_face * FACE_HORIZONTAL_PADDING:
        W_crop = int(w_face * FACE_HORIZONTAL_PADDING)
        H_crop = int(W_crop / PORTRAIT_ASPECT_RATIO)
    else:
        H_crop = H_crop_initial

    # 3. Determine the final BBox coordinates

    # Calculate desired vertical placement (y0_crop) using the global anchor
    y0_crop = int(y0_face - H_crop * FACE_VERTICAL_ANCHOR)
    y1_crop = y0_crop + H_crop

    # Calculate horizontal placement (x0_crop) - center the face within the crop
    cx_face = (x0_face + x1_face) / 2
    x0_crop = int(cx_face - W_crop / 2)
    x1_crop = x0_crop + W_crop

    # 4. Clip the crop box to the image boundaries

    # Clip horizontally
    dx = 0
    if x0_crop < 0:
        dx = -x0_crop
    elif x1_crop > img_width:
        dx = img_width - x1_crop
    x0_crop += dx
    x1_crop += dx

    # Clip vertically
    dy = 0
    if y0_crop < 0:
        dy = -y0_crop
    elif y1_crop > img_height:
        dy = img_height - y1_crop
    y0_crop += dy
    y1_crop += dy

    # Ensure coordinates are within bounds after shifting
    x0_final = max(0, int(x0_crop))
    y0_final = max(0, int(y0_crop))
    x1_final = min(img_width, int(x1_crop))
    y1_final = min(img_height, int(y1_crop))

    return (x0_final, y0_final, x1_final, y1_final)


# ----------------------------------------------------------------------
# MAIN PROCESSING FUNCTION (Logic retained)
# ----------------------------------------------------------------------
def crop_face_portrait_and_save(input_path: Path, output_path: Path):
    """
    Opens an image, finds the face bounding box, calculates a 3:4 crop
    centered on the face, and saves it.
    """
    try:
        # 1. Open and convert to RGB
        img = Image.open(input_path).convert("RGB")
        img_width, img_height = img.size

        # 2. Get Face BBox
        face_bbox = get_face_bbox(img)

        if face_bbox:
            # 3. Calculate 3:4 Aspect Ratio BBox anchored by the face
            # Note: No tuning params passed here, they are read from globals
            crop_bbox = calculate_3_4_crop(face_bbox, img_width, img_height)

            # 4. Crop using PIL
            cropped = img.crop(crop_bbox)

            # 5. Determine save format
            output_extension = input_path.suffix.upper()[1:]
            output_format = (
                "JPEG" if output_extension in ("JPG", "JPEG") else output_extension
            )

            # 6. Save the cropped image
            cropped.save(output_path, format=output_format)

            print(
                f"Cropped (3:4, Face-anchored) and saved: {input_path.relative_to(Path.cwd())} → {output_path.name}"
            )
        else:
            print(f"No face found in {input_path.name}, skipped.")

    except Exception as e:
        print(f"🛑 Failed to process {input_path.name}: {e}")


def process_directory(input_dir: Path, output_dir: Path):
    """
    Recursively scans the input directory, processes supported images,
    and saves them to the output directory while maintaining the relative structure.
    """
    # DeepFace models are often loaded lazily/automatically upon first call

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {input_dir}")
    print(f"Output directory:   {output_dir}")
    print("-" * 50)

    # Use rglob to find all files recursively
    for source_path in input_dir.rglob("*"):

        # Skip anything inside the output folder
        if source_path.is_relative_to(output_dir):
            continue

        if source_path.is_dir():
            continue

        if source_path.suffix.lower() in SUPPORTED_EXT:
            # Preserve directory structure
            rel_path = source_path.parent.relative_to(input_dir)
            target_dir = output_dir / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            target_file_path = target_dir / source_path.name

            # Process the image
            crop_face_portrait_and_save(source_path, target_file_path)


# --- Run ---
if __name__ == "__main__":
    # Define directories relative to the current working directory
    input_directory = Path.cwd()
    output_directory = input_directory / "deepface_3_4_portrait_crops"

    process_directory(input_directory, output_directory)

    print("-" * 50)
    print("✨ Processing complete.")
