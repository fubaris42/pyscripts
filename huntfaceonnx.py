#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
from PIL import Image
from rembg import new_session, remove
import numpy as np

session = new_session("u2net_human_seg")


def get_subject_bbox(img):
    result = remove(img, session=session, only_mask=True)
    mask_np = np.array(result.convert("L"))
    binary = mask_np > 10
    coords = np.argwhere(binary)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return (x0, y0, x1, y1)


def process_image(input_path, output_path, pil_img=None):
    try:
        img = (
            pil_img.convert("RGB") if pil_img else Image.open(input_path).convert("RGB")
        )
        bbox = get_subject_bbox(img)
        if bbox:
            cropped = img.crop(bbox)
            cropped.save(output_path, format="JPEG")
            print(f"Cropped: {input_path} → {output_path}")
        else:
            print(f"No subject found in {input_path}, skipped.")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")


def process_pdf_with_fitz(input_path, output_dir, rel_path):
    try:
        doc = fitz.open(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            output_filename = f"{base_name}_page{i}.jpg"
            save_dir = os.path.join(output_dir, rel_path)
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, output_filename)

            process_image(input_path, output_path, pil_img=img)

        doc.close()

    except Exception as e:
        print(f"Failed to process PDF {input_path}: {e}")


def process_images_recursive(input_dir, output_dir):
    supported_image_ext = (".png", ".jpg", ".jpeg")
    supported_pdf_ext = (".pdf",)

    os.makedirs(output_dir, exist_ok=True)
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    for dirpath, _, filenames in os.walk(input_dir):
        dirpath_abs = os.path.abspath(dirpath)
        if os.path.commonpath([dirpath_abs, output_dir_abs]) == output_dir_abs:
            continue

        for filename in filenames:
            input_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(dirpath, input_dir)

            if filename.lower().endswith(supported_image_ext):
                save_dir = os.path.join(output_dir, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                output_filename = os.path.splitext(filename)[0] + ".jpg"
                output_path = os.path.join(save_dir, output_filename)
                process_image(input_path, output_path)

            elif filename.lower().endswith(supported_pdf_ext):
                process_pdf_with_fitz(input_path, output_dir, rel_path)


# Run
input_directory = os.getcwd()
output_directory = os.path.join(input_directory, "cropped_subject_original_bg")
process_images_recursive(input_directory, output_directory)
