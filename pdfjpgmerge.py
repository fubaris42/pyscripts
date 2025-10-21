#!/usr/bin/env python3

import os
from PyPDF2 import PdfMerger
from PIL import Image
from io import BytesIO

"""
say fuck it and merge all kinda stuff
"""

pdf_dir = "."
output_file = "merged.pdf"
merger = PdfMerger()

image_exts = (".png", ".jpg", ".jpeg")

files = sorted(os.listdir(pdf_dir))

for file in files:
    ext = os.path.splitext(file)[1].lower()
    file_path = os.path.join(pdf_dir, file)

    if ext == ".pdf":
        merger.append(file_path)
        print(f"Added PDF: {file}")

    elif ext in image_exts:
        with Image.open(file_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img_pdf = BytesIO()
            img.save(img_pdf, format="PDF")
            img_pdf.seek(0)

            merger.append(img_pdf)
            print(f"Added Image: {file}")

with open(output_file, "wb") as out:
    merger.write(out)
print(f"Merged PDF saved as {output_file}")
merger.close()
