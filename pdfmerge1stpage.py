#!/usr/bin/env python3

import os
from PyPDF2 import PdfMerger

pdf_dir = "."
output_file = "merged.pdf"
merger = PdfMerger()

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
pdf_files.sort()

for pdf in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf)
    merger.append(pdf_path, pages=(0, 1))  # Only first page
    print(f"Added first page of {pdf}")

with open(output_file, "wb") as out:
    merger.write(out)

print(f"Merged PDF saved as {output_file}")
merger.close()
