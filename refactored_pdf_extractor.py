#!/usr/bin/env python3
"""
Refactored Production-Ready PDF Page Extractor
Extracts pages from PDF files, prepending matched text to the output filename.
Handles filename sanitization and duplicates.
"""

import argparse
import logging
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF is not installed. Please run: pip install PyMuPDF")
    sys.exit(1)

# --- Constants ---
DEFAULT_OUTPUT_DIR = "extracted_pages"
DEFAULT_LOG_FILE = "pdf_extractor.log"
MAX_WORKERS_RATIO = 0.8  # Use up to 80% of available CPU cores
MAX_FILENAME_TEXT_LEN = 50  # Max length of the matched text in the filename

# --- Data Classes for Configuration and Results ---


@dataclass(frozen=True)
class ExtractionConfig:
    """Immutable configuration for the PDF extraction process."""

    search_patterns: List[re.Pattern]
    output_dir: Path
    input_dirs: List[Path]
    file_patterns: List[str]
    max_workers: int
    log_level: str
    recursive: bool
    dry_run: bool


@dataclass
class ProcessResult:
    """Holds the result of processing a single PDF file."""

    pdf_path: Path
    status: str  # 'success', 'no_matches', 'failed'
    pages_extracted: int = 0
    matches_found: int = 0
    total_pages: int = 0
    error: str | None = None
    processing_time: float = 0.0


@dataclass
class ExtractionStats:
    """Mutable statistics for the entire extraction process."""

    total_pdfs: int = 0
    processed_pdfs: int = 0
    failed_pdfs: int = 0
    total_pages_extracted: int = 0
    total_matches: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


# --- Helper Functions ---


def sanitize_filename(text: str) -> str:
    """
    Sanitizes a string to be a valid filename component.
    - Replaces illegal characters with underscores.
    - Truncates to a maximum length.
    - Strips leading/trailing whitespace and underscores.
    """
    # Replace illegal characters with an underscore
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", text)
    # Replace newline/tab characters with a space
    sanitized = re.sub(r"[\n\r\t]", " ", sanitized)
    # Truncate to avoid overly long filenames
    if len(sanitized) > MAX_FILENAME_TEXT_LEN:
        sanitized = sanitized[:MAX_FILENAME_TEXT_LEN].strip()
    # Remove leading/trailing whitespace and underscores that might result from stripping
    return sanitized.strip(" _.").strip()


# --- Worker Functions (for Multiprocessing) ---


def process_pdf_worker(pdf_path: Path, config: ExtractionConfig) -> ProcessResult:
    """
    Worker process to find matches and extract pages from a single PDF.
    This function is designed to be run in a separate process.
    """
    start_time = time.time()
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            return ProcessResult(
                pdf_path=pdf_path, status="failed", error="PDF has no pages"
            )

        # Find pages and the specific text that matched
        matching_pages_with_text = find_matches_in_doc(doc, config.search_patterns)
        total_matches = sum(
            len(matches) for matches in matching_pages_with_text.values()
        )

        if not matching_pages_with_text:
            doc.close()
            return ProcessResult(
                pdf_path=pdf_path, status="no_matches", total_pages=total_pages
            )

        pages_extracted = 0
        if not config.dry_run:
            pages_extracted = extract_pages_from_doc(
                doc, pdf_path, matching_pages_with_text, config
            )

        doc.close()

        return ProcessResult(
            pdf_path=pdf_path,
            status="success",
            pages_extracted=(
                pages_extracted if not config.dry_run else len(matching_pages_with_text)
            ),
            matches_found=total_matches,
            total_pages=total_pages,
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        return ProcessResult(
            pdf_path=pdf_path,
            status="failed",
            error=f"An unexpected error occurred: {e}",
            processing_time=time.time() - start_time,
        )


def find_matches_in_doc(
    doc: fitz.Document, patterns: List[re.Pattern]
) -> Dict[int, List[str]]:
    """Scans a document and returns pages with the text of each match."""
    matching_pages = {}
    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_matches = set()  # Use a set to store unique matches per page
        for pattern in patterns:
            for match in pattern.finditer(text):
                page_matches.add(match.group(0))
        if page_matches:
            matching_pages[page_num] = sorted(list(page_matches))
    return matching_pages


def extract_pages_from_doc(
    doc: fitz.Document,
    pdf_path: Path,
    matching_pages: Dict[int, List[str]],
    config: ExtractionConfig,
) -> int:
    """Saves specified pages to new PDF files with unique, sanitized names."""
    extraction_count = 0
    for page_num, matches in matching_pages.items():
        # Determine output path while preserving subdirectory structure
        try:
            base_dir = next(d for d in config.input_dirs if pdf_path.is_relative_to(d))
            relative_path = pdf_path.relative_to(base_dir)
        except StopIteration:
            relative_path = pdf_path.name

        output_dir = config.output_dir / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the first match on the page for the filename
        first_match_text = matches[0]
        sanitized_match = sanitize_filename(first_match_text)

        # Handle potential filename collisions
        # --- FILENAME CHANGE IS HERE ---
        base_name = f"{sanitized_match}_{pdf_path.stem}_page_{page_num + 1}"
        output_path = output_dir / f"{base_name}.pdf"
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{base_name}_({counter}).pdf"
            counter += 1

        # Create new document with the single page
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        new_doc.save(output_path)
        new_doc.close()
        extraction_count += 1
    return extraction_count


# --- Main Application Class ---


class PDFExtractor:
    """Orchestrates the PDF extraction process."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.stats = ExtractionStats()
        self.logger = self._setup_logging()

    def run(self):
        start_time = time.time()
        self._log_initial_summary()

        pdf_files = self._find_pdf_files()
        if not pdf_files:
            self.logger.warning("No PDF files found to process.")
            return self.stats

        self.stats.total_pdfs = len(pdf_files)
        if not self.config.dry_run:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(process_pdf_worker, pdf, self.config)
                for pdf in pdf_files
            ]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    self._update_stats_and_log(result)
                    if i % 10 == 0 or i == self.stats.total_pdfs:
                        progress = (i / self.stats.total_pdfs) * 100
                        self.logger.info(
                            f"📈 Progress: {i}/{self.stats.total_pdfs} ({progress:.1f}%) complete."
                        )
                except Exception as e:
                    self.logger.error(f"A worker process failed unexpectedly: {e}")
                    self.stats.failed_pdfs += 1

        self.stats.processing_time = time.time() - start_time
        self._log_final_summary()
        return self.stats

    def _update_stats_and_log(self, result: ProcessResult):
        if result.status == "failed":
            self.stats.failed_pdfs += 1
            self.stats.errors.append(f"{result.pdf_path.name}: {result.error}")
            self.logger.error(
                f"❌ Failed to process {result.pdf_path.name}: {result.error}"
            )
        else:
            self.stats.processed_pdfs += 1
            if result.status == "success":
                self.stats.total_pages_extracted += result.pages_extracted
                self.stats.total_matches += result.matches_found
                self.logger.info(
                    f"✅ Extracted {result.pages_extracted} pages from {result.pdf_path.name} "
                    f"({result.matches_found} matches found)."
                )
            elif result.status == "no_matches":
                self.logger.info(f"📄 No matches found in: {result.pdf_path.name}")

    def _find_pdf_files(self) -> List[Path]:
        pdf_files = set()
        self.logger.info("🔍 Searching for PDF files...")
        for directory in self.config.input_dirs:
            if not directory.is_dir():
                self.logger.warning(f"Input directory not found: {directory}")
                continue
            for pattern in self.config.file_patterns:
                search_method = (
                    directory.rglob if self.config.recursive else directory.glob
                )
                for pdf_path in search_method(pattern):
                    if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
                        pdf_files.add(pdf_path)
        self.logger.info(f"Found {len(pdf_files)} unique PDF files to process.")
        return sorted(list(pdf_files))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("PDFExtractor")
        logger.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self.config.log_level.upper()))
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # File handler
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(
                self.config.output_dir / DEFAULT_LOG_FILE, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not create log file: {e}. Logging to console only.")
        return logger

    def _log_initial_summary(self):
        self.logger.info("=" * 60)
        self.logger.info("🚀 Starting PDF Page Extractor v3")
        self.logger.info(f"Output directory: {self.config.output_dir.resolve()}")
        if self.config.dry_run:
            self.logger.warning("DRY RUN is enabled. No files will be written.")
        self.logger.info("=" * 60)

    def _log_final_summary(self):
        self.logger.info("=" * 60)
        self.logger.info("🏁 Extraction Complete")
        self.logger.info(f"⏱️  Total time: {self.stats.processing_time:.2f}s")
        self.logger.info(
            f"📁 PDFs processed: {self.stats.processed_pdfs}/{self.stats.total_pdfs}"
        )
        self.logger.info(f"📄 Pages extracted: {self.stats.total_pages_extracted}")
        if self.stats.failed_pdfs > 0:
            self.logger.warning(f"⚠️ Failed to process {self.stats.failed_pdfs} PDF(s).")
        self.logger.info("=" * 60)


# --- CLI Parsing and Main Execution ---


def load_search_patterns(
    search_input: str, use_regex: bool, case_sensitive: bool, whole_word: bool
) -> List[re.Pattern]:
    patterns = []
    input_path = Path(search_input)
    if input_path.is_file():
        try:
            patterns = [
                line.strip()
                for line in input_path.read_text("utf-8").splitlines()
                if line.strip()
            ]
            if not patterns:
                raise ValueError("Search pattern file is empty.")
        except Exception as e:
            raise ValueError(f"Error reading search file '{input_path}': {e}")
    else:
        patterns = [search_input]

    compiled = []
    flags = 0 if case_sensitive else re.IGNORECASE
    for p_str in patterns:
        try:
            regex_str = (
                p_str
                if use_regex
                else (
                    r"\b" + re.escape(p_str) + r"\b" if whole_word else re.escape(p_str)
                )
            )
            compiled.append(re.compile(regex_str, flags))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{p_str}': {e}")
    return compiled


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracts pages from PDF files, prepending matched text to the filename.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Search for the word "Invoice" and save pages with sanitized filenames
  python %(prog)s "Invoice" -w

  # Use a regex to find amounts like "$1,234.56" and search recursively
  python %(prog)s "\\$\\d{1,3}(,\\d{3})*\\.\\d{2}" --regex -R
""",
    )
    parser.add_argument(
        "search_input", help="Search string or path to a file with search patterns."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-i",
        "--input-dirs",
        nargs="+",
        default=["."],
        help="Input directories (default: current directory).",
    )
    parser.add_argument(
        "-f",
        "--file-patterns",
        nargs="+",
        default=["*.pdf"],
        help="File glob patterns (default: *.pdf).",
    )
    parser.add_argument(
        "-r",
        "--regex",
        action="store_true",
        help="Treat patterns as regular expressions.",
    )
    parser.add_argument(
        "-c",
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive matching.",
    )
    parser.add_argument(
        "-w",
        "--whole-word",
        action="store_true",
        help="Match whole words only (ignored if --regex).",
    )
    parser.add_argument(
        "-R", "--recursive", action="store_true", help="Search directories recursively."
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Simulate extraction without writing files.",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        help="Max processes (default: 80%% of CPU cores).",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level.",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        max_workers = args.max_workers or max(
            1, int((os.cpu_count() or 1) * MAX_WORKERS_RATIO)
        )
        patterns = load_search_patterns(
            args.search_input, args.regex, args.case_sensitive, args.whole_word
        )

        config = ExtractionConfig(
            search_patterns=patterns,
            output_dir=Path(args.output_dir),
            input_dirs=[Path(d) for d in args.input_dirs],
            file_patterns=args.file_patterns,
            max_workers=max_workers,
            log_level=args.log_level,
            recursive=args.recursive,
            dry_run=args.dry_run,
        )

        stats = PDFExtractor(config).run()

        print("\n--- 📊 Final Summary ---")
        print(f"Total Time:      {stats.processing_time:.2f}s")
        print(f"PDFs Processed:  {stats.processed_pdfs}/{stats.total_pdfs}")
        print(f"Pages Extracted: {stats.total_pages_extracted}")
        if stats.failed_pdfs > 0:
            print(f"PDFs Failed:     {stats.failed_pdfs}")
            return 1
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n🚫 Process interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"💥 An unexpected error occurred: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
