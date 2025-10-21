#!/usr/bin/env python3
"""
Production-Ready PDF Page Extractor
Extracts pages from PDF files based on search patterns with multiprocessing and regex support.
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)

# Constants
DEFAULT_OUTPUT_DIR = "extracted_pages"
DEFAULT_LOG_FILE = "pdf_extractor.log"
MAX_WORKERS_RATIO = 0.8  # Use 80% of available CPU cores
CHUNK_SIZE = 10  # Pages to process in each thread chunk


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction process."""

    search_patterns: List[str]
    output_dir: str
    log_level: str
    max_workers: int
    use_regex: bool
    case_sensitive: bool
    whole_words_only: bool
    input_dirs: List[str]
    file_patterns: List[str]
    recursive: bool
    dry_run: bool
    stats_file: Optional[str]


@dataclass
class ExtractionStats:
    """Statistics for extraction process."""

    total_pdfs: int = 0
    processed_pdfs: int = 0
    failed_pdfs: int = 0
    total_pages_extracted: int = 0
    total_matches: int = 0
    processing_time: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def process_single_pdf(
    args: Tuple[str, List[re.Pattern], ExtractionConfig],
) -> Dict[str, Any]:
    """Process a single PDF file - standalone function for multiprocessing."""
    pdf_path_str, compiled_patterns, config = args
    pdf_path = Path(pdf_path_str)

    result = {
        "pdf_path": str(pdf_path),
        "pages_extracted": 0,
        "matches_found": 0,
        "errors": [],
        "extracted_files": [],
        "processing_start_time": time.time(),
        "file_size_mb": 0,
        "total_pages": 0,
        "matching_pages": [],
        "patterns_matched": [],
    }

    # Set up logging for this process
    logger = logging.getLogger(f"pdf_extractor_worker_{os.getpid()}")
    handler_exists = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    if not handler_exists:
        logger.setLevel(logging.INFO)  # Show extraction logs
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False  # Prevent duplicate logs

    try:
        # Get file size
        result["file_size_mb"] = round(pdf_path.stat().st_size / (1024 * 1024), 2)

        logger.info(f"🔍 Processing: {pdf_path.name} ({result['file_size_mb']} MB)")

        # Open PDF document
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        result["total_pages"] = total_pages

        if total_pages == 0:
            result["errors"].append("PDF has no pages")
            logger.warning(f"⚠️  Empty PDF: {pdf_path.name}")
            return result

        logger.info(f"📖 Scanning {total_pages} pages in: {pdf_path.name}")

        # Track patterns that matched
        matched_patterns = set()
        matching_pages = {}

        # Extract text and find matches
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                text = page.get_text()

                # Search for patterns
                page_matches = []
                page_pattern_matches = {}

                for pattern in compiled_patterns:
                    pattern_matches = list(pattern.finditer(text))
                    if pattern_matches:
                        page_matches.extend(pattern_matches)
                        matched_patterns.add(pattern.pattern)
                        page_pattern_matches[pattern.pattern] = len(pattern_matches)

                if page_matches:
                    matching_pages[page_num] = page_matches
                    result["matching_pages"].append(
                        {
                            "page_number": page_num + 1,
                            "matches_count": len(page_matches),
                            "patterns": page_pattern_matches,
                        }
                    )
                    result["matches_found"] += len(page_matches)

                    logger.info(
                        f"   ✅ Page {page_num + 1}: {len(page_matches)} matches found"
                    )
                    for pattern, count in page_pattern_matches.items():
                        logger.debug(f"      - Pattern '{pattern}': {count} matches")

            except Exception as e:
                error_msg = f"Error processing page {page_num + 1}: {e}"
                result["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
                continue

        result["patterns_matched"] = list(matched_patterns)

        if not matching_pages:
            logger.info(f"📄 No matches found in: {pdf_path.name}")
            return result

        logger.info(
            f"🎯 Found {len(matching_pages)} matching pages in: {pdf_path.name}"
        )

        # Extract matching pages
        if not config.dry_run:
            base_name = pdf_path.stem

            # Create subdirectory structure if original was in subdirectory
            input_base = (
                Path(config.input_dirs[0])
                if len(config.input_dirs) == 1
                else pdf_path.parent
            )
            try:
                relative_path = pdf_path.relative_to(input_base)
                output_subdir = Path(config.output_dir) / relative_path.parent
            except ValueError:
                # If relative path calculation fails, just use the output directory
                output_subdir = Path(config.output_dir)

            try:
                output_subdir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                error_msg = f"Cannot create output subdirectory {output_subdir}: {e}"
                result["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return result

            extraction_count = 0
            for page_num in matching_pages.keys():
                output_filename = f"{base_name}_page_{page_num + 1}.pdf"
                output_path = output_subdir / output_filename

                if output_path.exists():
                    logger.debug(
                        f"⏭️  File already exists: {output_path.relative_to(Path(config.output_dir))}"
                    )
                    result["extracted_files"].append(str(output_path))
                    continue

                try:
                    # Create new document with single page
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    new_doc.save(str(output_path))
                    new_doc.close()

                    result["extracted_files"].append(str(output_path))
                    result["pages_extracted"] += 1
                    extraction_count += 1

                    relative_output = output_path.relative_to(Path(config.output_dir))
                    logger.info(f"💾 Extracted page {page_num + 1} → {relative_output}")

                except Exception as e:
                    error_msg = f"Failed to extract page {page_num + 1}: {e}"
                    result["errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")

            if extraction_count > 0:
                logger.info(
                    f"✅ Successfully extracted {extraction_count} pages from: {pdf_path.name}"
                )

        else:
            # Dry run - just count what would be extracted
            result["pages_extracted"] = len(matching_pages)
            logger.info(
                f"🧪 [DRY RUN] Would extract {len(matching_pages)} pages from: {pdf_path.name}"
            )

            for page_num in matching_pages.keys():
                logger.info(f"   📋 Would extract page {page_num + 1}")

        doc.close()

        # Log processing completion
        processing_time = time.time() - result["processing_start_time"]
        logger.info(f"⏱️  Completed {pdf_path.name} in {processing_time:.2f}s")

    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {e}"
        result["errors"].append(error_msg)
        logger.error(f"❌ Fatal error in {pdf_path.name}: {e}")

    return result


class PDFExtractor:
    """Production-ready PDF page extractor with multiprocessing and regex support."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.stats = ExtractionStats()
        self.logger = self._setup_logging()
        self.compiled_patterns = self._compile_patterns()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("pdf_extractor")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create output directory for log file
        output_dir = Path(self.config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # File handler - only create if we can write to the directory
            log_file = output_dir / DEFAULT_LOG_FILE
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except (PermissionError, OSError) as e:
            # If we can't create log file, just use console logging
            print(f"Warning: Could not create log file in {output_dir}: {e}")
            print("Continuing with console logging only...")

        # Console handler (always add this)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile search patterns into regex objects."""
        compiled = []
        flags = 0 if self.config.case_sensitive else re.IGNORECASE

        for pattern in self.config.search_patterns:
            try:
                if self.config.use_regex:
                    # Use pattern as-is for regex
                    regex_pattern = pattern
                else:
                    # Escape special characters for literal search
                    regex_pattern = re.escape(pattern)

                    # Add word boundaries if whole words only
                    if self.config.whole_words_only:
                        regex_pattern = r"\b" + regex_pattern + r"\b"

                compiled.append(re.compile(regex_pattern, flags))
                self.logger.debug(f"Compiled pattern: {regex_pattern}")

            except re.error as e:
                self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        return compiled

    def _find_pdf_files(self) -> List[Path]:
        """Find all PDF files in specified directories (recursively)."""
        pdf_files = []

        for input_dir in self.config.input_dirs:
            dir_path = Path(input_dir)
            if not dir_path.exists():
                self.logger.warning(f"Directory not found: {input_dir}")
                continue

            if not dir_path.is_dir():
                self.logger.warning(f"Path is not a directory: {input_dir}")
                continue

            self.logger.info(f"Scanning directory recursively: {dir_path.absolute()}")

            # Find PDFs matching file patterns recursively
            for file_pattern in self.config.file_patterns:
                try:
                    # Use rglob for recursive search
                    if self.config.recursive:
                        matches = list(dir_path.rglob(file_pattern))
                        search_type = "recursive"
                    else:
                        matches = list(dir_path.glob(file_pattern))
                        search_type = "non-recursive"

                    pdf_matches = [
                        f for f in matches if f.suffix.lower() == ".pdf" and f.is_file()
                    ]
                    pdf_files.extend(pdf_matches)

                    self.logger.info(
                        f"Found {len(pdf_matches)} PDFs with pattern '{file_pattern}' ({search_type}) in {input_dir}"
                    )

                    # Log each found PDF file
                    for pdf_file in pdf_matches:
                        relative_path = (
                            pdf_file.relative_to(dir_path)
                            if pdf_file.is_relative_to(dir_path)
                            else pdf_file
                        )
                        self.logger.debug(f"  📄 Found PDF: {relative_path}")

                except Exception as e:
                    self.logger.error(
                        f"Error searching for pattern '{file_pattern}' in {input_dir}: {e}"
                    )

        # Remove duplicates while preserving order
        seen = set()
        unique_pdfs = []
        for pdf in pdf_files:
            if pdf not in seen:
                seen.add(pdf)
                unique_pdfs.append(pdf)

        # Log directory structure summary
        if unique_pdfs:
            self.logger.info(f"📊 Discovery Summary:")
            self.logger.info(f"   Total unique PDFs found: {len(unique_pdfs)}")

            # Group by directory for summary
            dir_counts = {}
            for pdf in unique_pdfs:
                parent_dir = str(pdf.parent)
                dir_counts[parent_dir] = dir_counts.get(parent_dir, 0) + 1

            self.logger.info(f"   PDFs by directory:")
            for directory, count in sorted(dir_counts.items()):
                self.logger.info(f"     {directory}: {count} files")

        return unique_pdfs

    def extract_pages(self) -> ExtractionStats:
        """Main extraction process using multiprocessing."""
        start_time = time.time()

        # Create output directory
        output_path = Path(self.config.output_dir)
        if not self.config.dry_run:
            output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting PDF extraction process")
        self.logger.info(f"Search patterns: {self.config.search_patterns}")
        self.logger.info(f"Use regex: {self.config.use_regex}")
        self.logger.info(f"Case sensitive: {self.config.case_sensitive}")
        self.logger.info(f"Whole words only: {self.config.whole_words_only}")
        self.logger.info(f"Recursive search: {self.config.recursive}")
        self.logger.info(f"Input directories: {self.config.input_dirs}")
        self.logger.info(f"File patterns: {self.config.file_patterns}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Max workers: {self.config.max_workers}")
        self.logger.info(f"Dry run: {self.config.dry_run}")
        self.logger.info("=" * 80)

        # Find PDF files
        pdf_files = self._find_pdf_files()
        if not pdf_files:
            self.logger.warning("No PDF files found")
            return self.stats

        self.stats.total_pdfs = len(pdf_files)

        # Prepare arguments for multiprocessing
        process_args = [
            (str(pdf_path), self.compiled_patterns, self.config)
            for pdf_path in pdf_files
        ]

        # Process PDFs using multiprocessing
        results = []
        processed_count = 0
        failed_count = 0

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_pdf = {
                executor.submit(process_single_pdf, args): Path(args[0])
                for args in process_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["errors"]:
                        failed_count += 1
                        self.stats.errors.extend(result["errors"])
                        self.logger.error(f"❌ Failed: {pdf_path.name}")
                        for error in result["errors"]:
                            self.logger.error(f"   💥 {error}")
                    else:
                        processed_count += 1
                        if result["pages_extracted"] > 0:
                            self.logger.info(
                                f"🎉 Successfully processed: {pdf_path.name}"
                            )
                            self.logger.info(
                                f"   📄 {result.get('total_pages', 'N/A')} total pages"
                            )
                            self.logger.info(
                                f"   ✅ {result['pages_extracted']} pages extracted"
                            )
                            self.logger.info(
                                f"   🎯 {result['matches_found']} total matches"
                            )
                            self.logger.info(
                                f"   📊 {result.get('file_size_mb', 'N/A')} MB file size"
                            )

                            # Log matched patterns
                            if result.get("patterns_matched"):
                                self.logger.info(
                                    f"   🔍 Patterns matched: {', '.join(result['patterns_matched'])}"
                                )
                        else:
                            self.logger.info(f"📭 No matches found in: {pdf_path.name}")

                    self.stats.total_pages_extracted += result["pages_extracted"]
                    self.stats.total_matches += result["matches_found"]

                    # Progress update with more details
                    completed = processed_count + failed_count
                    if completed % 5 == 0 or completed == len(pdf_files):
                        progress_pct = (completed / len(pdf_files)) * 100
                        self.logger.info(
                            f"📈 Progress: {completed}/{len(pdf_files)} ({progress_pct:.1f}%) - "
                            f"Extracted: {self.stats.total_pages_extracted} pages, "
                            f"Matches: {self.stats.total_matches}"
                        )

                except Exception as e:
                    failed_count += 1
                    error_msg = f"Process failed for {pdf_path}: {e}"
                    self.stats.errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")

        # Update final statistics
        self.stats.processed_pdfs = processed_count
        self.stats.failed_pdfs = failed_count
        self.stats.processing_time = time.time() - start_time

        # Save detailed results if requested
        if self.config.stats_file:
            self._save_detailed_stats(results)

        # Log final statistics
        self.logger.info("=" * 80)
        self.logger.info(f"🏁 EXTRACTION COMPLETE")
        self.logger.info(
            f"⏱️  Total processing time: {self.stats.processing_time:.2f} seconds"
        )
        self.logger.info(
            f"📁 PDFs processed successfully: {self.stats.processed_pdfs}/{self.stats.total_pdfs}"
        )
        self.logger.info(
            f"📄 Total pages extracted: {self.stats.total_pages_extracted}"
        )
        self.logger.info(f"🎯 Total matches found: {self.stats.total_matches}")

        if self.stats.failed_pdfs > 0:
            self.logger.warning(f"⚠️  Failed PDFs: {self.stats.failed_pdfs}")
            self.logger.warning(f"💥 Error summary:")
            for i, error in enumerate(
                self.stats.errors[:10], 1
            ):  # Show first 10 errors
                self.logger.warning(f"   {i}. {error}")
            if len(self.stats.errors) > 10:
                self.logger.warning(
                    f"   ... and {len(self.stats.errors) - 10} more errors"
                )

        if self.config.dry_run:
            self.logger.info(f"🧪 DRY RUN completed - no files were actually created")
        else:
            self.logger.info(f"💾 Output directory: {self.config.output_dir}")

        self.logger.info("=" * 80)

        return self.stats

    def _save_detailed_stats(self, results: List[Dict[str, Any]]):
        """Save detailed extraction statistics to JSON file."""
        stats_data = {
            "summary": self.stats.__dict__,
            "config": {
                "search_patterns": self.config.search_patterns,
                "use_regex": self.config.use_regex,
                "case_sensitive": self.config.case_sensitive,
                "whole_words_only": self.config.whole_words_only,
                "output_dir": self.config.output_dir,
                "input_dirs": self.config.input_dirs,
                "file_patterns": self.config.file_patterns,
                "dry_run": self.config.dry_run,
            },
            "detailed_results": results,
        }

        stats_path = Path(self.config.output_dir) / self.config.stats_file
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_data, f, indent=2, default=str)
            self.logger.info(f"Detailed statistics saved to: {stats_path}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")


def load_search_strings(input_arg: str) -> List[str]:
    """Load search strings from file or command line argument."""
    input_path = Path(input_arg)

    if input_path.is_file():
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    raise ValueError("Search file is empty")
                return lines
        except Exception as e:
            raise ValueError(f"Error reading search file '{input_path}': {e}")
    else:
        # Treat as direct search string
        return [input_arg.strip()]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production-ready PDF page extractor with multiprocessing and regex support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "search term"                    # Search for literal text
  %(prog)s patterns.txt                     # Load patterns from file
  %(prog)s "error|warning" --regex          # Use regex patterns
  %(prog)s "\\btest\\b" --regex --case-sensitive  # Case-sensitive whole word search
  %(prog)s "data" --input-dirs ./docs ./reports --dry-run  # Test run on multiple directories
        """,
    )

    parser.add_argument(
        "search_input",
        help="Search string or path to file containing search patterns (one per line)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for extracted pages (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--input-dirs",
        "-i",
        nargs="+",
        default=["."],
        help="Input directories to search for PDFs (default: current directory)",
    )

    parser.add_argument(
        "--file-patterns",
        "-f",
        nargs="+",
        default=["*.pdf"],
        help="File patterns to match (default: *.pdf)",
    )

    parser.add_argument(
        "--regex",
        "-r",
        action="store_true",
        help="Treat search patterns as regular expressions",
    )

    parser.add_argument(
        "--case-sensitive",
        "-c",
        action="store_true",
        help="Enable case-sensitive matching",
    )

    parser.add_argument(
        "--whole-words",
        "-w",
        action="store_true",
        help="Match whole words only (adds word boundaries)",
    )

    parser.add_argument(
        "--max-workers",
        "-j",
        type=int,
        help="Maximum number of worker processes (default: 80%% of CPU cores)",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search directories recursively for PDF files",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be extracted without actually creating files",
    )

    parser.add_argument(
        "--stats-file", "-s", help="Save detailed statistics to JSON file"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()

        # Load search patterns
        search_patterns = load_search_strings(args.search_input)

        # Calculate max workers
        max_workers = args.max_workers
        if max_workers is None:
            import multiprocessing

            max_workers = max(1, int(multiprocessing.cpu_count() * MAX_WORKERS_RATIO))

        # Create configuration
        config = ExtractionConfig(
            search_patterns=search_patterns,
            output_dir=args.output_dir,
            log_level=args.log_level,
            max_workers=max_workers,
            use_regex=args.regex,
            case_sensitive=args.case_sensitive,
            whole_words_only=args.whole_words,
            input_dirs=args.input_dirs,
            file_patterns=args.file_patterns,
            recursive=args.recursive,
            dry_run=args.dry_run,
            stats_file=args.stats_file,
        )

        # Create and run extractor
        extractor = PDFExtractor(config)
        stats = extractor.extract_pages()

        # Print summary
        print(f"\n📊 Extraction Summary:")
        print(f"   PDFs processed: {stats.processed_pdfs}/{stats.total_pdfs}")
        print(f"   Pages extracted: {stats.total_pages_extracted}")
        print(f"   Matches found: {stats.total_matches}")
        print(f"   Processing time: {stats.processing_time:.2f}s")

        if stats.failed_pdfs > 0:
            print(f"   ⚠️  Failed PDFs: {stats.failed_pdfs}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
