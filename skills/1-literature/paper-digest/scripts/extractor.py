"""
PDF extraction with multiple backends:
- Fast mode: PyMuPDF with multi-strategy table detection (good for simple tables)
- Accurate mode: IBM Docling with TableFormer AI (better for complex/borderless tables)
"""

import os
import sys
from pathlib import Path

# Suppress PyMuPDF's "Consider using pymupdf_layout" recommendation
# This prints to stdout and pollutes --stdout output
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

# Version for cache invalidation - increment when extraction logic changes
# Format: major.minor.patch
# 3.1.0: Page separators now use <!-- PAGE_BREAK --> instead of -----
#        Image extraction includes nested XObjects (full=True)
# 3.2.0: Fast mode now includes image references in markdown (write_images=True)
#        Cache keys now include no_images flag to avoid contamination
# 3.3.0: Image paths in cached markdown now use relative 'images/' prefix
#        (fixes broken temp directory references in cached output)
EXTRACTOR_VERSION = "3.3.0"


def check_docling_models():
    """Check if Docling models are downloaded."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        # Check for docling models in HF cache
        docling_repos = [r for r in cache_info.repos if "docling" in r.repo_id.lower()]
        return len(docling_repos) > 0
    except Exception:
        return False


def extract_pdf_fast(
    pdf_path: str, image_dir: str = None, show_progress: bool = False
) -> str:
    """
    Fast PDF extraction using PyMuPDF with text-based table detection.

    Uses 'text' table strategy which handles borderless/whitespace-based
    tables better than the default 'lines_strict' for mixed document types.

    Args:
        pdf_path: Path to the PDF file
        image_dir: Directory to save extracted images (None = skip images)
        show_progress: Whether to show progress output

    Returns:
        Markdown string of the PDF content with image references if image_dir provided
    """
    import pymupdf4llm

    if show_progress:
        print("Extracting with PyMuPDF (fast mode)...", file=sys.stderr)

    # Use text strategy which handles borderless tables better
    # than the default lines_strict
    markdown = pymupdf4llm.to_markdown(
        pdf_path,
        show_progress=show_progress,
        table_strategy="text",  # Better for mixed table types
        write_images=image_dir is not None,
        image_path=str(image_dir) if image_dir else None,
    )

    # Replace pymupdf4llm's default page separator with explicit sentinel.
    # This prevents false splits when documents contain literal "-----"
    # (horizontal rules, ASCII tables, etc.)
    markdown = markdown.replace("\n-----\n", "\n<!-- PAGE_BREAK -->\n")

    return markdown


def _save_docling_images(result, output_dir: Path) -> list:
    """
    Save images from a Docling conversion result to output directory.

    Images are saved in iteration order, which matches the order of
    <!-- image --> placeholders in the exported markdown.

    Args:
        result: Docling ConversionResult object
        output_dir: Directory to save images to

    Returns:
        List of saved image paths (in iteration order)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for i, (element, _level) in enumerate(result.document.iterate_items()):
        if hasattr(element, "image") and element.image is not None:
            img_path = output_dir / f"figure_{i:04d}.png"
            element.image.pil_image.save(str(img_path))
            image_paths.append(str(img_path))

    return image_paths


def extract_pdf_docling(
    pdf_path: str,
    output_dir: str = None,
    images_scale: float = 4.0,
    show_progress: bool = False,
) -> tuple:
    """
    Extract PDF using Docling with accurate tables + high-res images.

    Uses IBM's TableFormer AI model for ~93.6% table extraction accuracy.
    Also extracts images at configurable resolution (default 4x for crisp images).

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images (None = skip images)
        images_scale: Image resolution multiplier (default: 4.0 for high-res)
        show_progress: Whether to show progress output

    Returns:
        tuple: (markdown: str, image_paths: list[str])
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling_core.types.doc.base import ImageRefMode

    # Check if this is first run (models need downloading)
    if not check_docling_models():
        print(
            "First run: downloading Docling AI models (one-time setup, ~2-3 minutes)...",
            file=sys.stderr,
        )

    if show_progress:
        print(
            f"Processing PDF with Docling (accurate mode, ~1 sec/page)...",
            file=sys.stderr,
        )

    # Configure pipeline for accurate tables + image extraction
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=output_dir is not None,
        images_scale=images_scale,
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    result = converter.convert(pdf_path)

    # Check for conversion errors
    if hasattr(result, "errors") and result.errors:
        for error in result.errors:
            print(f"WARNING: Docling conversion error: {error}", file=sys.stderr)

    # Check conversion status
    from docling.datamodel.base_models import ConversionStatus

    if hasattr(result, "status") and result.status != ConversionStatus.SUCCESS:
        print(
            f"WARNING: Docling conversion status: {result.status.name}",
            file=sys.stderr,
        )

    # Save images to output directory (order matters for placeholder replacement)
    image_paths = []
    if output_dir:
        image_paths = _save_docling_images(result, Path(output_dir))
        if show_progress and image_paths:
            print(
                f"Extracted {len(image_paths)} images at {images_scale}x resolution",
                file=sys.stderr,
            )

    # Export markdown with placeholders
    md = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

    # Replace placeholders with actual image references (order must match iteration order)
    for img_path in image_paths:
        md = md.replace("<!-- image -->", f"![Figure](images/{Path(img_path).name})", 1)

    return md, image_paths


def extract_pdf_to_markdown(
    pdf_path: str, accurate: bool = False, show_progress: bool = False
) -> str:
    """
    Extract PDF to markdown with configurable accuracy/speed trade-off.

    Args:
        pdf_path: Path to the PDF file
        accurate: If True, use Docling AI (better for complex tables, slower).
                  If False, use PyMuPDF (fast, good for simple tables).
        show_progress: Whether to show progress output

    Returns:
        Markdown string of the PDF content
    """
    if accurate:
        # Use Docling without image extraction
        md, _ = extract_pdf_docling(
            pdf_path, output_dir=None, show_progress=show_progress
        )
        return md
    else:
        return extract_pdf_fast(pdf_path, show_progress)


def get_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF using pymupdf (faster than Docling for this)."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def extract_images(pdf_path: str, output_dir: str, show_progress: bool = False) -> list:
    """
    Extract images from PDF to output directory.

    Uses pymupdf for image extraction since Docling focuses on document structure.
    Deduplicates by xref to avoid extracting the same image multiple times
    (e.g., icons/logos reused across pages).

    Returns:
        List of extracted image paths
    """
    import pymupdf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    extracted = []
    image_count = 0
    seen_xrefs = set()  # Track already-extracted images by xref

    for page_num in range(len(doc)):
        page = doc[page_num]
        # full=True includes images nested inside form XObjects (common in
        # documents exported from Word/PowerPoint)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            try:
                xref = img[0]

                # Skip if we've already extracted this image
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                pix = pymupdf.Pixmap(doc, xref)

                # Convert CMYK to RGB if necessary
                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                image_count += 1
                img_filename = f"image_{image_count:04d}.png"
                img_path = output_path / img_filename
                pix.save(str(img_path))
                extracted.append(str(img_path))

                pix = None
            except Exception as e:
                # Log instead of silently swallowing errors
                print(
                    f"WARNING: Failed to extract image {img_index} on page {page_num + 1}: {e}",
                    file=sys.stderr,
                )
                continue

    doc.close()

    if show_progress and extracted:
        print(f"Extracted {len(extracted)} unique images", file=sys.stderr)

    return extracted
