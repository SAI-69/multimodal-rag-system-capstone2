import base64
import io
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# ---------------------------------------------------------------------------
# Docling label taxonomy (DocItemLabel enum values we care about):
#
#   section_header  — numbered or unnumbered section headings
#   title           — document-level title
#   text / paragraph— body paragraphs
#   list_item       — bullet / numbered list items
#   caption         — figure / table captions (emitted as separate nodes)
#   footnote        — footnotes at the bottom of a page
#   table           — tabular data (Docling reconstructs cell structure)
#   picture         — embedded raster / vector images
#   chart           — chart/graph images (rendered image, no raw data)
#   page_header     — running header printed on every page  ← NOISE, skipped
#   page_footer     — running footer printed on every page  ← NOISE, skipped
# ---------------------------------------------------------------------------

import hashlib
import re

# Module-level cache: skips VLM calls for identical images across pages/runs
_IMAGE_CAPTION_CACHE: dict[str, str] = {}

def normalize_vlm_output(content: str, page_no: int | None = None) -> str:
    """Clean VLM output while preserving natural language for embeddings."""
    if not content:
        return f"Image on page {page_no or 'unknown'} with no extractable content"
    
    content = content.strip()
    
    # Remove conversational filler
    content = re.sub(
        r'^(This image shows|This image displays|The image contains|Visible here|Here is)\s*[:.]?\s*', 
        '', content, flags=re.I
    )
    
    # Strip markdown but keep punctuation that aids semantics
    content = re.sub(r'[\*\_\#\`]', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Ensure it ends with a period for sentence-level embedding coherence
    if content and not content.endswith(('.', '!', '?')):
        content += '.'
    
    return content

def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of typed content chunks using Docling.

    Each chunk is a dict with three keys:
      content      — text or markdown representation of the element
      content_type — one of: "text", "table", "image"
      metadata     — dict with: content_type, element_type, section,
                     page_number, source_file, image_base64

    The metadata is passed through unchanged to PGVector, so every
    retrieved chunk tells the query layer what kind of content it is
    and where in the document it came from.
    """

    # ── Step 1: Configure Docling pipeline ───────────────────────────────────
    # do_ocr=True          — run OCR on scanned/rasterised pages so text is
    #                        extractable even when not embedded in the PDF
    # do_table_structure   — detect table grid lines and reconstruct rows/cols
    # generate_picture_images — render each picture element to a PIL Image so
    #                           we can base64-encode it for storage
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    # ── Step 2: Convert the PDF ───────────────────────────────────────────────
    # converter.convert() runs the full Docling pipeline:
    #   layout analysis → OCR → table structure → picture rendering
    # result.document is a DoclingDocument with a typed element tree.
    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    # Tracks the most recently seen section heading so every chunk carries
    # the section name it belongs to — useful for filtered retrieval.
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    # ── Step 3: Walk the document element tree ────────────────────────────────
    # iterate_items() yields (level, node) tuples in Docling >= 2.x.
    # level is the heading depth (1 = top-level); node is the DocItem.
    for item in doc.iterate_items():
        if isinstance(item, tuple):
            node, _ = item  # unpack yields (level, node); discard level
        else:
            node = item     # older Docling versions yield bare nodes

        # label is a DocItemLabel enum value — convert to lowercase string
        # for pattern matching (e.g. "section_header", "table", "picture")
        label = str(getattr(node, "label", "")).lower()

        # ── Skip page headers/footers ─────────────────────────────────────────
        # These repeat on every page (document title, page number, date stamp)
        # and would pollute retrieval results with irrelevant noise.
        if label in ("page_header", "page_footer"):
            continue

        # ── Extract page number and bounding box from provenance ──────────────
        # prov is a list of ProvenanceItem; prov[0] covers the first (usually
        # only) occurrence of the element. bbox gives the element's position
        # on the page as left/top/right/bottom coordinates (0–1 normalised or
        # absolute, depending on Docling version).
        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _make_metadata(content_type: str, element_type: str, img_b64=None):
            """Build a metadata dict that is stored alongside every chunk.

            content_type  — "text" | "table" | "image"  (used by the query
                            layer to decide how to render retrieved content)
            element_type  — raw Docling label ("section_header", "table", …)
            img_b64       — base64-encoded PNG string for image elements;
                            None for text and table elements
            """
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,       # bounding box stored in JSONB position column
                "image_base64": img_b64,    # decoded to BYTEA by db.store_chunks()
            }

        # ── Section headings & document title ─────────────────────────────────
        # Update current_section so all subsequent chunks carry the correct
        # section name until the next heading is encountered.
        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

        # ── Tables ────────────────────────────────────────────────────────────
        # Convert table cells to clean "Header: value" plain text rows so
        # that no markdown pipe/dash symbols pollute the vector store.
        # Strategy:
        #   1. export_to_dataframe() — preferred; yields a pandas DataFrame
        #      with header row and typed cell values from Docling's grid.
        #   2. Fallback: export_to_html() stripped of tags, then plain text.
        # Each table row is serialised as "Col1: val1 | Col2: val2" so the
        # column context travels with every value and embeddings are meaningful.

        
        elif "table" in label:
            table_text = ""
            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe()
                    if df is not None and not df.empty:
                        rows_text: list[str] = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows_text.append("  |  ".join(pairs))
                        table_text = "\n".join(rows_text)
                except Exception:
                    pass

            # Fallback: strip HTML tags from export_to_html()
            if not table_text and hasattr(node, "export_to_html"):
                try:
                    import re as _re
                    raw_html = node.export_to_html(doc)
                    table_text = _re.sub(r"<[^>]+>", " ", raw_html or "")
                    table_text = _re.sub(r"\s+", " ", table_text).strip()
                except Exception:
                    pass

            # Last resort: raw text attribute
            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text and table_text.strip():
                parsed_chunks.append(
                    {
                        "content": table_text.strip(),
                        "content_type": "table",
                        "metadata": _make_metadata("table", "table"),
                    }
                )

        # ── Pictures, figures, and charts ─────────────────────────────────────
        # Charts are rendered images in Docling (no structured data is
        # extracted), so they are handled identically to pictures.
        # Extraction strategy:
        #   1. get_image(doc) — preferred; uses pre-rendered PIL Images
        #      produced when generate_picture_images=True
        #   2. .image.pil_image — fallback attribute on some Docling versions
        # The PIL Image is encoded as a base64 PNG and stored in metadata so
        # the Gemini Vision LLM can receive it directly during generation.
                # ── Pictures, figures, and charts ─────────────────────────────────────
        elif "picture" in label or "figure" in label or label == "chart":
            img_b64 = None
            caption = getattr(node, "text", "") or ""

            try:
                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                if img_b64 is None and hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass

            content = caption.strip()

            # Only call VLM if caption is empty but we have an image
            if not content and img_b64 is not None:
                from langchain_core.messages import HumanMessage  # Lazy import
                img_hash = hashlib.sha256(base64.b64decode(img_b64.strip())).hexdigest()[:16]

                if img_hash in _IMAGE_CAPTION_CACHE:
                    content = _IMAGE_CAPTION_CACHE[img_hash]
                else:
                    vision_prompt = (
                        "Describe this image for document search. Output a single, concise sentence. "
                        "Prioritize: exact names > numbers/dates > business context > visual details. "
                        "Rules: "
                        "• NO conversational openers ('This image shows', 'visible here') "
                        "• NO markdown, bullets, or special characters "
                        "• Logos: '[Company] is a [industry] company specializing in [key areas].' "
                        "• Charts: '[Metric] [trend/value] for [entity] during [timeframe]. [Key insight].' "
                        "• Keep under 100 words. Plain text only."
                    )

                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    )

                    raw_content = ""

                    # ── 1️⃣ Try Gemini Primary ─────────────────────────────────────
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        gemini_model = ChatGoogleGenerativeAI(
                            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.1-pro-preivew"),
                            temperature=0,
                            candidate_count=1,
                            timeout=30,
                            max_retries=1
                        )
                        response = gemini_model.invoke([message])
                        raw_content = getattr(response, "content", "")
                    except Exception as e:
                        print(f"[docling_parser] Gemini vision failed: {e}. Falling back to OpenAI...")

                    # ── 2️⃣ Fallback to OpenAI ──────────────────────────────────────
                    # if not raw_content:
                    #     try:
                    #         from langchain_openai import ChatOpenAI
                    #         openai_model = ChatOpenAI(
                    #             model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
                    #             temperature=0,
                    #             timeout=30,
                    #             max_retries=1
                    #         )
                    #         response = openai_model.invoke([message])
                    #         raw_content = getattr(response, "content", "")
                    #     except Exception as e:
                    #         print(f"[docling_parser] OpenAI vision fallback also failed: {e}")

                    # ── Normalize & Cache ──────────────────────────────────────────
                    if isinstance(raw_content, list):
                        raw_content = raw_content[0].get("text", "") if raw_content else ""

                    content = normalize_vlm_output(str(raw_content), page_no)
                    
                    # Only cache successful, meaningful outputs
                    if content and "no extractable content" not in content.lower():
                        _IMAGE_CAPTION_CACHE[img_hash] = content

                # ── Final Safety Fallback ──────────────────────────────────────
                if not content or len(content.strip()) < 10:
                    content = f"Image on page {page_no or 'unknown'} with no extractable content."

            # Guarantee clean string before DB insertion
            content = str(content).strip()
            if not content.endswith('.'):
                content += '.'

            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": _make_metadata("image", "picture", img_b64),
                }
            )            

        # ── Plain text: paragraphs, list items, captions, footnotes, etc. ─────
        # Everything that is not a heading, table, or image falls here.
        # Empty nodes (layout artefacts with no text) are silently dropped.
        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append(
                    {
                        "content": text.strip(),
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

    return parsed_chunks
