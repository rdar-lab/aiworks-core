"""PowerPoint rendering from structured JSON spec."""

import base64
import io
import logging
import re
import zipfile
from copy import deepcopy
from io import BytesIO
from typing import Any, Literal

import cairosvg
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Inches, Pt

logger = logging.getLogger(__name__)

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _svg_to_png_bytes(svg_text: str, width_px: int = 300) -> bytes | None:
    """Rasterise an SVG string to PNG bytes using cairosvg."""
    try:
        import re
        svg_text = re.sub(r'width="([^"]+)in"', lambda m: f'width="{m.group(1)}"', svg_text)
        svg_text = re.sub(r'height="([^"]+)in"', lambda m: f'height="{m.group(1)}"', svg_text)
        return cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=width_px,
        )
    except Exception as exc:
        logger.warning("_svg_to_png_bytes | SVG→PNG conversion failed: %s", exc)
        return None


def _load_ppt_schema(schema_file_path: str) -> str:
    try:
        with open(schema_file_path) as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Was unable to locate file: {schema_file_path}")


def _fill_slide_background(slide, color):
    try:
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color
    except Exception as e:
        logger.warning("_fill_slide_background | failed: %s", e)


def _add_footer_bar(slide, width_inches, height_inches, secondary_color):
    try:
        color = _resolve_color_value(secondary_color, "#e94560")
        bar = slide.shapes.add_shape(
            1,
            Inches(0), Inches(height_inches - 0.5),
            Inches(width_inches), Inches(0.5)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = RGBColor.from_string(color.lstrip("#"))
        bar.line.fill.background()
    except Exception as e:
        logger.warning("_add_footer_bar | failed: %s", e)


def _resolve_color_value(color_spec, default_hex="#333333") -> str:
    """Resolve a color spec to a hex string."""
    if not color_spec:
        return default_hex
    if isinstance(color_spec, str):
        return color_spec if color_spec.startswith("#") else default_hex
    if isinstance(color_spec, dict):
        return color_spec.get("hex", default_hex)
    return default_hex


def _resolve_font_color(font_color_spec: str | None, colors, default_hex="#333333"):
    if not font_color_spec:
        font_color_spec = "text"

    if font_color_spec.startswith("#"):
        resolved = RGBColor.from_string(font_color_spec.lstrip("#"))
    else:
        slot = colors.get(font_color_spec)
        if isinstance(slot, str):
            resolved = RGBColor.from_string(slot.lstrip("#"))
        elif slot:
            resolved = RGBColor.from_string(slot.get("hex", default_hex).lstrip("#"))
        else:
            resolved = RGBColor.from_string(default_hex.lstrip("#"))

    return resolved


def _add_elements_to_slide(slide, elements, colors, images, is_bg_run, table_config=None):
    if is_bg_run:
        elements = [elem for elem in elements if elem.get("type") == 'rect']
    else:
        elements = [elem for elem in elements if elem.get("type") != 'rect']

    for elem in elements:
        elem_type = elem.get("type")
        position = elem.get("position", {})
        font_color_spec = elem.get("font_color")
        align = elem.get("align", "left")

        cx, cy, x, y = _calc_elm_position(position)

        if elem_type == "text":
            _handle_text(
                align, colors, cx, cy, elem, font_color_spec, slide, x, y
            )
        elif elem_type == "rect":
            _handle_rect(colors, cx, cy, elem, slide, x, y)
        elif elem_type == "image":
            _handle_image(x, y, cx, cy, elem, slide, images, colors)
        elif elem_type == "table":
            _handle_table(colors, cx, elem, slide, x, y, table_config)


def _add_custom_shape_image(x: int, y: int, cx: int, cy: int, elem, slide, images, colors=None):
    shape_data_str = elem.get("shape_data")
    try:
        spPr_elem = etree.fromstring(shape_data_str)

        solidFill_elem = spPr_elem.find(f"{{{A_NS}}}solidFill")
        if solidFill_elem is not None:
            srgbClr = solidFill_elem.find(f"{{{A_NS}}}srgbClr")
            if srgbClr is None:
                schemeClr = solidFill_elem.find(f"{{{A_NS}}}schemeClr")
                if schemeClr is not None and colors:
                    color_name = schemeClr.get("val")
                    color_val = colors.get(color_name, "")
                    if isinstance(color_val, dict):
                        color_val = color_val.get("hex", "") or color_val.get("rgb", "")
                    if color_val:
                        hex_val = color_val.lstrip("#")
                        srgbClr = etree.SubElement(solidFill_elem, f"{{{A_NS}}}srgbClr")
                        srgbClr.set("val", hex_val)
                        solidFill_elem.remove(schemeClr)

        cNvPr_list = (
                list(slide.shapes._spTree.iter(f"{{{P_NS}}}sp")) +
                list(slide.shapes._spTree.iter(f"{{{P_NS}}}pic"))
        )
        existing_ids = []
        for e in cNvPr_list:
            cNvPr = e.find(f"{{{P_NS}}}nvSpPr")
            if cNvPr is not None:
                cNvPr = cNvPr.find(f"{{{P_NS}}}cNvPr")
            if cNvPr is not None:
                existing_ids.append(int(cNvPr.get("id", 0)))
        next_id = max(existing_ids + [1]) + 1
        sp_xml = f"<p:sp xmlns:p=\"{P_NS}\" xmlns:a=\"{A_NS}\"><p:nvSpPr><p:cNvPr id=\"{next_id}\" name=\"Shape\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr></p:sp>"
        sp_elem = parse_xml(sp_xml)
        spPr_new = deepcopy(spPr_elem)
        sp_elem.append(spPr_new)
        xfrm = spPr_new.find(f"{{{A_NS}}}xfrm")
        if xfrm is None:
            xfrm = etree.SubElement(spPr_new, f"{{{A_NS}}}xfrm")
        off = xfrm.find(f"{{{A_NS}}}off")
        if off is None:
            off = etree.SubElement(xfrm, f"{{{A_NS}}}off")
        ext = xfrm.find(f"{{{A_NS}}}ext")
        if ext is None:
            ext = etree.SubElement(xfrm, f"{{{A_NS}}}ext")
        off.set("x", str(int(x)))
        off.set("y", str(int(y)))
        ext.set("cx", str(int(cx)))
        ext.set("cy", str(int(cy)))

        sp = slide.shapes._spTree
        sp.append(sp_elem)
        return
    except Exception as e:
        logger.warning("_handle_image | shape_data native path failed: %s", e)


def _handle_image(x: int, y: int, cx: int, cy: int, elem, slide, images, colors=None):
    shape_data = elem.get("shape_data")
    image_data = elem.get("image_data")
    svg_data = elem.get("svg_data")
    image_file_name = elem.get("image_file_name")

    if shape_data:
        _add_custom_shape_image(x,y, cx, cy, elem, slide, images, colors)
    elif image_data:
        png_bytes = base64.b64decode(image_data)
        img_stream = io.BytesIO(png_bytes)
        slide.shapes.add_picture(img_stream, x, y, cx, cy)
    elif svg_data:
        if elem.get("fill_color_key") and colors:
            fill_key = elem["fill_color_key"]
            fill_hex = colors.get(fill_key, "#888888")
            if isinstance(fill_hex, dict):
                fill_hex = fill_hex.get("hex", "#888888")
            svg_data = re.sub(r'fill="[^"]*"', f'fill="{fill_hex}"', svg_data, count=1)

        width_px = int(cx / 9525)
        png_bytes = _svg_to_png_bytes(svg_data, width_px=width_px)
        if png_bytes:
            try:
                img_stream = io.BytesIO(png_bytes)
                slide.shapes.add_picture(img_stream, x, y, cx, cy)
            except Exception as e:
                logger.warning("_add_elements_to_slide | svg embed failed: %s", e)
    elif image_file_name:
        img_data = images.get(image_file_name)
        if img_data:
            try:
                img_stream = io.BytesIO(img_data)
                slide.shapes.add_picture(img_stream, x, y, cx, cy)
            except Exception as e:
                logger.warning("_add_elements_to_slide | image embed failed: %s", e)
        else:
            logger.warning("_add_elements_to_slide | image data not found: %s", image_file_name)


def _calc_elm_position(position: dict) -> tuple[int, int, int, int]:
    x = Inches(position.get("x", 0))
    y = Inches(position.get("y", 0))
    cx = Inches(position.get("width", 1))
    cy = Inches(position.get("height", 1))
    return cx, cy, x, y


def _handle_text(align, colors, cx: int, cy: int, elem, font_color_spec, slide, x: int, y: int):
    resolved_font = _get_font_with_overrides("body", {}, elem, bold_default=False)
    text = elem.get("text", "")

    tb = slide.shapes.add_textbox(x, y, cx, cy)
    _handle_alignment(align, tb)

    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(resolved_font["size"])
    p.font.bold = resolved_font["bold"]
    p.font.italic = resolved_font["italic"]
    p.font.name = resolved_font["family"]
    p.font.color.rgb = _resolve_font_color(font_color_spec, colors)


def _handle_alignment(align, tb):
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.auto_size = MSO_AUTO_SIZE.NONE
    if align == "center":
        p.alignment = PP_ALIGN.CENTER
        tf.alignment = PP_ALIGN.CENTER
        tb.alignment = PP_ALIGN.CENTER
    elif align == "right":
        p.alignment = PP_ALIGN.RIGHT
        tf.alignment = PP_ALIGN.RIGHT
        tb.alignment = PP_ALIGN.RIGHT
    else:
        p.alignment = PP_ALIGN.LEFT
        tf.alignment = PP_ALIGN.LEFT
        tb.alignment = PP_ALIGN.LEFT

    p.width = tb.width
    p.height = tb.height


def _handle_rect(colors, cx: int, cy: int, elem, slide, x: int, y: int):
    fill_spec = elem.get("fill", "")
    stroke_spec = elem.get("stroke")

    shape = slide.shapes.add_shape(
        1, x, y, cx, cy
    )

    if fill_spec:
        if fill_spec.startswith("#"):
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(fill_spec.lstrip("#"))
        else:
            shape.fill.solid()
            shape.fill.fore_color.rgb = _resolve_color(colors, fill_spec, "#cccccc")
    else:
        shape.fill.background()

    if stroke_spec:
        if stroke_spec.startswith("#"):
            shape.line.color.rgb = RGBColor.from_string(stroke_spec.lstrip("#"))
        else:
            shape.line.color.rgb = _resolve_color(colors, stroke_spec, "#000000")
        shape.line.width = Pt(elem.get("width", 0))
    else:
        shape.line.fill.background()


def _handle_table(colors, cx: int, elem, slide, x: int, y: int, table_config=None):
    columns = elem.get("columns", [])
    rows = elem.get("rows", [])
    table_config = table_config or {}

    num_cols = len(columns)
    num_rows = len(rows) + 1

    tbl = slide.shapes.add_table(num_rows, num_cols, x, y, cx, Inches(num_rows * 0.3)).table

    header_fill = _resolve_color(colors, table_config.get("header_fill"), "#4472c4")
    header_font_color = _resolve_color(colors, table_config.get("header_font_color", "#ffffff"))
    cell_fill_primary = _resolve_color(colors, table_config.get("cell_fill_primary"), "#a6a6a6")
    cell_fill_alt = _resolve_color(colors, table_config.get("cell_fill_alt"), "#e7e6e6")
    cell_font_color = _resolve_color(colors, table_config.get("cell_font_color", "#000000"))
    table_font_spec = table_config.get("font", {})
    table_font = _resolve_font("body", table_font_spec, False)
    table_font_size = table_font.get("font_size", 16)
    table_font_family = table_font.get("family", "Arial")

    for ci, col in enumerate(columns):
        cell = tbl.cell(0, ci)
        cell.text = col.get("header", "")
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(table_font_size)
        p.font.name = table_font_family
        p.font.color.rgb = header_font_color
        try:
            tc = cell._tc
            tcPr = tc.find(qn("a:tcPr"))
            if tcPr is None:
                tcPr = etree.SubElement(tc, qn("a:tcPr"))
            solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
            srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
            srgbClr.set("val", str(header_fill))
        except Exception:
            pass

    for ri, row in enumerate(rows):
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = cell_text
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(table_font_size)
            p.font.bold = False
            p.font.italic = False
            p.font.name = table_font_family
            if cell_font_color:
                p.font.color.rgb = cell_font_color
            effective_fill = cell_fill_primary
            if cell_fill_alt and ri % 2 == 1:
                effective_fill = cell_fill_alt

            if effective_fill:
                try:
                    tc = cell._tc
                    tcPr = tc.find(qn("a:tcPr"))
                    if tcPr is None:
                        tcPr = etree.SubElement(tc, qn("a:tcPr"))
                    solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
                    srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
                    srgbClr.set("val", str(effective_fill))
                except Exception:
                    pass


def render_presentation_pptx(presentation_json: dict, images: dict[str, bytes]) -> bytes:
    """Render a PowerPoint presentation from a JSON spec.

    Args:
        presentation_json: Parsed presentation spec dict with "template" and "slides".
        images: Dict mapping filename (e.g. "chart.png") to raw bytes.

    Returns:
        Raw PPTX bytes.
    """
    prs = _build_presentation(presentation_json, images)

    buf = io.BytesIO()
    prs.save(buf)
    pptx_bytes = buf.getvalue()
    return _fix_duplicate_shape_ids(pptx_bytes)


def _resolve_color(colors, color_spec, default_hex: str | None = "#333333"):
    if not color_spec:
        if not default_hex:
            return None
        return RGBColor.from_string(default_hex.lstrip("#"))
    if isinstance(color_spec, str):
        if color_spec.startswith("#"):
            return RGBColor.from_string(color_spec.lstrip("#"))
        if color_spec.startswith("scheme:"):
            color_spec = color_spec[7:]
        slot = colors.get(color_spec)
        if slot:
            if isinstance(slot, str):
                return RGBColor.from_string(slot.lstrip("#"))
            if isinstance(slot, dict):
                hex_val = slot.get("hex", default_hex)
                return RGBColor.from_string(hex_val.lstrip("#")) if hex_val else (RGBColor.from_string(default_hex.lstrip("#")) if default_hex else None)
            return RGBColor.from_string(slot.get("hex", default_hex).lstrip("#"))
        return RGBColor.from_string(default_hex.lstrip("#")) if default_hex else None
    if isinstance(color_spec, dict):
        return RGBColor.from_string(color_spec.get("hex", default_hex).lstrip("#"))
    return RGBColor.from_string(default_hex.lstrip("#")) if default_hex else None


def _get_font(font_spec, default="Arial"):
    return font_spec if font_spec else default


SAFE_FONTS = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Georgia"]


def _safe_font(family: str) -> str:
    return family if family in SAFE_FONTS else "Arial"


def _resolve_font(font_key: str, fonts: dict, bold_default: bool = False) -> dict:
    """Resolve font definition from template fonts dict.

    Args:
        font_key: Key in fonts dict, e.g. "heading" or "body"
        fonts: The template fonts dict
        bold_default: Default bold value when font is a string (backward compat)

    Returns:
        dict with keys: family (str), size (int), bold (bool), italic (bool)
    """
    font_def = fonts.get(font_key, "Arial")
    if isinstance(font_def, str):
        return {"family": _safe_font(font_def), "size": 14, "bold": bold_default, "italic": False}
    if isinstance(font_def, dict):
        return {
            "family": _safe_font(font_def.get("family", "Arial")),
            "size": font_def.get("size", 14),
            "bold": font_def.get("bold", bold_default) if font_def.get("bold") is not None else bold_default,
            "italic": font_def.get("italic", False) if font_def.get("italic") is not None else False,
        }
    return {"family": "Arial", "size": 14, "bold": bold_default, "italic": False}


def _get_font_with_overrides(font_key: str, fonts: dict, elem: dict, bold_default: bool = False) -> dict:
    """Resolve font from template, then apply element-level overrides.

    Args:
        font_key: Key in fonts dict, e.g. "heading" or "body"
        fonts: The template fonts dict
        elem: The element dict (may have font, font_size, bold, italic overrides)
        bold_default: Default bold value when font is a string

    Returns:
        dict with keys: family (str), size (int), bold (bool), italic (bool)
    """
    resolved = _resolve_font(font_key, fonts, bold_default)

    elem_font = elem.get("font")
    if isinstance(elem_font, str):
        resolved["family"] = _safe_font(elem_font)
    elif isinstance(elem_font, dict):
        if elem_font.get("family"):
            resolved["family"] = _safe_font(elem_font["family"])
        if elem_font.get("size") is not None:
            resolved["size"] = elem_font["size"]
        if elem_font.get("bold") is not None:
            resolved["bold"] = elem_font["bold"]
        if elem_font.get("italic") is not None:
            resolved["italic"] = elem_font["italic"]

    if elem.get("font_size") is not None:
        resolved["size"] = elem["font_size"]
    if elem.get("bold") is not None:
        resolved["bold"] = elem["bold"]
    if elem.get("italic") is not None:
        resolved["italic"] = elem["italic"]

    return resolved


def _build_presentation(presentation_json: dict, images: dict[str, bytes]) -> Any:
    template = presentation_json.get("template", {})
    slides_data = presentation_json.get("slides", [])

    width_inches = template.get("slide_width_inches", 13.33)
    height_inches = template.get("slide_height_inches", 7.5)

    prs = Presentation()
    prs.slide_width = Inches(width_inches)
    prs.slide_height = Inches(height_inches)

    colors = template.get("colors", {})
    fonts = template.get("fonts", {})

    blank_layout = prs.slide_layouts[6]

    for slide_data in slides_data:
        slide = prs.slides.add_slide(blank_layout)
        slide_type = slide_data.get("type", "content")
        _handle_slide_background(slide, slide_type, colors)
        _handle_slide_title(slide, slide_type, slide_data, colors, fonts, width_inches)
        table_config = template.get("table", {})
        _handle_slide_elements(slide, slide_data, colors, fonts, images, table_config)

    return prs


def _handle_slide_elements(slide: Slide, slide_data, colors, fonts, images: dict[str, bytes], table_config=None):
    elements = slide_data.get("elements", [])
    if elements:
        _add_elements_to_slide(slide, elements, colors, images, True, table_config or {})
        _add_elements_to_slide(slide, elements, colors, images, False, table_config or {})


def _handle_slide_title(slide: Slide, slide_type: Literal["title", "section"] | Any, slide_data, colors, fonts,
                        width_inches):
    title = slide_data.get("title", "")

    if title:
        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3), Inches(width_inches - 1), Inches(2.0)
        )
        _handle_alignment("left", title_box)

        tf = title_box.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(24)
        p.font.bold = True
        heading_font = _resolve_font("heading", fonts, bold_default=True)["family"]
        p.font.name = heading_font
        p.font.color.rgb = _resolve_font_color(
            "primary", colors, default_hex="#ffffff" if slide_type in ("title", "section") else "#333333"
        )


def _handle_slide_background(slide: Slide, slide_type, colors):
    bg_color = _resolve_color(colors, colors.get("background") or "#ffffff")
    _fill_slide_background(slide, bg_color)
    return bg_color


def _fix_duplicate_shape_ids(pptx_bytes: bytes) -> bytes:
    """Fix duplicate shape IDs in PPTX ZIP that cause corruption."""
    P_NS = 'http://schemas.openxmlformats.org/presentationml/2006/main'
    buf_in = BytesIO(pptx_bytes)
    buf_out = BytesIO()
    zf_in = zipfile.ZipFile(buf_in)
    zf_out = zipfile.ZipFile(buf_out, 'w', zipfile.ZIP_DEFLATED)

    next_id = [2]

    def get_next_id():
        nid = next_id[0]
        next_id[0] += 1
        return nid

    for item in zf_in.infolist():
        data = zf_in.read(item.filename)
        if item.filename.startswith('ppt/slides/slide') and item.filename.endswith('.xml'):
            try:
                tree = etree.fromstring(data)
                id_map = {}
                for cNvPr in tree.iter(f'{{{P_NS}}}cNvPr'):
                    old_id_str = cNvPr.get('id')
                    if old_id_str and old_id_str.isdigit():
                        if old_id_str not in id_map:
                            id_map[old_id_str] = str(get_next_id())
                        cNvPr.set('id', id_map[old_id_str])
                data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
            except Exception:
                pass
        zf_out.writestr(item, data)

    zf_in.close()
    zf_out.close()
    return buf_out.getvalue()
