"""PPTX reverse parser — converts PPTX files to presentation.json."""

import base64
import json
import logging
import re
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any
from typing import cast
from xml.etree import ElementTree as ET  # noqa: N806

from .schema_validation_utils import force_to_schema, validate_json_schema
from django.conf import settings

# noinspection HttpUrlsUsage
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
# noinspection HttpUrlsUsage
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
# noinspection HttpUrlsUsage
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

EMU_PER_INCH = 914400

LAYOUT_ARTIFACT_NAMES = {
    "Title",
    "Subtitle",
    "Date Placeholder",
    "Footer Placeholder",
    "Slide Number Placeholder",
}

CLRSCHEME_MAP = {
    "accent1": "primary",
    "accent2": "secondary",
    "accent3": "accent",
    "dk1": "text",
    "dk2": "secondary",
    "lt1": "background",
    "lt2": "primary",
    "tx1": "text",
    "tx2": "secondary",
    "bg1": "background",
    "bg2": "primary",
}

TITLE_PH_TYPES = {"title", "ctrTitle", "subTitle"}
ALIGN_MAP = {"l": "left", "ctr": "center", "r": "right", "just": "center", "left": "left", "right": "right", "center": "center"}
VALID_FONTS = {"Arial", "Helvetica", "Times New Roman", "Courier New", "Georgia"}
VALID_SLOTS = {"title", "body", "footer"}

SCHEMA_FILE = str(Path(__file__).resolve().parent.parent / "resources" / "schema" / "ppt_schema.json")

logger = logging.getLogger(__name__)

_PPT_SCHEMA_CACHE: str | None = None


def _load_ppt_schema() -> str:
    global _PPT_SCHEMA_CACHE
    if _PPT_SCHEMA_CACHE is None:
        try:
            with open(str(Path(settings.BASE_DIR) / SCHEMA_FILE)) as f:
                _PPT_SCHEMA_CACHE = f.read()
        except FileNotFoundError:
            raise Exception(f"Was unable to locate file: {SCHEMA_FILE}")
    assert _PPT_SCHEMA_CACHE is not None
    return _PPT_SCHEMA_CACHE


def _find_parent_element(root: ET.Element, target: ET.Element) -> ET.Element | None:
    """Find the parent element of target by traversing all descendants of root.

    Skips root itself to avoid a self-reference when target is root.
    """
    target_id = id(target)
    for parent in root.iter():
        if parent is target:
            continue
        for child in parent:
            if id(child) == target_id:
                return parent
    return None


def _find_root(element: ET.Element, tree_root: ET.Element | None = None) -> ET.Element:
    """Find the document root element by walking up from element until no parent is found.

    When tree_root is provided, walks up within that tree.
    When tree_root is None, first finds the true document root (the element no other
    element in the tree is a parent of), then walks up from element within that tree.
    """
    if tree_root is None:
        all_elements = list(element.iter())
        doc_root = element
        for candidate in all_elements:
            if not any(id(child) == id(candidate) for parent in all_elements for child in parent if
                       parent is not candidate):
                doc_root = candidate
                break
        tree_root = doc_root

    current: ET.Element = element
    while True:
        parent = _find_parent_element(tree_root, current)
        if parent is None:
            return current
        current = parent


def _xpath(element: ET.Element, file_context: str, root: ET.Element | None) -> str:
    """Build a readable xpath string for an element within a PPTX zip file.

    Walks from element up to root, then builds the path from root to element.

    Args:
        element: the element to describe
        file_context: e.g. "ppt/slides/slide1.xml"
        root: the root element of the tree element belongs to.

    Returns a string like:
        "ppt/slideMasters/slideMaster1.xml/spTree[1]/sp[3](@name="Title 1")"
    """
    if root is None:
        return file_context

    segments: list[str] = []
    current: ET.Element | None = element
    while current is not None:
        parent = _find_parent_element(root, current)
        if parent is None:
            break

        siblings = [s for s in parent if s.tag == current.tag]
        try:
            idx = siblings.index(current) + 1
        except ValueError:
            idx = 1

        tag = current.tag.split("}")[-1] if "}" in current.tag else current.tag

        name_tag = f"{{{P_NS}}}nvSpPr/{{{P_NS}}}cNvPr"
        name_el = current.find(name_tag)
        name_part = f"(@name={name_el.get('name', '')!r})" if name_el is not None else ""

        ph_type_tag = f"{{{P_NS}}}nvSpPr/{{{P_NS}}}nvPr/{{{P_NS}}}ph"
        ph_el = current.find(ph_type_tag)
        ph_part = f"/ph:type={ph_el.get('type', '')!r}" if ph_el is not None else ""

        segments.append(f"{tag}[{idx}]{name_part}{ph_part}")
        current = parent

    path = file_context
    for seg in reversed(segments):
        path += "/" + seg

    return path


def _emu_to_inches(emu: int | float) -> float:
    """Convert EMU (English Metric Unit) to inches. 914400 EMU = 1 inch."""
    return emu / EMU_PER_INCH


def _halfpt_to_pt(halfpt: int) -> int:
    """Convert half-points (OOXML sz units) to points. sz is in 100ths of a point."""
    return halfpt // 100


def _parse_sld_sz(zf: zipfile.ZipFile) -> tuple[float, float]:
    """Extract slide dimensions (width, height in inches) from presentation.xml sldSz element."""
    try:
        pres_xml = zf.read("ppt/presentation.xml")
        tree = ET.fromstring(pres_xml)
        sldSz = tree.find(f"{{{P_NS}}}sldSz")  # noqa: N806
        if sldSz is not None:
            cx = int(sldSz.get("cx", 12192000))
            cy = int(sldSz.get("cy", 6858000))
            return _emu_to_inches(cx), _emu_to_inches(cy)
    except Exception as exp:
        logger.warning(f"_parse_sld_sz failed due to {exp}", exc_info=True)
        pass
    return 13.33, 7.5


def _read_rel(zf: zipfile.ZipFile, rel_path: str) -> dict[str, str]:
    """Parse a .rels XML file and return a mapping of Id -> Target."""
    try:
        rels_xml = zf.read(rel_path)
        tree = ET.fromstring(rels_xml)
        rels = {}
        for rel in tree:
            rels[rel.get("Id", "")] = rel.get("Target", "")
        return rels
    except Exception as exp:
        logger.warning(f"_read_rel failed due to {exp}", exc_info=True)
        return {}


def _read_master_rels(zf: zipfile.ZipFile) -> dict[str, str]:
    """Return the slideMaster1 rels mapping (Id -> Target)."""
    return _read_rel(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels")


def _resolve_media_path(zf: zipfile.ZipFile, slide_name: str, r_embed: str) -> str | None:
    """Resolve the media path for an r:embed reference in a slide or layout."""
    base_name = slide_name.replace("slideLayout", "slideLayout").replace("slide", "slide").split(".xml")[0]
    slide_num_or_layout = "".join(c for c in base_name if c.isdigit())

    if "Layout" in slide_name:
        rel_path = f"ppt/slideLayouts/_rels/slideLayout{slide_num_or_layout}.xml.rels"
    else:
        rel_path = f"ppt/slides/_rels/slide{slide_num_or_layout}.xml.rels"

    rels = _read_rel(zf, rel_path)
    target = rels.get(r_embed, "")
    if not target:
        return None
    if target.startswith("../"):
        target = target[3:]
    parts = target.split("/")
    if len(parts) > 2 and parts[0] == ".." and parts[1] == "media":
        target = "ppt/media/" + parts[2]
    elif not target.startswith("ppt/media/"):
        target = "ppt/media/" + target.split("/")[-1]
    return target


def _find_theme_path(zf: zipfile.ZipFile) -> str | None:  # noqa: N806
    """Find the path to the theme XML within the PPTX zip, or None if not found."""
    master_rels = _read_master_rels(zf)
    for rId, target in master_rels.items():  # noqa: N806
        if "theme" in target.lower():
            path = target.lstrip("../")
            if not path.startswith("ppt/"):
                path = "ppt/" + path
            if path in zf.namelist():
                return path
            return path
    return None


def _read_theme_colors_and_fonts(zf: zipfile.ZipFile) -> tuple[dict, dict, dict]:
    """Parse theme1.xml to extract colors, fonts, and the clrMap from the slide master."""
    theme_path = _find_theme_path(zf)
    if not theme_path:
        return {}, {}, {}
    try:
        theme_xml = zf.read(theme_path)
        tree = ET.fromstring(theme_xml)
        themeElements = tree.find(f"{{{A_NS}}}themeElements")  # noqa: N806
        if themeElements is None:
            return {}, {}, {}
        clrScheme = themeElements.find(f"{{{A_NS}}}clrScheme")  # noqa: N806

        clrMap = {}  # noqa: N806
        try:
            master_xml = zf.read("ppt/slideMasters/slideMaster1.xml")
            master_tree = ET.fromstring(master_xml)
            for mc in master_tree.iter(f"{{{P_NS}}}clrMap"):
                for attr, val in mc.attrib.items():
                    if attr not in ("xlmns", "xmlns") and val:
                        clrMap[attr] = val
        except Exception as exp:
            logger.warning(f"read from slide master failed due to {exp}", exc_info=True)
            pass

        colors = {}
        tag_to_hex = {}
        if clrScheme is not None:
            for child in clrScheme:
                tag = child.tag.split("}")[-1]
                srgb = child.find(f"{{{A_NS}}}srgbClr")
                sys_clr = child.find(f"{{{A_NS}}}sysClr")
                if srgb is not None:
                    hex_val = f"#{srgb.get('val', '')}"
                    tag_to_hex[tag] = hex_val
                    mapped = CLRSCHEME_MAP.get(tag)
                    if mapped:
                        colors[mapped] = hex_val
                elif sys_clr is not None:
                    hex_val = f"#{sys_clr.get('lastClr', '')}"
                    tag_to_hex[tag] = hex_val
                    mapped = CLRSCHEME_MAP.get(tag)
                    if mapped:
                        colors[mapped] = hex_val
        if clrMap:
            for clr_key, theme_key in clrMap.items():
                if theme_key in tag_to_hex:
                    colors[clr_key] = tag_to_hex[theme_key]

        for tag, hex_val in tag_to_hex.items():
            if tag not in colors:
                colors[tag] = hex_val

        if "bg1" not in colors and "lt1" in colors:
            colors["bg1"] = colors["lt1"]
        if "tx1" not in colors and "dk1" in colors:
            colors["tx1"] = colors["dk1"]

        fontScheme = themeElements.find(f"{{{A_NS}}}fontScheme")  # noqa: N806
        fonts = {}
        if fontScheme is not None:
            major_font = fontScheme.find(f"{{{A_NS}}}majorFont")
            minor_font = fontScheme.find(f"{{{A_NS}}}minorFont")
            if major_font is not None:
                latin = major_font.find(f"{{{A_NS}}}latin")
                family = latin.get("typeface") if latin is not None else "Arial"
                fonts["heading"] = {"family": family}
            if minor_font is not None:
                latin = minor_font.find(f"{{{A_NS}}}latin")
                family = latin.get("typeface") if latin is not None else "Arial"
                fonts["body"] = {"family": family}

        return colors, fonts, {}
    except Exception as exp:
        logger.warning(f"_read_theme_colors_and_fonts failed due to {exp}", exc_info=True)
        return {}, {}, {}


def _resolve_text_color_from_rPr(rPr: ET.Element | None, colors: dict | None = None) -> str | None:  # noqa: N806
    """Resolve the effective text color for a run.

    Checks solidFill on rPr first. If none, walks up the style hierarchy:
    run-level pPr/defRPr → paragraph-level pPr/defRPr → lstStyle/lvl1pPr/defRPr.
    Falls back to tx1 (default body text color from txStyles) when no explicit color.
    """
    if rPr is None:
        return None

    solid_fill = rPr.find(f"{{{A_NS}}}solidFill")
    if solid_fill is not None:
        result = _solid_fill_to_hex(solid_fill, colors)
        if result:
            return result

    defRPr = rPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
    if defRPr is not None:
        result = _solid_fill_to_hex(defRPr.find(f"{{{A_NS}}}solidFill"), colors)
        if result:
            return result

    pPr = rPr.find(f"{{{A_NS}}}pPr")  # noqa: N806
    if pPr is not None:
        result = _solid_fill_to_hex(pPr.find(f"{{{A_NS}}}solidFill"), colors)
        if result:
            return result
        pPr_defRPr = pPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
        if pPr_defRPr is not None:
            result = _solid_fill_to_hex(pPr_defRPr.find(f"{{{A_NS}}}solidFill"), colors)
            if result:
                return result

    txBody = rPr.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is not None:
        lstStyle = txBody.find(f"{{{A_NS}}}lstStyle")  # noqa: N806
        if lstStyle is not None:
            lvl1pPr = lstStyle.find(f"{{{A_NS}}}lvl1pPr")  # noqa: N806
            if lvl1pPr is not None:
                lst_defRPr = lvl1pPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
                if lst_defRPr is not None:
                    result = _solid_fill_to_hex(lst_defRPr.find(f"{{{A_NS}}}solidFill"), colors)
                    if result:
                        return result

    if colors:
        return colors.get("tx1")
    return None


def _apply_luminance_modifiers(base_hex: str, lumMod: ET.Element | None, lumOff: ET.Element | None) -> str:  # noqa: N806
    """Apply lumMod/lumOff children of a schemeClr to produce an adjusted hex color."""
    if not base_hex or len(base_hex) != 7 or base_hex[0] != "#":
        return base_hex
    r = int(base_hex[1:3], 16)
    g = int(base_hex[3:5], 16)
    b = int(base_hex[5:7], 16)
    mod_val = float(lumMod.get("val", "100000")) / 100000.0 if lumMod is not None else 1.0
    off_val = float(lumOff.get("val", "0")) / 100000.0 if lumOff is not None else 0.0
    r_final = min(255, int(r * mod_val + 255 * off_val))
    g_final = min(255, int(g * mod_val + 255 * off_val))
    b_final = min(255, int(b * mod_val + 255 * off_val))
    return f"#{r_final:02X}{g_final:02X}{b_final:02X}"


def _solid_fill_to_hex(solid_fill: ET.Element | None, colors: dict | None = None) -> str | None:  # noqa: N806
    """Convert a p:solidFill element to a hex color string, or None if not resolvable."""
    if solid_fill is None:
        return None
    srgb = solid_fill.find(f"{{{A_NS}}}srgbClr")
    if srgb is not None:
        return f"#{srgb.get('val', '')}"
    scheme = solid_fill.find(f"{{{A_NS}}}schemeClr")
    if scheme is not None and colors:
        color_name = scheme.get("val")
        if color_name and color_name in colors:
            base_hex = colors[color_name]
            lumMod = scheme.find(f"{{{A_NS}}}lumMod")  # noqa: N806
            lumOff = scheme.find(f"{{{A_NS}}}lumOff")  # noqa: N806
            if lumMod is not None or lumOff is not None:
                return _apply_luminance_modifiers(base_hex, lumMod, lumOff)
            return base_hex
    return None


def _bg_color_from_sp(sp: ET.Element, colors: dict | None = None) -> str | None:
    """Extract the background color from a p:sp element, or None if not present."""
    bg = sp.find(f"{{{P_NS}}}bg")
    if bg is None:
        return None
    solid_fill = bg.find(f"{{{A_NS}}}solidFill")
    return _solid_fill_to_hex(solid_fill, colors)


def _collect_slide_text(sp: ET.Element, colors: dict | None = None) -> list[
    tuple[str, str | None, int | None, bool, bool, str | None, str | None]]:
    """Collect all text runs from a p:sp's txBody. Returns list of (text, font_family, font_size, bold, italic, align, color) tuples."""
    results: list[Any] = []

    txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is None:
        return results

    for para in txBody.findall(f"{{{A_NS}}}p"):
        align = para.get("algn", "l")
        resolved_align = ALIGN_MAP.get(align, "left")

        for run in para.findall(f"{{{A_NS}}}r"):
            rPr = run.find(f"{{{A_NS}}}rPr")  # noqa: N806
            if rPr is None:
                continue

            t_elem = run.find(f"{{{A_NS}}}t")
            if t_elem is None or not t_elem.text:
                continue

            text = t_elem.text
            latin = rPr.find(f"{{{A_NS}}}latin")
            font_family = latin.get("typeface", "") if latin is not None else ""

            sz = rPr.get("sz")
            font_size = _halfpt_to_pt(int(sz)) if sz else None

            bold = rPr.get("b") is not None
            italic = rPr.get("i") is not None

            color = _resolve_text_color_from_rPr(rPr, colors)
            results.append((text, font_family, font_size, bold, italic, resolved_align, color))

        end_para = para.find(f"{{{A_NS}}}endParaRPr")
        if end_para is not None and results:
            pass

    return results


def _parse_rect(sp: ET.Element, colors: dict | None = None) -> dict | None:
    """Extract fill and stroke from a p:sp's spPr to produce a rect element dict, or None if no fill."""
    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    if spPr is None:
        return None

    solid_fill = spPr.find(f"{{{A_NS}}}solidFill")
    fill_color = _solid_fill_to_hex(solid_fill, colors)

    stroke_color = None
    stroke_width = None
    ln = spPr.find(f"{{{A_NS}}}ln")
    if ln is not None:
        w = ln.get("w")
        if w:
            stroke_width = round(int(w) / EMU_PER_INCH, 4)
        ln_fill = ln.find(f"{{{A_NS}}}solidFill")
        stroke_color = _solid_fill_to_hex(ln_fill, colors)

    return {
        "fill": fill_color,
        "stroke": stroke_color,
        "width": stroke_width,
    }


def _get_paragraph_align(para: ET.Element) -> str:  # noqa: N806
    """Return the text alignment for a paragraph element from its pPr algn attribute."""
    pPr = para.find(f"{{{A_NS}}}pPr")  # noqa: N806
    para_align = pPr.get("algn", "l") if pPr is not None else "l"
    return ALIGN_MAP.get(para_align, "left")


def _parse_xfrm_and_chxfrm(xfrm: ET.Element) -> tuple[dict | None, dict | None]:
    """Parse grpSpPr xfrm element into (xfrm_data, ch_xfrm_data) dicts or (None, None) on failure.
    
    xfrm_data: group transform in inches (EMU→inches)
    ch_xfrm_data: child transform in raw EMU
    """
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    chOff = xfrm.find(f"{{{A_NS}}}chOff")  # noqa: N806
    chExt = xfrm.find(f"{{{A_NS}}}chExt")  # noqa: N806
    xfrm_data: dict[str, int | float] | None = None
    ch_xfrm_data: dict[str, int | float] | None = None
    if off is not None and ext is not None:
        x_val, y_val, cx_val, cy_val = off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")
        if None not in (x_val, y_val, cx_val, cy_val):
            xfrm_data = {
                "x": _emu_to_inches(int(cast(str, x_val))),
                "y": _emu_to_inches(int(cast(str, y_val))),
                "width": _emu_to_inches(int(cast(str, cx_val))),
                "height": _emu_to_inches(int(cast(str, cy_val))),
            }
    if chOff is not None and chExt is not None:
        ch_x, ch_y, ch_cx, ch_cy = chOff.get("x"), chOff.get("y"), chExt.get("cx"), chExt.get("cy")
        if None not in (ch_x, ch_y, ch_cx, ch_cy):
            ch_xfrm_data = {
                "x": int(cast(str, ch_x)),
                "y": int(cast(str, ch_y)),
                "width": int(cast(str, ch_cx)),
                "height": int(cast(str, ch_cy)),
            }
    return xfrm_data, ch_xfrm_data


def _get_layout_text_fallback(ph_type: str, ph_idx: str, layout_positions: dict) -> tuple[str, bool, dict]:
    """Try to get text content from layout_positions as fallback. Returns (text, uses_layout_fallback, pos_data)."""
    if not ph_idx:
        ph_idx = ""
    for key in [f"{ph_type}:{ph_idx}" if ph_idx else None, f"ph:{ph_type}", f"{ph_type}:"]:
        if key and key in layout_positions:
            lt = layout_positions[key].get("layout_text")
            if lt:
                return lt, True, layout_positions[key]
    return "", False, {}


def _get_placeholder_attrs(sp: ET.Element) -> tuple[str, str]:  # noqa: N806
    """Extract (ph_type, ph_idx) from a shape's nvSpPr/ph element. Both default to ''."""
    nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
    if nvSpPr is None:
        return "", ""
    nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
    if nvPr is None:
        return "", ""
    ph = nvPr.find(f"{{{P_NS}}}ph")
    if ph is None:
        return "", ""
    return ph.get("type", ""), ph.get("idx", "")


def _rPr_has_font_attrs(rPr: ET.Element | None) -> bool:  # noqa: N806
    """Return True if rPr has explicit font attributes (latin, solidFill, sz, b, i)."""
    if rPr is None:
        return False
    return (
            rPr.find(f"{{{A_NS}}}latin") is not None
            or rPr.find(f"{{{A_NS}}}solidFill") is not None
            or rPr.get("sz") is not None
            or rPr.get("b") is not None
            or rPr.get("i") is not None
    )


def _apply_layout_font_color_fallback(result: dict, ph_type: str, ph_idx: str | None, layout_positions: dict) -> None:
    """Apply font_color from layout_positions to result if not already set."""
    if result.get("font_color") or not ph_type or not layout_positions:
        return
    for key in [f"{ph_type}:{ph_idx}" if ph_idx else None, f"ph:{ph_type}", f"{ph_type}:"]:
        if key and key in layout_positions:
            lp_fc = layout_positions[key].get("font_color")
            if lp_fc:
                result["font_color"] = lp_fc
                return
    if ph_idx:
        idx_key = f":{ph_idx}"
        if idx_key in layout_positions:
            lp_fc = layout_positions[idx_key].get("font_color")
            if lp_fc:
                result["font_color"] = lp_fc


def _build_rect_from_fill(fill: str | None, stroke: str | None, xfrm_data: dict) -> dict:
    """Build a rect element dict from fill/stroke color strings and position data."""
    rect_elem: dict[str, Any] = {"type": "rect", "position": xfrm_data}
    if fill:
        rect_elem["fill"] = fill
    if stroke:
        rect_elem["stroke"] = stroke
    return rect_elem


def _merge_pos_data(existing: dict, new: dict) -> None:
    """Merge non-None values from new pos_data into existing pos_data (in-place)."""
    if existing.get("font_size") is None and new.get("font_size") is not None:
        existing["font_size"] = new["font_size"]
    if not existing.get("font_family") and new.get("font_family"):
        existing["font_family"] = new["font_family"]
    if not existing.get("bold") and new.get("bold"):
        existing["bold"] = new["bold"]
    if not existing.get("italic") and new.get("italic"):
        existing["italic"] = new["italic"]
    if not existing.get("font_color") and new.get("font_color"):
        existing["font_color"] = new["font_color"]


def _parse_txstyles_from_master(master_xml: bytes | str) -> dict:
    """Extract title/body style info (sz, bold, italic, align) from p:txStyles block."""
    master_str = master_xml.decode("utf-8", errors="replace") if isinstance(master_xml, bytes) else master_xml
    result: dict[str, Any] = {
        "title_sz": None, "title_bold": None, "title_italic": None, "title_algn": None,
        "body_sz": None, "body_algn": None,
    }
    txstyles_match = re.search(r"<p:txStyles>.*?</p:txStyles>", master_str, re.DOTALL)
    if not txstyles_match:
        return result
    title_style_match = re.search(r"<p:titleStyle>.*?</p:titleStyle>", txstyles_match.group(), re.DOTALL)
    if title_style_match:
        ts = title_style_match.group()
        sz_m = re.search(r'sz="(\d+)"', ts)
        if sz_m:
            result["title_sz"] = _halfpt_to_pt(int(sz_m.group(1)))
        if re.search(r'b="1"', ts):
            result["title_bold"] = True
        if re.search(r'i="1"', ts):
            result["title_italic"] = True
        algn_m = re.search(r'algn="([^"]+)"', ts)
        if algn_m:
            result["title_algn"] = ALIGN_MAP.get(algn_m.group(1), "left")
    body_style_match = re.search(r"<p:bodyStyle>.*?</p:bodyStyle>", txstyles_match.group(), re.DOTALL)
    if body_style_match:
        bs = body_style_match.group()
        sz_m = re.search(r'sz="(\d+)"', bs)
        if sz_m:
            result["body_sz"] = _halfpt_to_pt(int(sz_m.group(1)))
        algn_m = re.search(r'algn="([^"]+)"', bs)
        if algn_m:
            result["body_algn"] = ALIGN_MAP.get(algn_m.group(1), "left")
    return result


def _determine_slide_type(idx: int, num_slides: int) -> str:
    """Return slide type based on position: title (first), closing (last), or content."""
    if num_slides == 1:
        return "title"
    if idx == 0:
        return "title"
    if idx == num_slides - 1:
        return "closing"
    return "content"


def _fix_up_custgeom_ns(spPr_str: str) -> str:  # noqa: N806
    """Replace ns0/ns1 namespace aliases with p/a in a custGeom spPr XML string."""
    return (
        spPr_str.replace(f"xmlns:ns0=\"{P_NS}\"", f"xmlns:p=\"{P_NS}\"")
        .replace(f"xmlns:ns1=\"{A_NS}\"", f"xmlns:a=\"{A_NS}\"")
        .replace("ns0:", "p:")
        .replace("ns1:", "a:")
    )


def _get_layout_entry(layout_positions: dict, ph_type: str, ph_idx: str | None) -> dict | None:
    """Return the first matching position dict from layout_positions.

    Key priority: {ph_type}:{ph_idx} → ph:{ph_type} → {ph_type}:
    """
    for key in (f"{ph_type}:{ph_idx}" if ph_idx else None, f"ph:{ph_type}", f"{ph_type}:"):
        if key and key in layout_positions:
            return layout_positions[key]
    return None


def _get_layout_font(layout_positions: dict, ph_type: str, ph_idx: str | None) -> dict | None:
    """Return the layout position entry (dict) for a placeholder, or None."""
    return _get_layout_entry(layout_positions, ph_type, ph_idx)


def _get_layout_color(layout_positions: dict, ph_type: str, ph_idx: str | None) -> str | None:
    """Return the font_color from the layout position entry, or None."""
    lp = _get_layout_entry(layout_positions, ph_type, ph_idx)
    return lp.get("font_color") if lp else None


def _apply_layout_font_fill(
        font_family: str, font_size: int | None, bold: bool, italic: bool, align: str,
        lp: dict
) -> tuple[str, int | None, bool, bool, str]:
    """Fill in missing font attrs from a layout_positions entry. Returns updated (font_family, font_size, bold, italic, align)."""
    layout_ff = lp.get("font_family", "")
    layout_sz = lp.get("font_size")
    layout_bold = lp.get("bold")
    layout_italic = lp.get("italic")
    layout_align = lp.get("align")
    if layout_ff:
        font_family = layout_ff
    if layout_sz and font_size is None:
        font_size = layout_sz
    if layout_bold and not bold:
        bold = True
    if layout_italic and not italic:
        italic = True
    if layout_align and align in ("left", "ctr"):
        align = cast(str, layout_align)
    return font_family, font_size, bold, italic, align


def _apply_layout_fallback(
        ph_type: str, ph_idx: str,
        layout_positions: dict,
        font_family: str, font_size: int | None, bold: bool, italic: bool,
        font_color: str | None, align: str
) -> tuple[str, int | None, bool, bool, str | None, str]:
    """Apply layout-position fallback for all text properties. Returns updated values."""
    if ph_type and layout_positions:
        lp = _get_layout_font(layout_positions, ph_type, ph_idx)
        if lp:
            if not font_family:
                font_family = lp.get("font_family", "")
            if font_size is None:
                font_size = lp.get("font_size")
            if lp.get("bold") and not bold:
                bold = True
            if lp.get("italic") and not italic:
                italic = True
            if not font_color:
                font_color = lp.get("font_color")
            if lp.get("align") and align in ("left", "ctr"):
                align = cast(str, lp.get("align"))

    if ph_type and layout_positions and not font_color:
        font_color = _get_layout_color(layout_positions, ph_type, ph_idx)

    return font_family, font_size, bold, italic, font_color, align


def _build_font_dict(font_family: str, font_size: int | None, bold: bool, italic: bool) -> dict:
    """Build a font dict from font attributes. Mirrors old inline behavior exactly."""
    result: dict[str, Any] = {}
    if font_family:
        result["family"] = font_family
    if font_size is not None:
        result["size"] = font_size
    if bold:
        result["bold"] = True
    if italic:
        result["italic"] = True
    return result


def _resolve_run_properties(first_run_rPr: ET.Element | None,  # noqa: N806
                            first_run_para: ET.Element | None) -> ET.Element | None:  # noqa: N806
    """Resolve effective run properties by walking run → pPr/defRPr hierarchy."""
    if first_run_rPr is not None:
        return first_run_rPr
    if first_run_para is not None:
        pPr = first_run_para.find(f"{{{A_NS}}}pPr")  # noqa: N806
        if pPr is not None:
            defRPr = pPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
            if defRPr is not None:
                return defRPr
    return None


def _extract_rPr_font_attrs(rPr: ET.Element) -> tuple[str, int | None, bool, bool]:  # noqa: N806
    """Extract font_family, font_size, bold, italic from an rPr element."""
    latin = rPr.find(f"{{{A_NS}}}latin")
    font_family = latin.get("typeface", "") if latin is not None else ""
    sz = rPr.get("sz")
    font_size = _halfpt_to_pt(int(sz)) if sz else None
    bold_attr = rPr.get("b")
    bold = bold_attr is not None and bold_attr != "0"
    italic_attr = rPr.get("i")
    italic = italic_attr is not None and italic_attr != "0"
    return font_family, font_size, bold, italic


def _parse_text_element(sp: ET.Element, xfrm_data: dict, layout_positions=None, theme_colors: dict | None = None,
                        theme_fonts: dict | None = None) -> dict | None:  # noqa: N806
    """Parse a p:sp text body into a text element dict with font, color, and alignment."""
    if layout_positions is None:
        layout_positions = {}

    txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is None:
        return None

    font_family = ""
    font_size: int | None = None
    bold = False
    italic = False
    font_color = None
    align = "left"
    ph_type = ""
    ph_idx = None
    uses_layout_fallback = False

    first_para = txBody.find(f"{{{A_NS}}}p")
    if first_para is not None:
        align = _get_paragraph_align(first_para)

    collected_text_parts: list[str] = []
    first_run_rPr = None  # noqa: N806
    first_run_para = None
    run_index = 0

    for para in txBody.findall(f"{{{A_NS}}}p"):
        for run in para.findall(f"{{{A_NS}}}r"):
            t_elem = run.find(f"{{{A_NS}}}t")
            if t_elem is None or not t_elem.text:
                run_index += 1
                continue

            collected_text_parts.append(t_elem.text)
            if first_run_rPr is None:
                first_run_rPr = run  # noqa: N806
                first_run_para = para
            run_index += 1

    first_run_text = "\n".join(collected_text_parts)

    if first_run_text and first_run_rPr is not None:
        rPr_elem = first_run_rPr.find(f"{{{A_NS}}}rPr")  # noqa: N806
        rPr: ET.Element | None = rPr_elem if rPr_elem is not None else first_run_rPr  # noqa: N806
        rPr = _resolve_run_properties(rPr, first_run_para)  # noqa: N806

        ph_type, ph_idx = _get_placeholder_attrs(sp)

        if rPr is not None and not _rPr_has_font_attrs(rPr):
            pPr = first_run_para.find(f"{{{A_NS}}}pPr") if first_run_para is not None else None  # noqa: N806
            if pPr is not None:
                defRPr = pPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
                if defRPr is not None:
                    rPr = defRPr  # noqa: N806

        if rPr is not None:
            font_family, font_size, bold, italic = _extract_rPr_font_attrs(rPr)

            layout_color = None
            effective_ph_type = ph_type if ph_type else ("body" if ph_idx else None)
            if effective_ph_type:
                layout_color = _get_layout_color(layout_positions, effective_ph_type, ph_idx)
                if layout_color is None and ph_idx:
                    layout_color = layout_positions.get(f":{ph_idx}", {}).get("font_color")

            color = _resolve_text_color_from_rPr(rPr, theme_colors)
            font_color = layout_color if layout_color is not None else color

            if not font_family or font_size is None or not bold:
                if effective_ph_type:
                    lp = _get_layout_font(layout_positions, effective_ph_type, ph_idx)
                    if lp:
                        font_family, font_size, bold, italic, align = _apply_layout_font_fill(
                            font_family, font_size, bold, italic, align, lp
                        )
        else:
            font_family, font_size, bold, italic, font_color, align = _apply_layout_fallback(
                ph_type, ph_idx, layout_positions,
                font_family, font_size, bold, italic, font_color, align
            )

    if not first_run_text:
        ph_type, ph_idx = _get_placeholder_attrs(sp)
        first_run_text, uses_layout_fallback, pos = _get_layout_text_fallback(ph_type, ph_idx, layout_positions)
        if first_run_text:
            if not font_family:
                font_family = pos.get("font_family", "")
            if font_size is None and pos.get("font_size") is not None:
                font_size = pos.get("font_size")
            if not bold:
                bold = pos.get("bold", False)
            if not italic:
                italic = pos.get("italic", False)
            if not font_color:
                font_color = pos.get("font_color")
            if align in ("left", "ctr"):
                layout_align = pos.get("align")
                if layout_align:
                    align = layout_align
        if not first_run_text:
            return None

    if not font_family and theme_fonts:
        if ph_type in TITLE_PH_TYPES:
            font_family = theme_fonts.get("heading", {}).get("family", "")
        else:
            font_family = theme_fonts.get("body", {}).get("family", "")

    result: dict[str, Any] = {
        "type": "text",
        "position": xfrm_data,
        "text": first_run_text,
        "align": align,
    }
    if uses_layout_fallback:
        result["uses_layout_fallback"] = True
    result["font"] = _build_font_dict(font_family, font_size, bold, italic)
    if font_color:
        result["font_color"] = font_color
    elif theme_colors:
        tx1_color = theme_colors.get("tx1")
        if tx1_color:
            result["font_color"] = tx1_color

    if not result.get("font_color") and ph_type and layout_positions:
        _apply_layout_font_color_fallback(result, ph_type, ph_idx, layout_positions)

    return result


def _parse_image(pic: ET.Element, slide_name: str, xfrm_data: dict, zf: zipfile.ZipFile) -> dict | None:
    """Parse a p:pic element into an image element dict, or None if no blip is found."""
    blipFill = pic.find(f"{{{P_NS}}}blipFill")  # noqa: N806
    if blipFill is None:
        return None
    blip = blipFill.find(f"{{{A_NS}}}blip")
    if blip is None:
        return None

    r_embed = blip.get(f"{{{R_NS}}}embed")
    if not r_embed:
        for elem in blip.iter():
            r_embed = elem.get(f"{{{R_NS}}}embed")
            if r_embed:
                break
    if not r_embed:
        return None

    media_path = _resolve_media_path(zf, slide_name, r_embed)
    if not media_path:
        return None

    try:
        media_bytes = zf.read(media_path)

        if media_path.lower().endswith(".svg"):
            svg_text = media_bytes.decode("utf-8", errors="replace")
            if svg_text.startswith("<?xml"):
                try:
                    decoded = base64.b64decode(svg_text)
                    svg_text = decoded.decode("utf-8", errors="replace")
                except Exception as exp:
                    logger.warning(f"_parse_image - decode svg media failed due to {exp}", exc_info=True)
                    pass
            return {
                "type": "image",
                "position": xfrm_data,
                "svg_data": svg_text,
            }

        b64 = base64.b64encode(media_bytes).decode("ascii")
        return {
            "type": "image",
            "position": xfrm_data,
            "image_data": b64,
        }
    except Exception as exp:
        logger.warning(f"_parse_image failed due to {exp}", exc_info=True)
        return None


def _parse_table(graphic_frame: ET.Element, xfrm_data: dict) -> dict | None:  # noqa: N806
    """Parse a graphicFrame containing a table into a table element dict, or None if no tbl found."""
    tbl = graphic_frame.find(f".//{{{A_NS}}}tbl")
    if tbl is None:
        return None

    tblGrid = tbl.find(f"{{{A_NS}}}tblGrid")  # noqa: N806
    if tblGrid is None:
        return None

    columns = []
    for gridCol in tblGrid.findall(f"{{{A_NS}}}gridCol"):  # noqa: N806
        w = gridCol.get("w")
        width_inches = _emu_to_inches(int(w)) if w else 1.0
        columns.append({"header": "", "width": width_inches})

    rows_data = []
    for row_idx, tr in enumerate(tbl.findall(f"{{{A_NS}}}tr")):
        row_cells = []
        for tc in tr.findall(f"{{{A_NS}}}tc"):
            cell_text = ""
            for t in tc.iter(f"{{{A_NS}}}t"):
                if t.text:
                    cell_text += t.text
            row_cells.append(cell_text)
        rows_data.append(row_cells)

    if not rows_data:
        return None

    first_row = rows_data[0]
    for i, cell_text in enumerate(first_row):
        if i < len(columns):
            columns[i]["header"] = cell_text

    table_elem = {
        "type": "table",
        "position": xfrm_data,
        "columns": columns,
        "rows": rows_data[1:] if len(rows_data) > 1 else [],
    }

    return table_elem


def _build_sp_custgeom(spPr: ET.Element, xfrm_data: dict) -> dict | None:  # noqa: N806
    """Build a shape_data image dict from custGeom spPr. Returns None if no custGeom."""
    if spPr is None:
        return None
    has_custGeom = spPr.find(f"{{{A_NS}}}custGeom") is not None  # noqa: N806
    if not has_custGeom:
        return None
    spPr_str = ET.tostring(spPr, encoding="unicode")  # noqa: N806
    shape_data = _fix_up_custgeom_ns(spPr_str)
    return {
        "type": "image",
        "position": xfrm_data,
        "shape_data": shape_data,
    }


def _apply_layout_rect_fallback(results: list, ph_type: str, ph_idx: str, layout_positions: dict,
                                xfrm_data: dict) -> None:
    """Append a rect from layout_positions fill if no fill/stroke rect exists in results."""
    if any(r.get("type") == "rect" and (r.get("fill") or r.get("stroke")) for r in results):
        return
    lp = _get_layout_entry(layout_positions, ph_type, ph_idx)
    if lp:
        layout_fill = lp.get("fill")
        if layout_fill:
            results.append(_build_rect_from_fill(layout_fill, None, xfrm_data))


def _parse_sp_shape(sp: ET.Element, layout_positions: dict, theme_colors: dict | None = None,
                    theme_fonts: dict | None = None, slide_name: str = "slide", tree_root: ET.Element | None = None) \
        -> list[dict]:
    """Parse a p:sp element into a list of element dicts (text, rect, image, etc.)."""
    results: list[dict] = []

    nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
    if nvSpPr is None:
        return results

    cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
    if cNvPr is None:
        return results

    name = cNvPr.get("name", "")
    if name in LAYOUT_ARTIFACT_NAMES:
        return results

    nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
    ph = nvPr.find(f"{{{P_NS}}}ph") if nvPr is not None else None
    ph_type = ph.get("type", "") if ph is not None else ""
    ph_idx = ph.get("idx", "") if ph is not None else ""

    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    xfrm_data: dict[str, int | float] | None = None

    if spPr is not None:
        xfrm = spPr.find(f"{{{A_NS}}}xfrm")
        if xfrm is not None:
            off = xfrm.find(f"{{{A_NS}}}off")
            ext = xfrm.find(f"{{{A_NS}}}ext")
            if off is not None and ext is not None:
                xfrm_data = {
                    "x": _emu_to_inches(int(off.get("x", 0))),
                    "y": _emu_to_inches(int(off.get("y", 0))),
                    "width": _emu_to_inches(int(ext.get("cx", 914400))),
                    "height": _emu_to_inches(int(ext.get("cy", 914400))),
                }

    if xfrm_data is None:
        effective_ph_type = ph_type if ph_type else ("body" if ph is not None else None)
        if effective_ph_type:
            xfrm_data = _get_pos_from_layout(effective_ph_type, ph_idx, layout_positions, require_valid=True)
            if xfrm_data is None and name and f"master:name={name}" in layout_positions:
                pos = layout_positions[f"master:name={name}"]
                xfrm_data = {
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "width": pos.get("width"),
                    "height": pos.get("height"),
                }
                if pos.get("fill"):
                    if spPr is not None:
                        ln = spPr.find(f"{{{A_NS}}}ln")
                        if ln is None or ln.find(f"{{{A_NS}}}noFill") is not None:
                            rect_data = _parse_rect(sp, theme_colors)
                            if rect_data and not rect_data.get("fill"):
                                rect_data["fill"] = pos["fill"]

    if xfrm_data is None:
        logger.warning(f"_parse_sp_shape: no position found for shape at {_xpath(sp, slide_name, tree_root)}, skipping")
        return results

    has_txbody = sp.find(f"{{{P_NS}}}txBody") is not None

    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    rect_data = _parse_rect(sp, theme_colors) if spPr is not None else None
    if rect_data is not None:
        has_fill = rect_data.get("fill")
        has_stroke = rect_data.get("stroke")
        if has_fill or has_stroke:
            results.append(_build_rect_from_fill(rect_data.get("fill"), rect_data.get("stroke"), xfrm_data))

    if not any(r.get("type") == "rect" and (r.get("fill") or r.get("stroke")) for r in results):
        _apply_layout_rect_fallback(results, ph_type, ph_idx, layout_positions, xfrm_data)

    if spPr is not None:
        custGeom_result = _build_sp_custgeom(spPr, xfrm_data)  # noqa: N806
        if custGeom_result is not None:
            results.append(custGeom_result)
            return results

    if has_txbody:
        text_elem = _parse_text_element(sp, xfrm_data, layout_positions, theme_colors, theme_fonts)
        if text_elem is not None:
            if ph_idx:
                text_elem["placeholder_idx"] = ph_idx
            results.append(text_elem)

    return results


def _parse_layout_background(layout_xml: bytes, layout_path: str, zf: zipfile.ZipFile, width_inches: float,
                             height_inches: float, theme_colors: dict | None = None) -> dict | None:  # noqa: N806
    """Extract background color or image from a slide layout, returning a bg element dict or None."""
    try:
        tree = ET.fromstring(layout_xml)
    except Exception as exp:
        logger.warning(f"_parse_layout_background ET.fromstring failed due to {exp}", exc_info=True)
        return None
    cSld = tree.find(f"{{{P_NS}}}cSld")  # noqa: N806
    if cSld is None:
        return None
    bg = cSld.find(f"{{{P_NS}}}bg")
    if bg is None:
        return None
    bgPr = bg.find(f"{{{P_NS}}}bgPr")  # noqa: N806
    if bgPr is None:
        return None

    blipFill = bgPr.find(f"{{{A_NS}}}blipFill")  # noqa: N806
    if blipFill is not None:
        blip = blipFill.find(f"{{{A_NS}}}blip")
        if blip is not None:
            r_embed = blip.get(f"{{{R_NS}}}embed")
            if r_embed:
                layout_num = "".join(c for c in layout_path if c.isdigit())
                rel_path = f"ppt/slideLayouts/_rels/slideLayout{layout_num}.xml.rels"
                rels = _read_rel(zf, rel_path)
                target = rels.get(r_embed, "")
                if target:
                    if target.startswith("../"):
                        target = target[3:]
                    if not target.startswith("ppt/media/"):
                        target = "ppt/media/" + target.split("/")[-1]
                    try:
                        media_bytes = zf.read(target)
                        b64 = base64.b64encode(media_bytes).decode("ascii")
                        return {
                            "type": "image",
                            "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                            "image_data": b64,
                        }
                    except Exception as exp:
                        logger.warning(f"_parse_layout_background failed to read media: {exp}", exc_info=True)

    solidFill = bgPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
    if solidFill is not None:
        hex_color = _solid_fill_to_hex(solidFill, theme_colors)
        if hex_color:
            return {
                "type": "rect",
                "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                "fill": hex_color,
            }
        scheme = solidFill.find(f"{{{A_NS}}}schemeClr")
        if scheme is not None and theme_colors:
            val = scheme.get("val", "")
            if val in theme_colors:
                return {
                    "type": "rect",
                    "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                    "fill": theme_colors[val],
                }

    return None


def _get_slide_master_background(zf: zipfile.ZipFile, width_inches: float, height_inches: float,
                                 theme_colors: dict | None = None) -> dict | None:  # noqa: N806
    """Extract the slide master's background as a bg element dict, or None."""
    try:
        master_xml = zf.read("ppt/slideMasters/slideMaster1.xml")
        master_tree = ET.fromstring(master_xml)
    except Exception as exp:
        logger.warning(f"_get_slide_master_background failed to read master: {exp}", exc_info=True)
        return None
    cSld = master_tree.find(f"{{{P_NS}}}cSld")  # noqa: N806
    if cSld is None:
        return None
    bg = cSld.find(f"{{{P_NS}}}bg")
    if bg is None:
        return None

    bgPr = bg.find(f"{{{P_NS}}}bgPr")  # noqa: N806
    if bgPr is not None:
        blipFill = bgPr.find(f"{{{A_NS}}}blipFill")  # noqa: N806
        if blipFill is not None:
            blip = blipFill.find(f"{{{A_NS}}}blip")
            if blip is not None:
                r_embed = blip.get(f"{{{R_NS}}}embed")
                if r_embed:
                    rels = _read_rel(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels")
                    target = rels.get(r_embed, "")
                    if target:
                        if target.startswith("../"):
                            target = target[3:]
                        if not target.startswith("ppt/media/"):
                            target = "ppt/media/" + target.split("/")[-1]
                        try:
                            media_bytes = zf.read(target)
                            b64 = base64.b64encode(media_bytes).decode("ascii")
                            return {
                                "type": "image",
                                "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                                "image_data": b64,
                            }
                        except Exception as exp:
                            logger.warning(f"_get_slide_master_background failed to read media: {exp}", exc_info=True)

        solidFill = bgPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
        if solidFill is not None:
            hex_color = _solid_fill_to_hex(solidFill, theme_colors)
            if hex_color:
                return {
                    "type": "rect",
                    "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                    "fill": hex_color,
                }
            scheme = solidFill.find(f"{{{A_NS}}}schemeClr")
            if scheme is not None and theme_colors:
                val = scheme.get("val", "")
                if val in theme_colors:
                    return {
                        "type": "rect",
                        "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                        "fill": theme_colors[val],
                    }

    bgRef = bg.find(f"{{{P_NS}}}bgRef")  # noqa: N806
    if bgRef is not None and theme_colors:
        scheme = bgRef.find(f"{{{A_NS}}}schemeClr")
        if scheme is not None:
            val = scheme.get("val", "")
            if val in theme_colors:
                return {
                    "type": "rect",
                    "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                    "fill": theme_colors[val],
                }

    return None


def _parse_slide_bg(tree: ET.Element, zf: zipfile.ZipFile, theme_colors: dict | None, width_inches: float,
                    height_inches: float) -> dict | None:  # noqa: N806
    """Parse the slide's own <p:bg> element and return a background element dict.

    Handles solidFill (schemeClr or srgbClr), blipFill (image), and bgRef (scheme colour
    reference resolved via the theme color map).
    """
    cSld = tree.find(f"{{{P_NS}}}cSld")  # noqa: N806
    if cSld is None:
        return None
    bg = cSld.find(f"{{{P_NS}}}bg")
    if bg is None:
        return None
    bgPr = bg.find(f"{{{P_NS}}}bgPr")  # noqa: N806
    if bgPr is None:
        return None

    blipFill = bgPr.find(f"{{{A_NS}}}blipFill")  # noqa: N806
    if blipFill is not None:
        blip = blipFill.find(f"{{{A_NS}}}blip")
        if blip is not None:
            r_embed = blip.get(f"{{{R_NS}}}embed")
            if r_embed:
                slide_num = "".join(c for c in ET.QName(cSld.tag).localname or "" if c.isdigit()) # type: ignore[attr-defined]
                if not slide_num:
                    return None
                rel_path = f"ppt/slides/_rels/slide{slide_num}.xml.rels"
                rels = _read_rel(zf, rel_path)
                target = rels.get(r_embed, "")
                if target.startswith("../"):
                    target = target[3:]
                if not target.startswith("ppt/media/"):
                    target = "ppt/media/" + target.split("/")[-1]
                try:
                    media_bytes = zf.read(target)
                    b64 = base64.b64encode(media_bytes).decode("ascii")
                    return {
                        "type": "image",
                        "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                        "image_data": b64,
                    }
                except Exception as exp:
                    logger.warning(f"_parse_slide_bg blipFill failed: {exp}", exc_info=True)
                    return None

    solidFill = bgPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
    if solidFill is not None:
        hex_color = _solid_fill_to_hex(solidFill, theme_colors)
        if hex_color:
            return {
                "type": "rect",
                "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                "fill": hex_color,
            }
        scheme = solidFill.find(f"{{{A_NS}}}schemeClr")
        if scheme is not None and theme_colors:
            val = scheme.get("val", "")
            if val in theme_colors:
                return {
                    "type": "rect",
                    "position": {"x": 0, "y": 0, "width": width_inches, "height": height_inches},
                    "fill": theme_colors[val],
                }

    return None


def _element_key(pos: dict, elem_type: str) -> tuple:
    """Build a dedup key from position dict and element type."""
    return (
        round(pos.get("x", 0), 3),
        round(pos.get("y", 0), 3),
        round(pos.get("width", 0), 3),
        round(pos.get("height", 0), 3),
        elem_type,
    )


def _filter_layout_elements(layout_elements: list, slide_elements: list, slide_placeholder_idxs: set) -> list:
    """Return layout elements that don't overlap with slide elements or placeholder idxs."""
    seen = set()
    result = []
    slide_keys = {_element_key(e.get("position", {}), e.get("type")) for e in slide_elements}
    for elem in layout_elements:
        if elem.get("placeholder_idx") in slide_placeholder_idxs:
            continue
        key = _element_key(elem.get("position", {}), elem.get("type"))
        if key in seen or key in slide_keys:
            continue
        seen.add(key)
        result.append(elem)
    return result


def _extract_gfrm_xfrm(graphic_frame: ET.Element) -> dict | None:
    """Extract position from graphicFrame's p:xfrm child. Returns None on failure."""
    xfrm = graphic_frame.find(f"{{{P_NS}}}xfrm")
    if xfrm is None:
        return None
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    if off is None or ext is None:
        return None
    x_val, y_val, cx_val, cy_val = off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")
    if None in (x_val, y_val, cx_val, cy_val):
        return None
    return {
        "x": _emu_to_inches(int(cast(str, x_val))),
        "y": _emu_to_inches(int(cast(str, y_val))),
        "width": _emu_to_inches(int(cast(str, cx_val))),
        "height": _emu_to_inches(int(cast(str, cy_val))),
    }


def _get_pos_from_layout(ph_type: str, ph_idx: str, layout_positions: dict, require_valid: bool = False) -> dict | None:
    """Look up position from layout_positions by placeholder type/idx.
    
    Args:
        ph_type: placeholder type (e.g. 'body', 'title')
        ph_idx: placeholder idx
        layout_positions: positions dict from layout/master
        require_valid: if True, also verify all position fields are non-None
    """
    for key in [f"{ph_type}:{ph_idx}" if ph_idx else None, f"ph:{ph_type}", f"{ph_type}:"]:
        if key and key in layout_positions:
            pos = layout_positions[key]
            if require_valid:
                if None not in (pos.get("x"), pos.get("y"), pos.get("width"), pos.get("height")):
                    return {"x": pos.get("x"), "y": pos.get("y"), "width": pos.get("width"),
                            "height": pos.get("height")}
            else:
                return {"x": pos.get("x"), "y": pos.get("y"), "width": pos.get("width"), "height": pos.get("height")}
    return None


def _extract_xfrm_from_element(elem: ET.Element, ns_prefix: str) -> dict | None:  # noqa: N806
    """Extract position (EMU→inches) from an element's xfrm/off/ext. Returns None on failure."""
    spPr = elem.find(f"{{{P_NS}}}spPr")  # noqa: N806
    if spPr is not None:
        xfrm = spPr.find(f"{{{A_NS}}}xfrm")
        if xfrm is not None:
            off = xfrm.find(f"{{{A_NS}}}off")
            ext = xfrm.find(f"{{{A_NS}}}ext")
            if off is not None and ext is not None:
                x_val, y_val, cx_val, cy_val = off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")
                if None not in (x_val, y_val, cx_val, cy_val):
                    return {
                        "x": _emu_to_inches(int(cast(str, x_val))),
                        "y": _emu_to_inches(int(cast(str, y_val))),
                        "width": _emu_to_inches(int(cast(str, cx_val))),
                        "height": _emu_to_inches(int(cast(str, cy_val))),
                    }
    if ns_prefix == "pic":
        pic_xfrm = elem.find(f"{{{P_NS}}}xfrm")
        if pic_xfrm is not None:
            off = pic_xfrm.find(f"{{{A_NS}}}off")
            ext = pic_xfrm.find(f"{{{A_NS}}}ext")
            if off is not None and ext is not None:
                x_val, y_val, cx_val, cy_val = off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")
                if None not in (x_val, y_val, cx_val, cy_val):
                    return {
                        "x": _emu_to_inches(int(cast(str, x_val))),
                        "y": _emu_to_inches(int(cast(str, y_val))),
                        "width": _emu_to_inches(int(cast(str, cx_val))),
                        "height": _emu_to_inches(int(cast(str, cy_val))),
                    }
    return None


def _iter_slide_pics(tree: ET.Element, slide_name: str, zf: zipfile.ZipFile, layout_positions: dict) -> list[
    dict]:  # noqa: N806
    """Iterate over p:pic elements in a slide, returning image element dicts."""
    results = []
    pic_idx = 0
    for pic in tree.iter(f"{{{P_NS}}}pic"):
        pic_idx += 1
        xfrm_data = _extract_xfrm_from_element(pic, "pic")

        if xfrm_data is None:
            nvPicPr = pic.find(f"{{{P_NS}}}nvPicPr")  # noqa: N806
            if nvPicPr is not None:
                nvPr = nvPicPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
                if nvPr is not None:
                    ph = nvPr.find(f"{{{P_NS}}}ph")
                    if ph is not None:
                        ph_type = ph.get("type", "")
                        ph_idx = ph.get("idx", "")
                        xfrm_data = _get_pos_from_layout(ph_type, ph_idx, layout_positions)

        if xfrm_data is None:
            logger.warning(f"_iter_slide_pics: pic[{pic_idx}] has no xfrm at {_xpath(pic, slide_name, tree)}, skipping")
            continue

        img = _parse_image(pic, slide_name, xfrm_data, zf)
        if img is not None:
            results.append(img)
    return results


def _add_master_non_placeholder_shapes(  # noqa: N806
        layout_elements_to_add: list,
        slide_taken_positions: set,
        layout_data: str,
        zf: zipfile.ZipFile,
        theme_colors: dict | None,
) -> None:
    """Add master non-placeholder shapes to layout_elements_to_add if not already on slide."""
    show_master = re.search(r'showMasterSp="(\d)"', layout_data)
    show_master_val = show_master.group(1) if show_master else "1"
    preserve_master = "preserve=\"1\"" in layout_data
    if not (show_master_val == "1" and preserve_master):
        return

    try:
        master_non_placeholder = _get_slide_master_non_placeholder_positions(zf, theme_colors)
        master_tree = ET.fromstring(zf.read("ppt/slideMasters/slideMaster1.xml"))
    except Exception as exp:
        logger.warning(f"Failed to add master non-placeholder shapes: {exp}", exc_info=True)
        return

    for mp_key, mp_pos in master_non_placeholder.items():
        mp_x = round(mp_pos.get("x", 0), 3)
        mp_y = round(mp_pos.get("y", 0), 3)
        mp_pos_key = (mp_x, mp_y)
        if mp_pos_key in slide_taken_positions:
            continue
        mp_spPr = None  # noqa: N806
        for sp in master_tree.iter(f"{{{P_NS}}}sp"):
            sp_nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
            if sp_nvSpPr is None:
                continue
            sp_cNvPr = sp_nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
            if sp_cNvPr is None:
                continue
            if sp_cNvPr.get("name", "") == mp_key.replace("master:name=", ""):
                mp_spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
                break
        has_custGeom = mp_spPr is not None and mp_spPr.find(f"{{{A_NS}}}custGeom") is not None  # noqa: N806
        if has_custGeom and mp_spPr is not None:
            spPr_str = ET.tostring(mp_spPr, encoding="unicode")  # noqa: N806
            spPr_str = _fix_up_custgeom_ns(spPr_str)  # noqa: N806
            layout_elements_to_add.append({
                "type": "image",
                "position": {
                    "x": mp_pos.get("x"),
                    "y": mp_pos.get("y"),
                    "width": mp_pos.get("width"),
                    "height": mp_pos.get("height"),
                },
                "shape_data": spPr_str,
            })
        elif mp_pos.get("fill"):
            layout_elements_to_add.append({
                "type": "rect",
                "position": {
                    "x": mp_pos.get("x"),
                    "y": mp_pos.get("y"),
                    "width": mp_pos.get("width"),
                    "height": mp_pos.get("height"),
                },
                "fill": mp_pos["fill"],
            })


def _get_layout_path(zf: zipfile.ZipFile, slide_name: str) -> str | None:
    """Extract the slide layout path from slide relationships, or None if not found."""
    slide_num = "".join(c for c in slide_name if c.isdigit())
    rel_path = f"ppt/slides/_rels/slide{slide_num}.xml.rels"
    try:
        rels_xml = zf.read(rel_path)
        rels_tree = ET.fromstring(rels_xml)
        for rel in rels_tree:
            target = rel.get("Target", "")
            if "slideLayout" in target:
                m = re.search(r"slideLayout(\d+)", target)
                if m:
                    return f"ppt/slideLayouts/slideLayout{m.group(1)}.xml"
    except Exception as exp:
        logger.warning(f"_get_layout_path failed due to {exp}", exc_info=True)
        pass
    return None


def _parse_slide_with_layout(slide_xml: bytes, slide_name: str, zf: zipfile.ZipFile, layout_positions: dict,
                             theme_colors: dict | None = None, theme_fonts: dict | None = None,
                             width_inches: float = 0.0, height_inches: float = 0.0) -> list[dict]:  # noqa: N806
    """Parse a slide's XML and its layout into a list of element dicts (text, rect, image, table, bg)."""
    try:
        tree = ET.fromstring(slide_xml)
    except Exception as exp:
        logger.warning(f"_parse_slide_with_layout ET.fromstring failed due to {exp}", exc_info=True)
        return []

    elements = []

    slide_bg = _parse_slide_bg(tree, zf, theme_colors, width_inches, height_inches)
    if slide_bg is not None:
        elements.append(slide_bg)

    elements.extend(_iter_slide_pics(tree, slide_name, zf, layout_positions))

    gf_idx = 0
    for graphic_frame in tree.iter(f"{{{P_NS}}}graphicFrame"):
        gf_idx += 1
        xfrm_data = _extract_gfrm_xfrm(graphic_frame)
        if xfrm_data is None:
            logger.warning(
                f"_parse_slide_with_layout: graphicFrame[{gf_idx}] has no xfrm at {_xpath(graphic_frame, slide_name, tree)}, skipping")
            continue

        tbl_elem = _parse_table(graphic_frame, xfrm_data)
        if tbl_elem is not None:
            elements.append(tbl_elem)

    cSld = tree.find(f"{{{P_NS}}}cSld")  # noqa: N806
    spTree = cSld.find(f"{{{P_NS}}}spTree") if cSld is not None else None  # noqa: N806

    slide_placeholder_idxs: set[str] = set()
    if spTree is not None:
        for sp in spTree.findall(f"{{{P_NS}}}sp"):
            shapes = _parse_sp_shape(sp, layout_positions, theme_colors, theme_fonts, slide_name, tree)
            for shape in shapes:
                if shape.get("placeholder_idx"):
                    slide_placeholder_idxs.add(shape["placeholder_idx"])
            elements.extend(shapes)

        for grp_sp in spTree.findall(f"{{{P_NS}}}grpSp"):
            grp_results = _parse_group_sp(grp_sp, slide_name, zf, theme_colors)
            elements.extend(grp_results)
    else:
        for sp in tree.iter(f"{{{P_NS}}}sp"):
            shapes = _parse_sp_shape(sp, layout_positions, theme_colors, theme_fonts, slide_name, tree)
            elements.extend(shapes)

    layout_path = _get_layout_path(zf, slide_name)

    layout_elements_to_add: list[dict] = []
    if layout_path and layout_path in zf.namelist():
        layout_xml = zf.read(layout_path)
        layout_elements = _parse_slide_layout(layout_xml, layout_path, zf, theme_colors)

        bg_elem = _parse_layout_background(layout_xml, layout_path, zf, width_inches, height_inches, theme_colors)
        if bg_elem is None:
            layout_tree = ET.fromstring(layout_xml)
            layout_has_bg = (
                    layout_tree.find(f"{{{P_NS}}}cSld/{{{P_NS}}}bg/{{{P_NS}}}bgPr/{{{A_NS}}}blipFill") is not None
                    or layout_tree.find(f"{{{P_NS}}}cSld/{{{P_NS}}}bg/{{{P_NS}}}bgPr/{{{A_NS}}}solidFill") is not None
            )
            if not layout_has_bg:
                bg_elem = _get_slide_master_background(zf, width_inches, height_inches, theme_colors)
        if bg_elem is not None:
            layout_elements.insert(0, bg_elem)

        layout_elements_to_add = _filter_layout_elements(layout_elements, elements, slide_placeholder_idxs)
        slide_taken_positions = {
            (round(e.get("position", {}).get("x", 0), 3),
             round(e.get("position", {}).get("y", 0), 3))
            for e in elements
        }

        if layout_path:
            layout_data_raw = zf.read(layout_path).decode("utf-8", errors="replace")
            _add_master_non_placeholder_shapes(layout_elements_to_add, slide_taken_positions, layout_data_raw, zf,
                                               theme_colors)

    return layout_elements_to_add + elements


def _get_slide_master_non_placeholder_positions(zf: zipfile.ZipFile,
                                                theme_colors: dict | None = None) -> dict:  # noqa: N806
    """Extract non-placeholder shape positions from the slide master."""
    try:
        master_xml = zf.read("ppt/slideMasters/slideMaster1.xml")
        master_tree = ET.fromstring(master_xml)
        positions = {}
        for sp in master_tree.iter(f"{{{P_NS}}}sp"):
            nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
            if nvSpPr is None:
                continue
            cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
            if cNvPr is None:
                continue
            name = cNvPr.get("name", "")
            if cNvPr.get("hidden") == "1":
                continue
            nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
            ph = nvPr.find(f"{{{P_NS}}}ph") if nvPr is not None else None
            if ph is not None:
                continue
            spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
            if spPr is None:
                continue
            xfrm = spPr.find(f"{{{A_NS}}}xfrm")
            if xfrm is None:
                continue
            off = xfrm.find(f"{{{A_NS}}}off")
            ext = xfrm.find(f"{{{A_NS}}}ext")
            if off is None or ext is None:
                continue
            x_val = off.get("x")
            y_val = off.get("y")
            cx_val = ext.get("cx")
            cy_val = ext.get("cy")
            if None in (x_val, y_val, cx_val, cy_val):
                continue
            solid_fill = spPr.find(f"{{{A_NS}}}solidFill")
            fill_color = _solid_fill_to_hex(solid_fill, theme_colors)
            pos_data: dict[str, Any] = {
                "x": _emu_to_inches(int(cast(str, x_val))),
                "y": _emu_to_inches(int(cast(str, y_val))),
                "width": _emu_to_inches(int(cast(str, cx_val))),
                "height": _emu_to_inches(int(cast(str, cy_val))),
            }
            if fill_color:
                pos_data["fill"] = fill_color
            if name:
                positions[f"master:name={name}"] = pos_data
        return positions
    except Exception as exp:
        logger.warning(f"_get_slide_master_non_placeholder_positions failed due to {exp}", exc_info=True)
        return {}


def _extract_txbody_font_attrs(
        txBody: ET.Element,  # noqa: N806
        pos_data: dict,
        ph_type: str,
        theme_fonts: dict | None,
        theme_colors: dict | None,
        txstyles_title_sz: int | None,
        txstyles_title_bold: bool | None,
        txstyles_title_italic: bool | None,
        txstyles_title_algn: str | None,
        txstyles_body_sz: int | None,
        txstyles_body_algn: str | None,
) -> None:
    """Apply txBody lstStyle and txstyles attrs to pos_data for a placeholder."""
    lstStyle = txBody.find(f"{{{A_NS}}}lstStyle")  # noqa: N806
    if lstStyle is not None:
        _apply_lststyle_alignment(lstStyle, pos_data)
        for lvl in lstStyle:
            defRPr = lvl.find(f"{{{A_NS}}}defRPr")  # noqa: N806
            if defRPr is not None:
                _apply_lststyle_to_pos(defRPr, pos_data, theme_fonts, theme_colors)
    _apply_font_size_from_txbody(txBody, pos_data)

    if "font_size" not in pos_data and txstyles_title_sz is not None and ph_type in TITLE_PH_TYPES:
        pos_data["font_size"] = txstyles_title_sz
    if not pos_data.get("bold") and txstyles_title_bold and ph_type in TITLE_PH_TYPES:
        pos_data["bold"] = True
    if not pos_data.get("italic") and txstyles_title_italic and ph_type in TITLE_PH_TYPES:
        pos_data["italic"] = True
    if "align" not in pos_data and txstyles_title_algn and ph_type in TITLE_PH_TYPES:
        pos_data["align"] = txstyles_title_algn
    if "font_size" not in pos_data and txstyles_body_sz and ph_type == "body":
        pos_data["font_size"] = txstyles_body_sz
    if "align" not in pos_data and txstyles_body_algn and ph_type == "body":
        pos_data["align"] = txstyles_body_algn
    if "layout_text" not in pos_data:
        text_parts = []
        for t_elem in txBody.iter(f"{{{A_NS}}}t"):
            if t_elem.text:
                text_parts.append(t_elem.text)
        if text_parts:
            pos_data["layout_text"] = "\n".join(text_parts)


def _get_slide_master_positions(zf: zipfile.ZipFile, theme_colors: dict | None = None,
                                theme_fonts: dict | None = None) -> dict:  # noqa: N806
    """Extract placeholder positions and attributes from the slide master."""
    try:
        master_xml = zf.read("ppt/slideMasters/slideMaster1.xml")
        master_tree = ET.fromstring(master_xml)
        positions = {}
        txstyles = _parse_txstyles_from_master(master_xml)
        txstyles_title_sz = txstyles["title_sz"]
        txstyles_title_bold = txstyles["title_bold"]
        txstyles_title_italic = txstyles["title_italic"]
        txstyles_title_algn = txstyles["title_algn"]
        txstyles_body_sz = txstyles["body_sz"]
        txstyles_body_algn = txstyles["body_algn"]
        for sp in master_tree.iter(f"{{{P_NS}}}sp"):
            nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
            if nvSpPr is None:
                continue
            cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
            if cNvPr is None:
                continue
            nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
            ph = nvPr.find(f"{{{P_NS}}}ph") if nvPr is not None else None
            if ph is None:
                continue
            ph_type = ph.get("type", "")
            ph_idx = ph.get("idx", "")
            spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
            if spPr is not None:
                xfrm = spPr.find(f"{{{A_NS}}}xfrm")
                if xfrm is not None:
                    off = xfrm.find(f"{{{A_NS}}}off")
                    ext = xfrm.find(f"{{{A_NS}}}ext")
                    if off is not None and ext is not None:
                        x_val = off.get("x")
                        y_val = off.get("y")
                        cx_val = ext.get("cx")
                        cy_val = ext.get("cy")
                        if None in (x_val, y_val, cx_val, cy_val):
                            logger.warning(
                                f"_get_slide_master_positions: xfrm missing values at {_xpath(sp, 'ppt/slideMasters/slideMaster1.xml', master_tree)}, skipping placeholder")
                            continue
                        pos_data: dict[str, Any] = {
                            "x": _emu_to_inches(int(cast(str, x_val))),
                            "y": _emu_to_inches(int(cast(str, y_val))),
                            "width": _emu_to_inches(int(cast(str, cx_val))),
                            "height": _emu_to_inches(int(cast(str, cy_val))),
                        }
                        txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
                        if txBody is not None:
                            _extract_txbody_font_attrs(
                                txBody, pos_data, ph_type, theme_fonts, theme_colors,
                                txstyles_title_sz, txstyles_title_bold, txstyles_title_italic,
                                txstyles_title_algn, txstyles_body_sz, txstyles_body_algn,
                            )
                        if not pos_data.get("font_family") and theme_fonts:
                            if ph_type in TITLE_PH_TYPES:
                                pos_data["font_family"] = theme_fonts.get("heading", {}).get("family", "")
                            else:
                                pos_data["font_family"] = theme_fonts.get("body", {}).get("family", "")
                        if ph_type:
                            positions[f"ph:{ph_type}"] = pos_data
                        if ph_idx:
                            positions[f"{ph_type}:{ph_idx}"] = pos_data
                        else:
                            positions[f"{ph_type}:"] = pos_data
        return positions
    except Exception as exp:
        logger.warning(f"_get_slide_master_positions failed due to {exp}", exc_info=True)
        return {}


def _apply_txstyles_fallback(layout_positions: dict, txstyles_title_sz: int | None,
                             txstyles_body_sz: int | None) -> None:
    """Fill in missing font_size from txstyles defaults based on placeholder type."""
    for key, pos_data in layout_positions.items():
        if pos_data.get("font_size") is None:
            ph_key = key.split(":")[0] if ":" in key else key
            if ph_key in TITLE_PH_TYPES and txstyles_title_sz is not None:
                pos_data["font_size"] = txstyles_title_sz
            elif ph_key == "body" and txstyles_body_sz is not None:
                pos_data["font_size"] = txstyles_body_sz


def _collect_layout_text(txBody: ET.Element) -> str | None:  # noqa: N806
    """Extract text from txBody t elements, joined by newlines."""
    parts: list[str] = [cast(str, t.text) for t in txBody.iter(f"{{{A_NS}}}t") if t.text]
    return "\n".join(parts) if parts else None


def _apply_font_size_from_txbody(txBody: ET.Element, pos_data: dict) -> None:  # noqa: N806
    """Extract font_size from the first r/rPr sz in txBody if not already set."""
    if "font_size" not in pos_data:
        first_r = txBody.find(f".//{{{A_NS}}}r")
        if first_r is not None:
            first_rPr = first_r.find(f"{{{A_NS}}}rPr")  # noqa: N806
            if first_rPr is not None:
                sz = first_rPr.get("sz")
                if sz:
                    pos_data["font_size"] = _halfpt_to_pt(int(sz))


def _apply_lststyle_alignment(lstStyle: ET.Element, pos_data: dict) -> None:  # noqa: N806
    """Extract alignment from lstStyle level pPr and lvl algn attributes."""
    for lvl in lstStyle:
        pPr_in_lvl = lvl.find(f"{{{A_NS}}}pPr")  # noqa: N806
        if pPr_in_lvl is not None:
            p_algn = pPr_in_lvl.get("algn")
            if p_algn and "align" not in pos_data:
                pos_data["align"] = ALIGN_MAP.get(p_algn, p_algn)
        lvl_algn = lvl.get("algn")
        if lvl_algn and "align" not in pos_data:
            pos_data["align"] = ALIGN_MAP.get(lvl_algn, lvl_algn)


def _apply_lststyle_to_pos(defRPr: ET.Element, pos_data: dict, theme_fonts: dict | None,  # noqa: N806
                           theme_colors: dict | None) -> None:  # noqa: N806
    """Extract font attrs from a lstStyle defRPr and merge into pos_data, respecting existing values."""
    if "font_family" not in pos_data:
        latin = defRPr.find(f"{{{A_NS}}}latin")
        if latin is not None:
            tf = latin.get("typeface", "")
            if tf:
                if tf in ("+mj-lt", "+mn-lt") and theme_fonts:
                    tf = theme_fonts.get("heading", {}).get("family", "") if tf == "+mj-lt" else theme_fonts.get("body",
                                                                                                                 {}).get(
                        "family", "")
                pos_data["font_family"] = tf
    if "font_size" not in pos_data:
        sz = defRPr.get("sz")
        if sz:
            pos_data["font_size"] = _halfpt_to_pt(int(sz))
    if "bold" not in pos_data:
        b = defRPr.get("b")
        if b and b != "0":
            pos_data["bold"] = True
    if "italic" not in pos_data:
        i = defRPr.get("i")
        if i and i != "0":
            pos_data["italic"] = True
    if "font_color" not in pos_data:
        solidFill = defRPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
        fc = _solid_fill_to_hex(solidFill, theme_colors)
        if fc:
            pos_data["font_color"] = fc


def _merge_master_positions(layout_positions: dict, zf: zipfile.ZipFile, theme_colors: dict | None,
                            theme_fonts: dict | None) -> None:
    """Merge master placeholder and non-placeholder positions into layout_positions."""
    master_positions = _get_slide_master_positions(zf, theme_colors, theme_fonts)
    if master_positions:
        for key, pos_data in master_positions.items():
            if key not in layout_positions:
                layout_positions[key] = pos_data
            else:
                _merge_pos_data(layout_positions[key], pos_data)
    master_non_placeholder = _get_slide_master_non_placeholder_positions(zf, theme_colors)
    if master_non_placeholder:
        for key, pos_data in master_non_placeholder.items():
            if key not in layout_positions:
                layout_positions[key] = pos_data


def _extract_layout_placeholder_positions(  # noqa: N806
        layout_tree: ET.Element,
        layout_path: str,
        theme_fonts: dict | None,
        theme_colors: dict | None,
        txstyles_title_sz: int | None,
        txstyles_body_sz: int | None,
) -> dict:
    """Extract placeholder positions from a layout tree element."""
    layout_positions = {}
    for sp in layout_tree.iter(f"{{{P_NS}}}sp"):
        nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
        if nvSpPr is None:
            continue
        cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
        if cNvPr is None:
            continue
        nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
        ph = nvPr.find(f"{{{P_NS}}}ph") if nvPr is not None else None
        ph_type = ph.get("type", "") if ph is not None else ""
        ph_idx = ph.get("idx", "") if ph is not None else ""
        spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
        if spPr is None:
            continue
        xfrm = spPr.find(f"{{{A_NS}}}xfrm")
        if xfrm is None:
            continue
        off = xfrm.find(f"{{{A_NS}}}off")
        ext = xfrm.find(f"{{{A_NS}}}ext")
        if off is None or ext is None:
            continue
        x_val = off.get("x")
        y_val = off.get("y")
        cx_val = ext.get("cx")
        cy_val = ext.get("cy")
        if None in (x_val, y_val, cx_val, cy_val):
            logger.warning(
                f"_extract_layout_placeholder_positions: xfrm missing values at {_xpath(sp, layout_path, layout_tree)}, skipping placeholder")
            continue
        pos_data: dict[str, Any] = {
            "x": _emu_to_inches(int(cast(str, x_val))),
            "y": _emu_to_inches(int(cast(str, y_val))),
            "width": _emu_to_inches(int(cast(str, cx_val))),
            "height": _emu_to_inches(int(cast(str, cy_val))),
        }
        txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
        if txBody is not None:
            lstStyle = txBody.find(f"{{{A_NS}}}lstStyle")  # noqa: N806
            layout_text = _collect_layout_text(txBody)
            if layout_text:
                pos_data["layout_text"] = layout_text
            if lstStyle is not None:
                _apply_lststyle_alignment(lstStyle, pos_data)
                for lvl in lstStyle:
                    defRPr = lvl.find(f"{{{A_NS}}}defRPr")  # noqa: N806
                    if defRPr is not None:
                        _apply_lststyle_to_pos(defRPr, pos_data, theme_fonts, theme_colors)
            _apply_font_size_from_txbody(txBody, pos_data)
            if "font_size" not in pos_data and txstyles_title_sz is not None and ph_type in TITLE_PH_TYPES:
                pos_data["font_size"] = txstyles_title_sz
            if "font_size" not in pos_data and txstyles_body_sz is not None and ph_type == "body":
                pos_data["font_size"] = txstyles_body_sz
        if not pos_data.get("font_family") and theme_fonts:
            if ph_type in TITLE_PH_TYPES:
                pos_data["font_family"] = theme_fonts.get("heading", {}).get("family", "")
            else:
                pos_data["font_family"] = theme_fonts.get("body", {}).get("family", "")
        if ph_type:
            layout_positions[f"ph:{ph_type}"] = pos_data
        if ph_idx:
            key_type = ph_type if ph_type else "body"
            layout_positions[f"{key_type}:{ph_idx}"] = pos_data
        else:
            layout_positions[f"{ph_type}:"] = pos_data
    return layout_positions


def _get_slide_layout_positions(zf: zipfile.ZipFile, slide_name: str, theme_colors: dict | None = None,
                                theme_fonts: dict | None = None) -> dict:
    """Build a merged positions dict from slide layout + master, for layout fallback."""
    layout_path = _get_layout_path(zf, slide_name)
    if layout_path is None:
        return _get_slide_master_positions(zf, theme_colors, theme_fonts)
    if layout_path not in zf.namelist():
        return _get_slide_master_positions(zf, theme_colors, theme_fonts)
    try:
        layout_xml = zf.read(layout_path)
        layout_tree = ET.fromstring(layout_xml)
    except Exception as exp:
        logger.warning(f"_get_slide_layout_positions failed due to {exp}", exc_info=True)
        return _get_slide_master_positions(zf, theme_colors, theme_fonts)

    try:
        master_xml = zf.read("ppt/slideMasters/slideMaster1.xml")
        txstyles = _parse_txstyles_from_master(master_xml)
        txstyles_body_sz = txstyles["body_sz"]
        txstyles_title_sz = txstyles["title_sz"]
    except Exception as exp:
        logger.warning(f"_parse_txstyles_from_master failed due to {exp}", exc_info=True)
        txstyles_body_sz = None
        txstyles_title_sz = None

    try:
        layout_positions = _extract_layout_placeholder_positions(
            layout_tree, layout_path, theme_fonts, theme_colors, txstyles_title_sz, txstyles_body_sz
        )
        _merge_master_positions(layout_positions, zf, theme_colors, theme_fonts)
        _apply_txstyles_fallback(layout_positions, txstyles_title_sz, txstyles_body_sz)
        return layout_positions
    except Exception as exp:
        logger.warning(f"_get_slide_layout_positions failed (switching to master) due to {exp}", exc_info=True)
        return _get_slide_master_positions(zf, theme_colors, theme_fonts)


def _coerce_to_schema(result: dict) -> dict:
    """Ensure all values in the result are schema-compliant."""

    def _coerce_font(font_entry: dict) -> None:
        if isinstance(font_entry, dict) and font_entry.get("family") is not None:
            if font_entry["family"] not in VALID_FONTS:
                font_entry["family"] = "Arial"

    def _coerce_slot(slot_entry: dict) -> None:
        if isinstance(slot_entry, dict) and slot_entry.get("slot") is not None:
            if slot_entry["slot"] not in VALID_SLOTS:
                del slot_entry["slot"]

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "family" in obj:
                _coerce_font(obj)

            if "slot" in obj:
                _coerce_slot(obj)

            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(result)
    schema = _load_ppt_schema()
    parsed_schema = json.loads(schema)
    result, removed_items = force_to_schema(result, parsed_schema)

    if removed_items:
        for removed in removed_items:
            logger.warning(f"Removed element {removed}")

    if result is None:
        raise Exception("Was unable to produce a valid result")

    is_valid, _ = validate_json_schema(json.dumps(result), schema)
    if not is_valid:
        raise Exception("Inconsistent state detected: Schema validation failure detected")

    logger.info("Presentation file is VALID")

    return cast(dict, result)


def _majority(lst: list[str]) -> str | None:
    """Return the most common element in lst, or None if lst is empty."""
    if not lst:
        return None
    counter = Counter(lst)
    return counter.most_common(1)[0][0]


def _collect_slide_stats(  # noqa: N806
        slide_path: str,
        slide_xml: bytes,
        theme_colors: dict,
) -> tuple[ET.Element | None, str, list[str], list[str], list[str]]:
    """Collect stats (colors, fonts) from a slide. Returns (tree, slide_name, text_colors, bg_colors, font_families)."""
    text_colors = []
    bg_colors = []
    font_families = []
    try:
        tree = ET.fromstring(slide_xml)
    except Exception as exp:
        logger.warning(f"parse_pptx - ET.fromstring failed due to {exp}", exc_info=True)
        return None, "", [], [], []
    cSld = tree.find(f"{{{P_NS}}}cSld")  # noqa: N806
    if cSld is not None:
        slide_bg = cSld.find(f"{{{P_NS}}}bg")
        if slide_bg is not None:
            solid_fill = slide_bg.find(f"{{{A_NS}}}solidFill")
            if solid_fill is not None:
                hex_color = _solid_fill_to_hex(solid_fill, theme_colors)
                if hex_color:
                    bg_colors.append(hex_color)
    for sp in tree.iter(f"{{{P_NS}}}sp"):
        runs_data = _collect_slide_text(sp, theme_colors)
        for (text, font_family, font_size, bold, italic, align, color) in runs_data:
            if font_family:
                font_families.append(font_family)
            if color:
                text_colors.append(color)
        bg_color = _bg_color_from_sp(sp, theme_colors)
        if bg_color:
            bg_colors.append(bg_color)
    slide_name = slide_path.split("/")[-1]
    return tree, slide_name, text_colors, bg_colors, font_families


def parse_pptx(pptx_bytes: bytes) -> str:
    """Parse a PPTX file and return a presentation.json string.

    Args:
        pptx_bytes: Raw PPTX file bytes.

    Returns:
        JSON string conforming to ppt_schema.json.
    """
    with zipfile.ZipFile(BytesIO(pptx_bytes)) as zf:
        width_inches, height_inches = _parse_sld_sz(zf)

        theme_colors, theme_fonts, _ = _read_theme_colors_and_fonts(zf)

        slide_files = []
        for name in zf.namelist():
            m = re.match(r"ppt/slides/slide(\d+)\.xml$", name)
            if m:
                slide_files.append((int(m.group(1)), name))
        slide_files.sort(key=lambda x: x[0])
        slide_names = [name for _, name in slide_files]

        all_text_colors = []
        all_bg_colors = []
        all_font_families = []
        all_table_colors = []

        slide_data_list = []

        for slide_path in slide_names:
            slide_xml = zf.read(slide_path)
            tree, slide_name, text_colors, bg_colors, font_families = _collect_slide_stats(
                slide_path, slide_xml, theme_colors,
            )
            if tree is None:
                continue

            layout_positions = _get_slide_layout_positions(zf, slide_name, theme_colors, theme_fonts)
            elements = _parse_slide_with_layout(slide_xml, slide_name, zf, layout_positions, theme_colors, theme_fonts,
                                                width_inches, height_inches)

            all_text_colors.extend(text_colors)
            all_bg_colors.extend(bg_colors)
            all_font_families.extend(font_families)
            for elem in elements:
                if elem.get("type") == "table" and "table_colors" in elem:
                    all_table_colors.append(elem["table_colors"])
            slide_data_list.append({"path": slide_path, "elements": elements})

        template_colors = {}
        bg = _majority(all_bg_colors)
        if bg:
            template_colors["background"] = bg
        elif "background" in theme_colors:
            template_colors["background"] = theme_colors["background"]

        text_color = _majority(all_text_colors)
        if text_color:
            template_colors["text"] = text_color
        elif "text" in theme_colors:
            template_colors["text"] = theme_colors["text"]

        for key in ("primary", "secondary", "accent"):
            if key in theme_colors and key not in template_colors:
                template_colors[key] = theme_colors[key]

        for key in ("tx1", "tx2", "bg1", "bg2", "accent4", "accent5", "accent6", "hlink", "folHlink"):
            if key in theme_colors and key not in template_colors:
                template_colors[key] = theme_colors[key]

        template_fonts = {}
        if all_font_families:
            counter = Counter(all_font_families)
            most_common = counter.most_common()
            heading_family = most_common[0][0]
            template_fonts["heading"] = {"family": heading_family}
            if len(most_common) > 1:
                body_family = most_common[1][0]
                if body_family != heading_family:
                    template_fonts["body"] = {"family": body_family}
            else:
                template_fonts["body"] = {"family": heading_family}
        elif theme_fonts:
            template_fonts = theme_fonts

        template: dict[str, Any] = {
            "slide_width_inches": width_inches,
            "slide_height_inches": height_inches,
        }
        if template_colors:
            template["colors"] = template_colors
        if template_fonts:
            template["fonts"] = template_fonts

        if all_table_colors:
            header_fills = [tc.get("header_fill") for tc in all_table_colors if tc.get("header_fill")]
            header_font_colors = [tc.get("header_font_color") for tc in all_table_colors if tc.get("header_font_color")]
            cell_fill_alts = [tc.get("cell_fill_alt") for tc in all_table_colors if tc.get("cell_fill_alt")]
            cell_font_colors = [tc.get("cell_font_color") for tc in all_table_colors if tc.get("cell_font_color")]
            table_template = {}
            majority_header_fill = _majority(header_fills)
            if majority_header_fill:
                table_template["header_fill"] = majority_header_fill
            majority_header_font_color = _majority(header_font_colors)
            if majority_header_font_color:
                table_template["header_font_color"] = majority_header_font_color
            majority_cell_fill_alt = _majority(cell_fill_alts)
            if majority_cell_fill_alt:
                table_template["cell_fill_alt"] = majority_cell_fill_alt
            majority_cell_font_color = _majority(cell_font_colors)
            if majority_cell_font_color:
                table_template["cell_font_color"] = majority_cell_font_color
            if table_template:
                template["table"] = table_template

        num_slides = len(slide_data_list)
        slides_out = []
        for idx, slide_info in enumerate(slide_data_list):
            slide_type = _determine_slide_type(idx, num_slides)

            slides_out.append({
                "type": slide_type,
                "elements": slide_info["elements"],
            })

        result = {"template": template, "slides": slides_out}
        result = _coerce_to_schema(result)
        return json.dumps(result, indent=2)


def _extract_lststyle_font_from_sp(sp: ET.Element, ph_type: str, theme_fonts: dict | None, theme_colors: dict | None) -> \
tuple[str, int | None, bool, bool, str | None]:  # noqa: N806
    """Extract lstStyle font attrs (family, size, bold, italic, color) from lvl1pPr/defRPr in sp's txBody."""
    font_family, font_size, bold, italic, font_color = "", None, False, False, None
    txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is None:
        return font_family, font_size, bold, italic, font_color
    lstStyle = txBody.find(f"{{{A_NS}}}lstStyle")  # noqa: N806
    if lstStyle is None:
        return font_family, font_size, bold, italic, font_color
    lvl1pPr = lstStyle.find(f"{{{A_NS}}}lvl1pPr")  # noqa: N806
    if lvl1pPr is None:
        return font_family, font_size, bold, italic, font_color
    defRPr = lvl1pPr.find(f"{{{A_NS}}}defRPr")  # noqa: N806
    if defRPr is None:
        return font_family, font_size, bold, italic, font_color
    latin = defRPr.find(f"{{{A_NS}}}latin")
    typeface = latin.get("typeface", "") if latin is not None else ""
    if typeface in ("+mj-lt", "+mn-lt"):
        if theme_fonts:
            font_family = theme_fonts.get("heading", {}).get("family",
                                                             "") if ph_type in TITLE_PH_TYPES else theme_fonts.get(
                "body", {}).get("family", "")
    elif typeface:
        font_family = typeface
    sz = defRPr.get("sz")
    if sz:
        font_size = _halfpt_to_pt(int(sz))
    if defRPr.get("b") and defRPr.get("b") != "0":
        bold = True
    if defRPr.get("i") and defRPr.get("i") != "0":
        italic = True
    solidFill = defRPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
    font_color = _solid_fill_to_hex(solidFill, theme_colors)
    return font_family, font_size, bold, italic, font_color


def _merge_lststyle_font_into_text_elem(
        text_elem: dict,
        tx_body_info: dict | None,
        txBody: ET.Element | None,  # noqa: N806
        lstStyle_font_family: str,  # noqa: N806
        lstStyle_font_size: int | None,  # noqa: N806
        lstStyle_bold: bool,  # noqa: N806
        lstStyle_italic: bool,  # noqa: N806
        lstStyle_font_color: str | None,  # noqa: N806
) -> None:
    """Merge lstStyle font attrs into text_elem, preferring tx_body_info values."""
    if tx_body_info:
        tx_body_info_font = tx_body_info.get("font", {})
        if not tx_body_info_font.get("family"):
            tx_body_info_font["family"] = lstStyle_font_family
        if "size" not in tx_body_info_font and lstStyle_font_size is not None:
            tx_body_info_font["size"] = lstStyle_font_size
        if not tx_body_info_font.get("bold") and lstStyle_bold:
            tx_body_info_font["bold"] = True
        if not tx_body_info_font.get("italic") and lstStyle_italic:
            tx_body_info_font["italic"] = True
        if not tx_body_info.get("font_color") and lstStyle_font_color:
            tx_body_info["font_color"] = lstStyle_font_color
        text_elem.update(tx_body_info)
    else:
        if txBody is not None:
            first_para = txBody.find(f"{{{A_NS}}}p")
            if first_para is not None:
                para_align = first_para.get("algn", "l")
                text_elem["align"] = ALIGN_MAP.get(para_align, "left")
        if lstStyle_font_family:
            text_elem["font"] = {}
            if lstStyle_font_size is not None:
                text_elem["font"]["size"] = lstStyle_font_size
            if lstStyle_bold:
                text_elem["font"]["bold"] = True
            if lstStyle_italic:
                text_elem["font"]["italic"] = True
            text_elem["font"]["family"] = lstStyle_font_family
        if lstStyle_font_color:
            text_elem["font_color"] = lstStyle_font_color


def _parse_master_sp_shape(sp: ET.Element, theme_colors: dict | None = None, theme_fonts: dict | None = None,
                           master_tree: ET.Element | None = None) -> list[dict]:  # noqa: N806
    """Parse a placeholder shape from the slide master into element dicts, with lstStyle font fallback."""
    results: list[dict] = []

    nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
    if nvSpPr is None:
        return results
    cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
    if cNvPr is None:
        return results

    name = cNvPr.get("name", "")
    if name in LAYOUT_ARTIFACT_NAMES:
        return results

    ph = sp.find(f"{{{P_NS}}}nvSpPr/{{{P_NS}}}nvPr/{{{P_NS}}}ph")
    if ph is None:
        return results

    ph_type = ph.get("type", "")
    ph_idx = ph.get("idx", "")

    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    xfrm_data: dict[str, int | float] | None = None

    if spPr is not None:
        xfrm = spPr.find(f"{{{A_NS}}}xfrm")
        if xfrm is not None:
            off = xfrm.find(f"{{{A_NS}}}off")
            ext = xfrm.find(f"{{{A_NS}}}ext")
            if off is not None and ext is not None:
                x_val = off.get("x")
                y_val = off.get("y")
                cx_val = ext.get("cx")
                cy_val = ext.get("cy")
                if None in (x_val, y_val, cx_val, cy_val):
                    logger.warning(
                        f"_parse_master_sp_shape: xfrm missing values at {_xpath(sp, 'ppt/slideMasters/slideMaster1.xml', master_tree)}, skipping")
                    return results
                xfrm_data = {
                    "x": _emu_to_inches(int(cast(str, x_val))),
                    "y": _emu_to_inches(int(cast(str, y_val))),
                    "width": _emu_to_inches(int(cast(str, cx_val))),
                    "height": _emu_to_inches(int(cast(str, cy_val))),
                }

    if xfrm_data is None:
        logger.warning(
            f"_parse_master_sp_shape: no xfrm at {_xpath(sp, 'ppt/slideMasters/slideMaster1.xml', master_tree)}, skipping")
        return results

    lstStyle_font_family, lstStyle_font_size, lstStyle_bold, lstStyle_italic, lstStyle_font_color = _extract_lststyle_font_from_sp(  # noqa: N806
        sp, ph_type, theme_fonts, theme_colors)

    text_elem: dict[str, Any] = {
        "type": "text",
        "position": xfrm_data,
        "slot": ph_type,
    }
    if ph_idx:
        text_elem["placeholder_idx"] = ph_idx

    txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is not None:
        tx_body_info = _parse_text_element(sp, xfrm_data, {}, theme_colors, theme_fonts)
        _merge_lststyle_font_into_text_elem(
            text_elem, tx_body_info, txBody,
            lstStyle_font_family, lstStyle_font_size, lstStyle_bold, lstStyle_italic, lstStyle_font_color,
        )
    results.append(text_elem)
    return results


def _scale_sp_elem_xfrm(sp: ET.Element, xfrm_data: dict, accumulated_xfrm: dict, ch_xfrm_data: dict, scale_x: float,
                        scale_y: float, parent_xfrm: dict | None) -> dict:  # noqa: N806
    """Extract and scale xfrm from sp child element. Returns accumulated_xfrm if no xfrm found."""
    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    if spPr is None:
        return dict(accumulated_xfrm)
    xfrm = spPr.find(f"{{{A_NS}}}xfrm")
    if xfrm is None:
        return dict(accumulated_xfrm)
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    if off is None:
        return dict(accumulated_xfrm)
    child_x = int(off.get("x", 0))
    child_y = int(off.get("y", 0))
    if parent_xfrm is not None:
        elem_xfrm = {
            "x": accumulated_xfrm["x"] + (child_x - ch_xfrm_data["x"]) * scale_x / EMU_PER_INCH,
            "y": accumulated_xfrm["y"] + (child_y - ch_xfrm_data["y"]) * scale_y / EMU_PER_INCH,
            "width": xfrm_data["width"],
            "height": xfrm_data["height"],
        }
    else:
        elem_xfrm = {
            "x": xfrm_data["x"] + (child_x - ch_xfrm_data["x"]) * scale_x / EMU_PER_INCH,
            "y": xfrm_data["y"] + (child_y - ch_xfrm_data["y"]) * scale_y / EMU_PER_INCH,
            "width": xfrm_data["width"],
            "height": xfrm_data["height"],
        }
    if ext is not None:
        elem_xfrm["width"] = int(ext.get("cx", 0)) * scale_x / EMU_PER_INCH
        elem_xfrm["height"] = int(ext.get("cy", 0)) * scale_y / EMU_PER_INCH
    return elem_xfrm


def _has_text_in_sp(sp: ET.Element) -> bool:  # noqa: N806
    """Check if an sp element has text content in its txBody."""
    txBody = sp.find(f"{{{P_NS}}}txBody")  # noqa: N806
    if txBody is None:
        return False
    for r in txBody.iter(f"{{{A_NS}}}r"):
        t_elem = r.find(f"{{{A_NS}}}t")
        if t_elem is not None and t_elem.text:
            return True
    return False


def _compute_pic_xfrm(
        child: ET.Element,
        xfrm_data: dict,
        accumulated_xfrm: dict,
        ch_xfrm_data: dict,
        scale_x: float,
        scale_y: float,
        parent_xfrm: dict | None,
) -> dict:
    """Extract and scale xfrm for a pic child element, falling back to p:xfrm route."""
    p_spPr = child.find(f"{{{P_NS}}}spPr")  # noqa: N806
    p_xfrm = p_spPr.find(f"{{{A_NS}}}xfrm") if p_spPr is not None else None
    p_off = p_xfrm.find(f"{{{A_NS}}}off") if p_xfrm is not None else None
    p_ext = p_xfrm.find(f"{{{A_NS}}}ext") if p_xfrm is not None else None

    if p_off is None or p_ext is None:
        pic_xfrm = child.find(f"{{{P_NS}}}xfrm")
        if pic_xfrm is not None:
            p_off = pic_xfrm.find(f"{{{A_NS}}}off")
            p_ext = pic_xfrm.find(f"{{{A_NS}}}ext")

    if p_off is not None and p_ext is not None:
        child_x = int(p_off.get("x", 0))
        child_y = int(p_off.get("y", 0))
        child_cx = int(p_ext.get("cx", 0))
        child_cy = int(p_ext.get("cy", 0))

        if parent_xfrm is not None:
            return {
                "x": accumulated_xfrm["x"] + (child_x - ch_xfrm_data["x"]) * scale_x / EMU_PER_INCH,
                "y": accumulated_xfrm["y"] + (child_y - ch_xfrm_data["y"]) * scale_y / EMU_PER_INCH,
                "width": child_cx * scale_x / EMU_PER_INCH,
                "height": child_cy * scale_y / EMU_PER_INCH,
            }
        else:
            return {
                "x": xfrm_data["x"] + (child_x - ch_xfrm_data["x"]) * scale_x / EMU_PER_INCH,
                "y": xfrm_data["y"] + (child_y - ch_xfrm_data["y"]) * scale_y / EMU_PER_INCH,
                "width": child_cx * scale_x / EMU_PER_INCH,
                "height": child_cy * scale_y / EMU_PER_INCH,
            }
    return dict(accumulated_xfrm)


def _apply_grp_fill_to_shape(shape_data: str, grp_fill: str) -> str:
    """Replace grpFill ns0/ns1 placeholders with solidFill using grp_fill color."""
    return (
        shape_data
        .replace("<ns1:grpFill />", f"<ns1:solidFill><ns1:srgbClr val=\"{grp_fill.lstrip('#')}\" /></ns1:solidFill>")
        .replace("<ns0:grpFill />", f"<ns0:solidFill><ns0:srgbClr val=\"{grp_fill.lstrip('#')}\" /></ns0:solidFill>")
    )


def _process_group_sp_shape(  # noqa: N806
        child: ET.Element,
        xfrm_data: dict,
        accumulated_xfrm: dict,
        ch_xfrm_data: dict,
        scale_x: float,
        scale_y: float,
        parent_xfrm: dict | None,
        grp_fill: str | None,
        theme_colors: dict | None,
) -> list[dict]:
    """Process an sp child element within a group, returning a list of result dicts."""
    results = []
    has_text = _has_text_in_sp(child)
    spPr = child.find(f"{{{P_NS}}}spPr")  # noqa: N806
    elem_xfrm = _scale_sp_elem_xfrm(child, xfrm_data, accumulated_xfrm, ch_xfrm_data, scale_x, scale_y, parent_xfrm)
    if spPr is not None:
        if has_text:
            text_elem = _parse_text_element(child, elem_xfrm, {}, theme_colors, None)
            if text_elem is not None:
                results.append(text_elem)
        else:
            shape_data = ET.tostring(spPr, encoding="unicode")
            elem_dict: dict[str, Any] = {
                "type": "image",
                "position": elem_xfrm,
                "shape_data": shape_data,
            }
            has_custGeom = spPr.find(f"{{{A_NS}}}custGeom") is not None  # noqa: N806
            if has_custGeom and grp_fill:
                grpFill = spPr.find(f"{{{A_NS}}}grpFill")  # noqa: N806
                if grpFill is not None and grpFill.find(f"{{{A_NS}}}noFill") is None:
                    elem_dict["shape_data"] = _apply_grp_fill_to_shape(shape_data, grp_fill)
            results.append(elem_dict)
    return results


def _parse_group_sp(grp_sp: ET.Element, slide_name: str, zf: zipfile.ZipFile, theme_colors: dict | None = None,
                    parent_xfrm: dict | None = None) -> list[dict]:  # noqa: N806
    """Recursively parse a p:grpSp (group shape) into element dicts, handling custom geometry and nested shapes."""
    results: list[dict] = []
    nvGrpSpPr = grp_sp.find(f"{{{P_NS}}}nvGrpSpPr")  # noqa: N806
    cNvPr = nvGrpSpPr.find(f"{{{P_NS}}}cNvPr") if nvGrpSpPr is not None else None  # noqa: N806
    if cNvPr is None or cNvPr.get("name", "") in LAYOUT_ARTIFACT_NAMES:
        return results
    grpSpPr = grp_sp.find(f"{{{P_NS}}}grpSpPr")  # noqa: N806
    grp_fill: str | None = None
    if grpSpPr is not None:
        solidFill = grpSpPr.find(f"{{{A_NS}}}solidFill")  # noqa: N806
        if solidFill is not None:
            grp_fill = _solid_fill_to_hex(solidFill, theme_colors)
    xfrm_data: dict[str, int | float] | None = None
    ch_xfrm_data: dict[str, int | float] | None = None

    if grpSpPr is not None:
        xfrm = grpSpPr.find(f"{{{A_NS}}}xfrm")
        if xfrm is not None:
            xfrm_data, ch_xfrm_data = _parse_xfrm_and_chxfrm(xfrm)

    if xfrm_data is None or ch_xfrm_data is None:
        logger.warning(
            f"_parse_master_group_shape: missing xfrm or child xfrm at {_xpath(grp_sp, 'ppt/slideMasters/slideMaster1.xml', _find_root(grp_sp))}, skipping")
        return results

    ch_w_emu = ch_xfrm_data["width"] if ch_xfrm_data["width"] else 1
    ch_h_emu = ch_xfrm_data["height"] if ch_xfrm_data["height"] else 1
    scale_x = xfrm_data["width"] * EMU_PER_INCH / ch_w_emu if ch_w_emu else 1.0
    scale_y = xfrm_data["height"] * EMU_PER_INCH / ch_h_emu if ch_h_emu else 1.0

    if parent_xfrm is not None:
        parent_scale_x = parent_xfrm.get("scale_x", 1.0)
        parent_scale_y = parent_xfrm.get("scale_y", 1.0)
        accumulated_xfrm = {
            "x": parent_xfrm["x"] + (xfrm_data["x"] - parent_xfrm.get("ch_off_x", 0)) * parent_scale_x,
            "y": parent_xfrm["y"] + (xfrm_data["y"] - parent_xfrm.get("ch_off_y", 0)) * parent_scale_y,
            "scale_x": parent_scale_x * scale_x,
            "scale_y": parent_scale_y * scale_y,
            "ch_off_x": ch_xfrm_data["x"] / EMU_PER_INCH,
            "ch_off_y": ch_xfrm_data["y"] / EMU_PER_INCH,
        }
    else:
        accumulated_xfrm = {
            "x": xfrm_data["x"],
            "y": xfrm_data["y"],
            "scale_x": scale_x,
            "scale_y": scale_y,
            "ch_off_x": ch_xfrm_data["x"] / EMU_PER_INCH,
            "ch_off_y": ch_xfrm_data["y"] / EMU_PER_INCH,
        }

    for child in grp_sp:
        if child.tag != f"{{{P_NS}}}pic" and child.tag != f"{{{P_NS}}}sp" and child.tag != f"{{{P_NS}}}grpSp":
            continue

        if child.tag == f"{{{P_NS}}}pic":
            elem_xfrm = _compute_pic_xfrm(
                child, xfrm_data, accumulated_xfrm, ch_xfrm_data, scale_x, scale_y, parent_xfrm
            )
            img = _parse_image(child, slide_name, elem_xfrm, zf)
            if img is not None:
                results.append(img)

        elif child.tag == f"{{{P_NS}}}sp":
            results.extend(_process_group_sp_shape(
                child, xfrm_data, accumulated_xfrm, ch_xfrm_data,
                scale_x, scale_y, parent_xfrm, grp_fill, theme_colors
            ))

        elif child.tag == f"{{{P_NS}}}grpSp":
            for elem in _parse_group_sp(child, slide_name, zf, theme_colors, parent_xfrm=accumulated_xfrm):
                results.append(elem)

    return results


def _extract_spPr_xfrm(elt: ET.Element, pic_idx: int, tree: ET.Element, file_context: str) -> dict | None:  # noqa: N806
    """Try to extract xfrm from p:spPr/a:xfrm. Returns None on failure."""
    spPr = elt.find(f"{{{P_NS}}}spPr")  # noqa: N806
    if spPr is None:
        return None
    xfrm = spPr.find(f"{{{A_NS}}}xfrm")
    if xfrm is None:
        return None
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    if off is None or ext is None:
        return None
    x_val, y_val = off.get("x"), off.get("y")
    cx_val, cy_val = ext.get("cx"), ext.get("cy")
    if None in (x_val, y_val, cx_val, cy_val):
        logger.warning(
            f"_extract_spPr_xfrm: pic[{pic_idx}] xfrm missing values at {_xpath(elt, file_context, tree)}, skipping")
        return None
    return {
        "x": _emu_to_inches(int(cast(str, x_val))),
        "y": _emu_to_inches(int(cast(str, y_val))),
        "width": _emu_to_inches(int(cast(str, cx_val))),
        "height": _emu_to_inches(int(cast(str, cy_val))),
    }


def _extract_pic_xfrm_fallback(elt: ET.Element) -> dict | None:
    """Try to extract xfrm from p:xfrm (fallback route for pics without spPr)."""
    xfrm = elt.find(f"{{{P_NS}}}xfrm")
    if xfrm is None:
        return None
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    if off is None or ext is None:
        return None
    x_val, y_val = off.get("x"), off.get("y")
    cx_val, cy_val = ext.get("cx"), ext.get("cy")
    if None in (x_val, y_val, cx_val, cy_val):
        return None
    return {
        "x": _emu_to_inches(int(cast(str, x_val))),
        "y": _emu_to_inches(int(cast(str, y_val))),
        "width": _emu_to_inches(int(cast(str, cx_val))),
        "height": _emu_to_inches(int(cast(str, cy_val))),
    }


def _parse_user_drawn_sp(sp: ET.Element, theme_colors: dict | None) -> list[dict]:  # noqa: N806
    """Process userDrawn sp elements that are not placeholders."""
    results: list[dict] = []
    nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
    if nvSpPr is None:
        return results
    nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
    if nvPr is None:
        return results
    if nvPr.find(f"{{{P_NS}}}ph") is not None:
        return results
    if nvPr.get("userDrawn") != "1":
        return results
    spPr = sp.find(f"{{{P_NS}}}spPr")  # noqa: N806
    if spPr is None:
        return results
    xfrm = spPr.find(f"{{{A_NS}}}xfrm")
    if xfrm is None:
        return results
    off = xfrm.find(f"{{{A_NS}}}off")
    ext = xfrm.find(f"{{{A_NS}}}ext")
    if off is None or ext is None:
        return results
    x_val = off.get("x")
    y_val = off.get("y")
    cx_val = ext.get("cx")
    cy_val = ext.get("cy")
    if None in (x_val, y_val, cx_val, cy_val):
        return results
    xfrm_data = {
        "x": _emu_to_inches(int(cast(str, x_val))),
        "y": _emu_to_inches(int(cast(str, y_val))),
        "width": _emu_to_inches(int(cast(str, cx_val))),
        "height": _emu_to_inches(int(cast(str, cy_val))),
    }
    if _has_text_in_sp(sp):
        text_elem = _parse_text_element(sp, xfrm_data, {}, theme_colors, None)
        if text_elem is not None:
            results.append(text_elem)
    else:
        shape_data = ET.tostring(spPr, encoding="unicode")
        results.append({
            "type": "image",
            "position": xfrm_data,
            "shape_data": shape_data,
        })
    return results


def _parse_slide_layout(layout_xml: bytes, layout_name: str, zf: zipfile.ZipFile, theme_colors: dict | None = None) -> \
        list[dict]:  # noqa: N806
    """Parse a slide layout's shapes (sp, pic, graphicFrame) into element dicts."""
    try:
        tree = ET.fromstring(layout_xml)
    except Exception as exp:
        logger.warning(f"_parse_slide_layout - ET.fromstring failed due to {exp}", exc_info=True)
        return []

    elements = []
    pic_idx = 0
    for pic in tree.iter(f"{{{P_NS}}}pic"):
        pic_idx += 1
        xfrm_data = _extract_spPr_xfrm(pic, pic_idx, tree, layout_name)
        if xfrm_data is None:
            xfrm_data = _extract_pic_xfrm_fallback(pic)
        if xfrm_data is None:
            logger.warning(
                f"_parse_slide_layout: pic[{pic_idx}] has no xfrm at {_xpath(pic, layout_name, tree)}, skipping")
            continue

        img = _parse_image(pic, layout_name, xfrm_data, zf)
        if img is not None:
            elements.append(img)

    for grp_sp in tree.iter(f"{{{P_NS}}}grpSp"):
        grp_elems = _parse_group_sp(grp_sp, layout_name, zf, theme_colors)
        elements.extend(grp_elems)

    for sp in tree.iter(f"{{{P_NS}}}sp"):
        shapes = _parse_master_sp_shape(sp, theme_colors, None, tree)
        elements.extend(shapes)

    for sp in tree.iter(f"{{{P_NS}}}sp"):
        results = _parse_user_drawn_sp(sp, theme_colors)
        elements.extend(results)

    for sp in tree.iter(f"{{{P_NS}}}sp"):
        nvSpPr = sp.find(f"{{{P_NS}}}nvSpPr")  # noqa: N806
        if nvSpPr is None:
            continue
        nvPr = nvSpPr.find(f"{{{P_NS}}}nvPr")  # noqa: N806
        if nvPr is None:
            continue
        if nvPr.find(f"{{{P_NS}}}ph") is not None:
            continue
        if nvPr.get("userDrawn") == "1":
            continue
        cNvPr = nvSpPr.find(f"{{{P_NS}}}cNvPr")  # noqa: N806
        if cNvPr is not None and cNvPr.get("hidden") == "1":
            continue
        shapes = _parse_sp_shape(sp, {}, theme_colors, None, layout_name, tree)
        elements.extend(shapes)

    return elements
