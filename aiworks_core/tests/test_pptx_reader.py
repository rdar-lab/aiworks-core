"""Tests for pptx_reader — PPTX reverse parser (parse_pptx and parse_pptx_slide_master)."""

import base64
import io
import json
import zipfile

from django.test import TestCase
from ..logic.pptx_generator import render_presentation_pptx
from ..logic.pptx_reader import parse_pptx


def _make_pptx(slides_xml: list[str], presentation_xml: str = "", slide_width: int = 12192000, slide_height: int = 6858000) -> bytes:
    """Build a minimal PPTX bytes from a list of slide XML strings."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>')
        zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        pres = presentation_xml or f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<p:sldSz cx="{slide_width}" cy="{slide_height}"/>
<p:sldIdLst/>'''
        zf.writestr("ppt/presentation.xml", pres)
        for i, sld_xml in enumerate(slides_xml, 1):
            zf.writestr(f"ppt/slides/slide{i}.xml", sld_xml)
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
        zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
    return buf.getvalue()


def _make_img_png() -> bytes:
    """Return minimal valid PNG bytes (1x1 red pixel)."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class ParsePptxTests(TestCase):
    def test_one_slide_is_title(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="2400" b="1"/><a:t>My Title</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(parsed["slides"][0]["type"], "title")

    def test_three_slides_title_content_closing(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"/><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld, sld, sld]))
        parsed = json.loads(result)
        self.assertEqual(parsed["slides"][0]["type"], "title")
        self.assertEqual(parsed["slides"][1]["type"], "content")
        self.assertEqual(parsed["slides"][2]["type"], "closing")

    def test_text_element_parsed(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="TextBox"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="457200" y="914400"/><a:ext cx="4572000" cy="2286000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/>
<a:p><a:r><a:rPr lang="en-US" sz="1800" b="1" i="1"><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t>Hello World</a:t></a:r></a:p>
</p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        elem = parsed["slides"][0]["elements"][0]
        self.assertEqual(elem["type"], "text")
        self.assertEqual(elem["text"], "Hello World")
        self.assertEqual(elem["font"]["family"], "Arial")
        self.assertEqual(elem["font"]["size"], 18)
        self.assertTrue(elem["font"]["bold"])
        self.assertTrue(elem["font"]["italic"])
        self.assertEqual(elem["font_color"], "#FF0000")

    def test_rect_element_parsed(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Rectangle"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="00FF00"/></a:solidFill>
<a:ln w="25400"><a:solidFill><a:srgbClr val="0000FF"/></a:solidFill></a:ln>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        elem = parsed["slides"][0]["elements"][0]
        self.assertEqual(elem["type"], "rect")
        self.assertEqual(elem["fill"], "#00FF00")
        self.assertEqual(elem["stroke"], "#0000FF")

    def test_rect_scheme_clr_with_luminance_modifiers(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slidemaster+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Rectangle"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/>
<a:solidFill><a:schemeClr val="accent2"><a:lumMod val="40000"/><a:lumOff val="60000"/></a:schemeClr></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>''')
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
            zf.writestr("ppt/theme/theme1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="TestTheme">
<a:themeElements><a:clrScheme name="Test">
<a:dk1><a:srgbClr val="3D4647"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="C8C9C7"/></a:dk2><a:lt2><a:srgbClr val="40AA1D"/></a:lt2>
<a:accent1><a:srgbClr val="40AA1D"/></a:accent1>
<a:accent2><a:srgbClr val="617480"/></a:accent2>
<a:accent3><a:srgbClr val="26481F"/></a:accent3>
<a:accent4><a:srgbClr val="197BC0"/></a:accent4>
<a:accent5><a:srgbClr val="EB6D00"/></a:accent5>
<a:accent6><a:srgbClr val="1D496E"/></a:accent6>
<a:hlink><a:srgbClr val="197BC0"/></a:hlink><a:folHlink><a:srgbClr val="197BC0"/></a:folHlink>
</a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"/></a:themeElements></a:theme>''')
            zf.writestr("ppt/slideMasters/slideMaster1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr></p:sp></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/></p:clrMap></p:sldMaster>')
            zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        elem = parsed["slides"][0]["elements"][0]
        self.assertEqual(elem["type"], "rect")
        self.assertEqual(elem["fill"], "#BFC7CC")

    def test_image_data_roundtrip(self):
        img_bytes = _make_img_png()
        _b64 = base64.b64encode(img_bytes).decode("ascii")

        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<p:cSld><p:spTree>
<p:pic><p:nvPicPr><p:cNvPr id="1" name="Picture"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="914400"/></a:xfrm></p:spPr>
</p:pic>
</p:spTree></p:cSld></p:sld>'''
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", sld)
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/></Relationships>')
            zf.writestr("ppt/media/image1.png", img_bytes)
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')

        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        elem = parsed["slides"][0]["elements"][0]
        self.assertEqual(elem["type"], "image")
        self.assertIn("image_data", elem)
        decoded = base64.b64decode(elem["image_data"])
        self.assertEqual(decoded, img_bytes)

    def test_no_sld_sz_defaults(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>X</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        self.assertEqual(parsed["template"]["slide_width_inches"], 13.33)
        self.assertEqual(parsed["template"]["slide_height_inches"], 7.5)

    def test_color_majority_vote(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:bg><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></p:bg><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"><a:solidFill><a:srgbClr val="000000"/></a:solidFill></a:rPr><a:t>Text</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld, sld, sld, sld]))
        parsed = json.loads(result)
        self.assertEqual(parsed["template"]["colors"]["background"], "#FFFFFF")
        self.assertEqual(parsed["template"]["colors"]["text"], "#000000")

    def test_font_majority_vote(self):
        def make_sld(fonts: list[str]) -> str:
            runs = "".join(f'<a:r><a:rPr lang="en-US" sz="1800"><a:latin typeface="{f}"/></a:rPr><a:t>T</a:t></a:r>' for f in fonts)
            return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p>{runs}</a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        slides = [
            make_sld(["Arial", "Arial", "Times New Roman"]),
            make_sld(["Arial", "Arial", "Arial"]),
            make_sld(["Times New Roman", "Times New Roman"]),
        ]
        result = parse_pptx(_make_pptx(slides))
        parsed = json.loads(result)
        self.assertEqual(parsed["template"]["fonts"]["heading"]["family"], "Arial")
        self.assertEqual(parsed["template"]["fonts"]["body"]["family"], "Times New Roman")

    def test_skips_layout_artifact_shapes(self):
        for name in ["Title", "Subtitle", "Date Placeholder", "Footer Placeholder", "Slide Number Placeholder"]:
            sld = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Should not appear</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
            result = parse_pptx(_make_pptx([sld]))
            parsed = json.loads(result)
            self.assertEqual(len(parsed["slides"][0]["elements"]), 0, f"Shape '{name}' should have been skipped")

    def test_no_text_runs_no_fonts_key(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertNotIn("fonts", parsed["template"])

    def test_ignores_slide_master_and_layouts(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="Shape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Real content</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
            zf.writestr("ppt/slideMasters/slideMaster1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="2" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="4400" b="1"><a:t>MASTER TITLE</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sldMaster>')
            zf.writestr("ppt/slideLayouts/slideLayout1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="title" preserve="1"><p:cSld name="Title Slide"><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="3" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="685800" y="2130425"/><a:ext cx="7772400" cy="1470025"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>LAYOUT TITLE</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["text"], "Real content")


class GroupShapeTests(TestCase):
    def test_group_shape_with_child_text_element(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:grpSp>
<p:nvGrpSpPr><p:cNvPr id="1" name="Group 1"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr>
<a:xfrm>
<a:off x="457200" y="457200"/>
<a:ext cx="2286000" cy="1143000"/>
<a:chOff x="0" y="0"/>
<a:chExt cx="2286000" cy="1143000"/>
</a:xfrm>
</p:grpSpPr>
<p:sp>
<p:nvSpPr><p:cNvPr id="2" name="TextInGroup"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"/><a:t>Grouped Text</a:t></a:r></a:p></p:txBody>
</p:sp>
</p:grpSp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "text")
        self.assertEqual(parsed["slides"][0]["elements"][0]["text"], "Grouped Text")

    def test_group_shape_with_child_image(self):
        img_bytes = _make_img_png()
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<p:cSld><p:spTree>
<p:grpSp>
<p:nvGrpSpPr><p:cNvPr id="1" name="Group 2"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr>
<a:xfrm>
<a:off x="0" y="0"/>
<a:ext cx="1828800" cy="1371600"/>
<a:chOff x="0" y="0"/>
<a:chExt cx="1828800" cy="1371600"/>
</a:xfrm>
</p:grpSpPr>
<p:pic>
<p:nvPicPr><p:cNvPr id="3" name="Picture in Group"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="914400"/></a:xfrm></p:spPr>
</p:pic>
</p:grpSp>
</p:spTree></p:cSld></p:sld>'''
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", sld)
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/></Relationships>')
            zf.writestr("ppt/media/image1.png", img_bytes)
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 2)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "image")

    def test_group_shape_layout_artifact_skipped(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:grpSp>
<p:nvGrpSpPr><p:cNvPr id="1" name="Title"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr>
<a:xfrm>
<a:off x="0" y="0"/>
<a:ext cx="12192000" cy="6858000"/>
<a:chOff x="0" y="0"/>
<a:chExt cx="12192000" cy="6858000"/>
</a:xfrm>
</p:grpSpPr>
<p:sp>
<p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="2400"/><a:t>Should not appear</a:t></a:r></a:p></p:txBody>
</p:sp>
</p:grpSp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 0)

    def test_nested_group_shapes(self):
        inner_sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:grpSp>
<p:nvGrpSpPr><p:cNvPr id="1" name="Outer Group"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr>
<a:xfrm>
<a:off x="0" y="0"/>
<a:ext cx="2286000" cy="1828800"/>
<a:chOff x="0" y="0"/>
<a:chExt cx="2286000" cy="1828800"/>
</a:xfrm>
</p:grpSpPr>
<p:grpSp>
<p:nvGrpSpPr><p:cNvPr id="2" name="Inner Group"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr>
<a:xfrm>
<a:off x="0" y="0"/>
<a:ext cx="914400" cy="914400"/>
<a:chOff x="0" y="0"/>
<a:chExt cx="914400" cy="914400"/>
</a:xfrm>
</p:grpSpPr>
<p:sp>
<p:nvSpPr><p:cNvPr id="3" name="NestedText"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="457200" cy="228600"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Nested</a:t></a:r></a:p></p:txBody>
</p:sp>
</p:grpSp>
</p:grpSp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([inner_sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)


class UserDrawnShapeTests(TestCase):
    def test_user_drawn_text_shape_parsed(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="UserShape"/><p:cNvSpPr/><p:nvPr><p:ph/>userDrawn="1"</p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="457200" y="914400"/><a:ext cx="4572000" cy="2286000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="1800"/><a:t>User Drawn Text</a:t></a:r></a:p></p:txBody>
</p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "text")
        self.assertEqual(parsed["slides"][0]["elements"][0]["text"], "User Drawn Text")

    def test_user_drawn_non_text_shape_becomes_image(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="UserRect"/><p:cNvSpPr/><p:nvPr userDrawn="1"/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>
</p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "rect")
        self.assertEqual(parsed["slides"][0]["elements"][0]["fill"], "#FF0000")

    def test_user_drawn_flag_zero_skipped(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="NotUserDrawn"/><p:cNvSpPr/><p:nvPr userDrawn="0"/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="00FF00"/></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>
</p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "rect")
        self.assertEqual(parsed["slides"][0]["elements"][0]["fill"], "#00FF00")

    def test_user_drawn_absent_is_skipped(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="RegularShape"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="0000FF"/></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>
</p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "rect")
        self.assertEqual(parsed["slides"][0]["elements"][0]["fill"], "#0000FF")

    def test_user_drawn_placeholder_skipped(self):
        sld = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="Placeholder"/><p:cNvSpPr/><p:nvPr userDrawn="1"><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="914400" cy="457200"/></a:xfrm>
<a:prstGeom prst="rect"/><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>
</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>
</p:sp>
</p:spTree></p:cSld></p:sld>'''
        result = parse_pptx(_make_pptx([sld]))
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"][0]["elements"]), 1)
        self.assertEqual(parsed["slides"][0]["elements"][0]["type"], "rect")


class LayoutFallbackTests(TestCase):
    def test_layout_placeholder_fallback_to_master(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slidemaster+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="457200" y="274638"/><a:ext cx="8229600" cy="1143000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="4400" b="1"/><a:t>My Title</a:t></a:r></a:p></p:txBody></p:sp>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Content"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="457200" y="1600200"/><a:ext cx="8229600" cy="4532400"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Content text</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>''')
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
            zf.writestr("ppt/slideLayouts/slideLayout1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="title" preserve="1">
<p:cSld name="Title">
<p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="685800" y="2130425"/><a:ext cx="7772400" cy="1470025"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="3200" b="1"/><a:t>Layout Title</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree>
</p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>''')
            zf.writestr("ppt/slideMasters/slideMaster1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="4400" b="1"><a:latin typeface="Times New Roman"/></a:rPr><a:t>MASTER TITLE</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:txStyles>
<p:txStyle name="title">
<a:defRPr sz="4400" b="1"><a:latin typeface="Arial"/></a:defRPr>
</p:txStyle>
</p:txStyles>
</p:sldMaster>''')
            zf.writestr("ppt/theme/theme1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="TestTheme">
<a:themeElements><a:clrScheme name="Test">
<a:dk1><a:srgbClr val="3D4647"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="C8C9C7"/></a:dk2><a:lt2><a:srgbClr val="40AA1D"/></a:lt2>
<a:accent1><a:srgbClr val="40AA1D"/></a:accent1>
<a:accent2><a:srgbClr val="617480"/></a:accent2>
<a:accent3><a:srgbClr val="26481F"/></a:accent3>
<a:accent4><a:srgbClr val="197BC0"/></a:accent4>
<a:accent5><a:srgbClr val="EB6D00"/></a:accent5>
<a:accent6><a:srgbClr val="1D496E"/></a:accent6>
<a:hlink><a:srgbClr val="197BC0"/></a:hlink><a:folHlink><a:srgbClr val="197BC0"/></a:folHlink>
</a:clrScheme>
<a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Office"/></a:themeElements></a:theme>''')
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"]), 1)
        self.assertEqual(parsed["slides"][0]["type"], "title")
        texts = [e.get("text") for e in parsed["slides"][0]["elements"] if e.get("type") == "text" and e.get("text")]
        self.assertIn("My Title", texts)
        self.assertIn("Content text", texts)

    def test_slide_with_no_layout_falls_back_to_master(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slidemaster+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
            zf.writestr("ppt/presentation.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst/></p:presentation>')
            zf.writestr("ppt/slides/slide1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title 1"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="457200" y="274638"/><a:ext cx="8229600" cy="1143000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="4400" b="1"/><a:t>Fallback Title</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>''')
            zf.writestr("ppt/slides/_rels/slide1.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
            zf.writestr("ppt/slideMasters/slideMaster1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree>
<p:sp><p:nvSpPr><p:cNvPr id="1" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="4400" b="1"><a:latin typeface="Georgia"/></a:rPr><a:t>MASTER</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
</p:sldMaster>''')
            zf.writestr("ppt/theme/theme1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="TestTheme">
<a:themeElements><a:clrScheme name="Test">
<a:dk1><a:srgbClr val="3D4647"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="C8C9C7"/></a:dk2><a:lt2><a:srgbClr val="40AA1D"/></a:lt2>
<a:accent1><a:srgbClr val="40AA1D"/></a:accent1>
<a:accent2><a:srgbClr val="617480"/></a:accent2>
<a:accent3><a:srgbClr val="26481F"/></a:accent3>
<a:accent4><a:srgbClr val="197BC0"/></a:accent4>
<a:accent5><a:srgbClr val="EB6D00"/></a:accent5>
<a:accent6><a:srgbClr val="1D496E"/></a:accent6>
<a:hlink><a:srgbClr val="197BC0"/></a:hlink><a:folHlink><a:srgbClr val="197BC0"/></a:folHlink>
</a:clrScheme>
<a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Office"/></a:themeElements></a:theme>''')
            zf.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>')
        result = parse_pptx(buf.getvalue())
        parsed = json.loads(result)
        self.assertEqual(len(parsed["slides"]), 1)
        self.assertEqual(parsed["slides"][0]["type"], "title")
        self.assertEqual(parsed["slides"][0]["elements"][0]["text"], "Fallback Title")


class ImageDataRoundtripViaGenerator(TestCase):
    def test_image_data_roundtrip_render(self):
        img_bytes = _make_img_png()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        spec = {
            "template": {
                "slide_width_inches": 13.33,
                "slide_height_inches": 7.5,
                "colors": {"primary": "#1a1a2e", "secondary": "#e94560", "background": "#ffffff", "text": "#333333"},
                "fonts": {"heading": {"family": "Arial", "size": 18}, "body": {"family": "Arial", "size": 12}},
            },
            "slides": [
                {
                    "type": "title",
                    "elements": [
                        {
                            "type": "image",
                            "position": {"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0},
                            "image_data": b64,
                        },
                    ],
                },
            ],
        }
        pptx_bytes = render_presentation_pptx(spec, {})
        self.assertGreater(len(pptx_bytes), 100)
        zf = zipfile.ZipFile(io.BytesIO(pptx_bytes))
        self.assertIn("ppt/slides/slide1.xml", zf.namelist())