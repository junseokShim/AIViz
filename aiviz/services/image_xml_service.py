"""
XML save/load service for AIViz image analysis sessions.

Schema:
  <AIVizImageAnalysis version="0.2.0">
    <Metadata>
      <ImagePath>…</ImagePath>
      <ImageSize>WxH</ImageSize>
      <ImageMode>RGB</ImageMode>
      <CreatedAt>ISO-8601</CreatedAt>
      <AIVizVersion>0.2.0</AIVizVersion>
    </Metadata>
    <Preprocessing>
      <brightness>1.0</brightness>
      …
    </Preprocessing>
    <EdgeDetection method="canny">
      <low_threshold>50</low_threshold>
      <high_threshold>150</high_threshold>
    </EdgeDetection>
    <Segmentation method="threshold">
      <threshold>128</threshold>
    </Segmentation>
    <ChannelStats>
      <Channel channel="R" mean="127.5" std="45.2" min="0" max="255"/>
      …
    </ChannelStats>
  </AIVizImageAnalysis>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImageSessionData:
    """All analysis context that can be persisted for an image session."""
    image_path: str
    image_size: tuple[int, int]        # (width, height)
    image_mode: str
    preprocess_params: dict            # from PreprocessParams.to_dict()
    edge_method: str
    edge_params: dict
    segment_method: str
    segment_params: dict
    channel_stats: list[dict]          # [{"channel": "R", "mean": …}, …]
    created_at: str = ""
    aiviz_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_xml(session: ImageSessionData, output_path: str) -> None:
    """Serialise *session* to an XML file at *output_path*."""
    root = ET.Element("AIVizImageAnalysis", version=session.aiviz_version)

    # Metadata
    meta = ET.SubElement(root, "Metadata")
    ET.SubElement(meta, "ImagePath").text = session.image_path
    ET.SubElement(meta, "ImageSize").text = (
        f"{session.image_size[0]}x{session.image_size[1]}"
    )
    ET.SubElement(meta, "ImageMode").text = session.image_mode
    ET.SubElement(meta, "CreatedAt").text = session.created_at
    ET.SubElement(meta, "AIVizVersion").text = session.aiviz_version

    # Preprocessing
    preproc = ET.SubElement(root, "Preprocessing")
    for k, v in session.preprocess_params.items():
        ET.SubElement(preproc, str(k)).text = str(v)

    # Edge detection
    edge = ET.SubElement(root, "EdgeDetection")
    edge.set("method", session.edge_method)
    for k, v in session.edge_params.items():
        ET.SubElement(edge, str(k)).text = str(v)

    # Segmentation
    seg = ET.SubElement(root, "Segmentation")
    seg.set("method", session.segment_method)
    for k, v in session.segment_params.items():
        ET.SubElement(seg, str(k)).text = str(v)

    # Channel statistics
    stats_el = ET.SubElement(root, "ChannelStats")
    for row in session.channel_stats:
        ch_el = ET.SubElement(stats_el, "Channel")
        for k, v in row.items():
            ch_el.set(str(k), str(v))

    # Pretty-print (Python 3.9+)
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass  # Python < 3.9

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def load_xml(xml_path: str) -> Optional[ImageSessionData]:
    """
    Parse *xml_path* and return an ImageSessionData.

    Returns None if parsing fails or file is not a valid AIViz XML.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.tag != "AIVizImageAnalysis":
            return None

        # Metadata
        meta = root.find("Metadata")
        image_path = (meta.findtext("ImagePath") or "") if meta is not None else ""
        size_str = (meta.findtext("ImageSize") or "0x0") if meta is not None else "0x0"
        try:
            w, h = [int(x) for x in size_str.split("x")]
        except Exception:
            w, h = 0, 0
        image_mode = (meta.findtext("ImageMode") or "RGB") if meta is not None else "RGB"
        created_at = (meta.findtext("CreatedAt") or "") if meta is not None else ""
        version = root.get("version", SCHEMA_VERSION)

        # Preprocessing
        preproc_el = root.find("Preprocessing")
        preprocess_params: dict = {}
        if preproc_el is not None:
            for child in preproc_el:
                preprocess_params[child.tag] = child.text or ""

        # Edge detection
        edge_el = root.find("EdgeDetection")
        edge_method = (edge_el.get("method") or "canny") if edge_el is not None else "canny"
        edge_params: dict = {}
        if edge_el is not None:
            for child in edge_el:
                edge_params[child.tag] = child.text or ""

        # Segmentation
        seg_el = root.find("Segmentation")
        segment_method = (
            seg_el.get("method") or "threshold"
        ) if seg_el is not None else "threshold"
        segment_params: dict = {}
        if seg_el is not None:
            for child in seg_el:
                segment_params[child.tag] = child.text or ""

        # Channel stats
        stats_el = root.find("ChannelStats")
        channel_stats: list[dict] = []
        if stats_el is not None:
            for ch in stats_el.findall("Channel"):
                channel_stats.append(dict(ch.attrib))

        return ImageSessionData(
            image_path=image_path,
            image_size=(w, h),
            image_mode=image_mode,
            preprocess_params=preprocess_params,
            edge_method=edge_method,
            edge_params=edge_params,
            segment_method=segment_method,
            segment_params=segment_params,
            channel_stats=channel_stats,
            created_at=created_at,
            aiviz_version=version,
        )

    except Exception:
        return None
