from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "arquitetura_geral.drawio"


def mx_cell(
    root: ET.Element,
    cell_id: str,
    value: str = "",
    style: str = "",
    parent: str = "1",
    vertex: bool = False,
    edge: bool = False,
    x: float | None = None,
    y: float | None = None,
    w: float | None = None,
    h: float | None = None,
    source: str | None = None,
    target: str | None = None,
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "parent": parent,
        },
    )
    if vertex:
        cell.set("vertex", "1")
    if edge:
        cell.set("edge", "1")
    if source is not None:
        cell.set("source", source)
    if target is not None:
        cell.set("target", target)
    geometry = ET.SubElement(cell, "mxGeometry", {"as": "geometry"})
    if edge:
        geometry.set("relative", "1")
    else:
        if x is not None:
            geometry.set("x", str(x))
        if y is not None:
            geometry.set("y", str(y))
        if w is not None:
            geometry.set("width", str(w))
        if h is not None:
            geometry.set("height", str(h))
    return cell


def main() -> None:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "agent": "Codex",
            "version": "24.7.17",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "arquitetura-geral", "name": "Arquitetura"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "2200",
            "dy": "1320",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "2200",
            "pageHeight": "1320",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    title_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
        "fontSize=30;fontStyle=1;fontColor=#10243b;"
    )
    subtitle_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
        "fontSize=16;fontStyle=1;fontColor=#4f647d;"
    )
    group_stage = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#dcecff;"
        "fontSize=24;fontStyle=1;align=left;verticalAlign=top;spacingTop=14;spacingLeft=16;"
    )
    group_rswa = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#ffe6d6;"
        "fontSize=24;fontStyle=1;align=left;verticalAlign=top;spacingTop=14;spacingLeft=16;"
    )
    box_white = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#ffffff;"
        "fontSize=18;align=center;verticalAlign=middle;"
    )
    box_stage = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#dcecff;"
        "fontSize=18;align=center;verticalAlign=middle;"
    )
    box_fusion = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#e8f7ea;"
        "fontSize=18;align=center;verticalAlign=middle;"
    )
    box_temp = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#efe8ff;"
        "fontSize=18;align=center;verticalAlign=middle;"
    )
    box_head = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeColor=#9fb0c7;fillColor=#fff5d6;"
        "fontSize=18;align=center;verticalAlign=middle;"
    )
    edge_style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#10243b;strokeWidth=2;endArrow=classic;endFill=1;"
    )

    mx_cell(root, "title", "Arquitetura Geral Atual do Modelo Conjunto", title_style, x=470, y=20, w=1260, h=50, vertex=True)
    mx_cell(root, "subtitle", "CNN + BiMamba • topologia separate • branch 01_teste_30seg", subtitle_style, x=620, y=75, w=960, h=30, vertex=True)

    mx_cell(root, "stage_group", "Ramo de Estagiamento (RE)", group_stage, x=40, y=140, w=1020, h=640, vertex=True)
    mx_cell(root, "rswa_group", "Ramo de Detecção de RSWA (RDR)", group_rswa, x=1320, y=140, w=840, h=770, vertex=True)

    vertices = [
        ("eeg", "EEG branch<br>3 canais<br>k = 30, 70, 150", box_stage, 90, 240, 340, 90),
        ("eog", "EOG branch<br>1 canal<br>k = 50, 150, 250", box_stage, 460, 240, 340, 90),
        ("concat_stage", "Concat<br>128 canais", box_white, 830, 240, 180, 90),
        ("se_stage", "SEBlock 128 -&gt; 16 -&gt; 128", box_white, 210, 390, 690, 84),
        ("conv5_stage", "Conv1d k=5 + GN + ReLU + Pool", box_white, 210, 505, 690, 84),
        ("conv3_stage", "Conv1d k=3 + GN + ReLU", box_white, 210, 620, 690, 84),
        ("pool_stage", "AdaptiveAvgPool1d(1)<br>staging_features [B, T_stage, 256]", box_white, 210, 735, 690, 54),
        ("bimamba_stage", "BiMamba do staging<br>1 bloco bidirecional", box_temp, 160, 820, 790, 78),
        ("head_stage", "Head de staging<br>staging_logits [B, T_stage, 5]", box_head, 160, 923, 790, 78),
        ("softmax_stage", "Softmax -&gt; stage_probs [B, T_stage, 5]<br>Alinhamento temporal, se necessário.", box_fusion, 160, 1030, 790, 88),
        ("emg_inputs", "Canais possíveis do EMG:<br>1) z-score<br>2) EMG / basal<br>3) amplitude relativa<br>4) opcional: RMS relativo", box_white, 1370, 240, 740, 105),
        ("emg_cnn", "Encoder CNN multiescala<br>3 paths EMG<br>k = 50, 150, 300<br>Conv1d + GN + ReLU + Pool", box_white, 1370, 390, 740, 105),
        ("emg_embed", "Concat -&gt; Dropout -&gt; SEBlock<br>Conv1d k=5 + GN + ReLU + Pool<br>Conv1d k=3 + GN + ReLU<br>emg_cnn_embedding [B, T_rswa, 256]", box_white, 1370, 525, 740, 120),
        ("emg_local", "Ramo local opcional de sub-janelas<br>subwindows de emg_subwindow_ms<br>features: RMS, MAV, STD e amplitude média<br>emg_local_embedding [B, T_rswa, 64]", box_white, 1370, 680, 740, 120),
        ("local_fusion", "Fusão local opcional<br>Concat [256 + 64] -&gt; Linear + LN + ReLU + Dropout<br>rswa_features [B, T_rswa, 256]", box_fusion, 1370, 835, 740, 70),
        ("stage_out", "Saída do<br>estagiamento", box_fusion, 790, 1022, 260, 96),
        ("fusion", "Módulo de fusão<br>Concat [rswa_features,<br>stage_probs] -&gt; Linear 261-&gt;256<br>LN + ReLU + Dropout", box_fusion, 1180, 1015, 320, 110),
        ("bimamba_rswa", "BiMamba do RSWA<br>1 bloco bidirecional", box_temp, 1550, 1015, 610, 82),
        ("heads_rswa", "Heads por mini-época<br>tonic • phasic • any", box_head, 1550, 1120, 610, 70),
    ]

    for cell_id, value, style, x, y, w, h in vertices:
        mx_cell(root, cell_id, value, style, x=x, y=y, w=w, h=h, vertex=True)

    edges = [
        ("e1", "eeg", "se_stage"),
        ("e2", "eog", "se_stage"),
        ("e3", "concat_stage", "se_stage"),
        ("e4", "se_stage", "conv5_stage"),
        ("e5", "conv5_stage", "conv3_stage"),
        ("e6", "conv3_stage", "pool_stage"),
        ("e7", "pool_stage", "bimamba_stage"),
        ("e8", "bimamba_stage", "head_stage"),
        ("e9", "head_stage", "softmax_stage"),
        ("e10", "emg_inputs", "emg_cnn"),
        ("e11", "emg_cnn", "emg_embed"),
        ("e12", "emg_embed", "emg_local"),
        ("e13", "emg_local", "local_fusion"),
        ("e14", "softmax_stage", "stage_out"),
        ("e15", "stage_out", "fusion"),
        ("e16", "local_fusion", "fusion"),
        ("e17", "fusion", "bimamba_rswa"),
        ("e18", "bimamba_rswa", "heads_rswa"),
    ]
    for edge_id, source, target in edges:
        mx_cell(root, edge_id, style=edge_style, edge=True, source=source, target=target)

    ET.indent(mxfile, space="  ")
    OUT.write_text(ET.tostring(mxfile, encoding="unicode"), encoding="utf-8")


if __name__ == "__main__":
    main()
