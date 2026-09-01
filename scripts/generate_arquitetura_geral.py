from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "arquitetura_geral_v2.png"

WIDTH = 2200
HEIGHT = 1320
BG = "#f5f7fb"
TEXT = "#10243b"
MUTED = "#4f647d"
LINE = "#9fb0c7"
STAGE = "#dcecff"
RSWA = "#ffe6d6"
FUSION = "#e8f7ea"
TEMP = "#efe8ff"
HEAD = "#fff5d6"
WHITE = "#ffffff"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Verdana.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(50, bold=True)
FONT_SUBTITLE = load_font(28, bold=True)
FONT_BODY = load_font(24, bold=False)
FONT_SMALL = load_font(20, bold=False)
FONT_LABEL = load_font(22, bold=True)


def text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=TEXT, max_width=40, line_gap=6):
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(wrap(paragraph, width=max_width, break_long_words=False, break_on_hyphens=False))
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y = bbox[3] + line_gap
    return y


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=TEXT):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - w) / 2
    y = y0 + (y1 - y0 - h) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


def box(draw: ImageDraw.ImageDraw, coords: tuple[int, int, int, int], title: str, body: str, fill: str):
    draw.rounded_rectangle(coords, radius=24, fill=fill, outline=LINE, width=3)
    x0, y0, x1, y1 = coords
    draw.text((x0 + 18, y0 + 14), title, font=FONT_SUBTITLE, fill=TEXT)
    text_block(draw, (x0 + 18, y0 + 60), body, FONT_BODY, fill=MUTED, max_width=max(18, int((x1 - x0) / 15)))


def small_box(draw: ImageDraw.ImageDraw, coords: tuple[int, int, int, int], text: str, fill: str = WHITE, font=FONT_BODY):
    draw.rounded_rectangle(coords, radius=16, fill=fill, outline=LINE, width=2)
    centered_text(draw, coords, text, font=font)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=TEXT, width=4):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        draw.polygon([(ex, ey), (ex - 14 * sign, ey - 8), (ex - 14 * sign, ey + 8)], fill=fill)
    else:
        sign = 1 if ey >= sy else -1
        draw.polygon([(ex, ey), (ex - 8, ey - 14 * sign), (ex + 8, ey - 14 * sign)], fill=fill)


def elbow_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    via: tuple[int, int],
    end: tuple[int, int],
    fill=TEXT,
    width=4,
):
    draw.line([start, via, end], fill=fill, width=width)
    ex, ey = end
    vx, vy = via
    if abs(ex - vx) >= abs(ey - vy):
        sign = 1 if ex >= vx else -1
        draw.polygon([(ex, ey), (ex - 14 * sign, ey - 8), (ex - 14 * sign, ey + 8)], fill=fill)
    else:
        sign = 1 if ey >= vy else -1
        draw.polygon([(ex, ey), (ex - 8, ey - 14 * sign), (ex + 8, ey - 14 * sign)], fill=fill)


def make_image() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2 - 560, 30), "Arquitetura Geral Atual do Modelo Conjunto", font=FONT_TITLE, fill=TEXT)
    draw.text((WIDTH // 2 - 420, 95), "CNN + BiMamba • topologia separate • branch 01_teste_30seg", font=FONT_LABEL, fill=MUTED)

    box(draw, (40, 150, 1060, 840), "Ramo de Estagiamento (RE)", "", STAGE)
    small_box(draw, (90, 255, 430, 355), "EEG branch\n3 canais\nk = 30, 70, 150", STAGE)
    small_box(draw, (460, 255, 800, 355), "EOG branch\n1 canal\nk = 50, 150, 250", STAGE)
    small_box(draw, (830, 255, 1010, 355), "Concat\n128 canais", WHITE)
    small_box(draw, (210, 410, 900, 500), "SEBlock 128 -> 16 -> 128", WHITE)
    small_box(draw, (210, 530, 900, 620), "Conv1d k=5 + GN + ReLU + Pool", WHITE)
    small_box(draw, (210, 650, 900, 740), "Conv1d k=3 + GN + ReLU", WHITE)
    small_box(draw, (210, 770, 900, 830), "AdaptiveAvgPool1d(1)\nstaging_features [B, T_stage, 256]", WHITE)
    small_box(draw, (160, 875, 950, 960), "BiMamba do staging\n1 bloco bidirecional", TEMP)
    small_box(draw, (160, 985, 950, 1075), "Head de staging\nstaging_logits [B, T_stage, 5]", HEAD)
    small_box(draw, (160, 1095, 950, 1195), "Softmax -> stage_probs [B, T_stage, 5]\nAlinhamento temporal, se necessário.", FUSION, font=FONT_SMALL)

    arrow(draw, (260, 355), (260, 410))
    arrow(draw, (630, 355), (630, 410))
    arrow(draw, (920, 355), (920, 410))
    arrow(draw, (560, 500), (560, 530))
    arrow(draw, (560, 620), (560, 650))
    arrow(draw, (560, 740), (560, 770))
    arrow(draw, (560, 830), (560, 875))
    arrow(draw, (560, 960), (560, 985))
    arrow(draw, (560, 1075), (560, 1095))

    box(draw, (1320, 150, 2160, 975), "Ramo de Detecção de RSWA (RDR)", "", RSWA)
    small_box(draw, (1370, 255, 2110, 370),
              "Canais possíveis do EMG:\n"
              "1) z-score\n"
              "2) EMG / basal\n"
              "3) amplitude relativa\n"
              "4) opcional: RMS relativo",
              WHITE,
              font=FONT_SMALL)
    small_box(draw, (1370, 410, 2110, 515),
              "Encoder CNN multiescala\n"
              "3 paths EMG\n"
              "k = 50, 150, 300\n"
              "Conv1d + GN + ReLU + Pool",
              WHITE)
    small_box(draw, (1370, 545, 2110, 685),
              "Concat -> Dropout -> SEBlock\n"
              "Conv1d k=5 + GN + ReLU + Pool\n"
              "Conv1d k=3 + GN + ReLU\n"
              "emg_cnn_embedding [B, T_rswa, 256]",
              WHITE)
    small_box(draw, (1370, 720, 2110, 860),
              "Ramo local opcional de sub-janelas\n"
              "subwindows de emg_subwindow_ms\n"
              "features: RMS, MAV, STD e amplitude média\n"
              "emg_local_embedding [B, T_rswa, 64]",
              WHITE)
    small_box(draw, (1370, 895, 2110, 955),
              "Fusão local opcional\nConcat [256 + 64] -> Linear + LN + ReLU + Dropout\n"
              "rswa_features [B, T_rswa, 256]",
              FUSION)
    small_box(draw, (1180, 1080, 1500, 1200),
              "Módulo de fusão\nConcat [rswa_features,\nstage_probs] -> Linear 261->256\nLN + ReLU + Dropout",
              FUSION,
              font=FONT_SMALL)
    small_box(draw, (1550, 1080, 2160, 1165),
              "BiMamba do RSWA\n1 bloco bidirecional", TEMP)
    small_box(draw, (1550, 1190, 2160, 1260),
              "Heads por mini-época\n"
              "tonic • phasic • any",
              HEAD)

    for sy, ey in [(370, 410), (515, 545), (685, 720), (860, 895)]:
        arrow(draw, (1740, sy), (1740, ey))
    elbow_arrow(draw, (1740, 955), (1740, 1140), (1180, 1140))
    arrow(draw, (1500, 1140), (1550, 1122))
    arrow(draw, (1855, 1165), (1855, 1190))

    small_box(
        draw,
        (850, 1090, 1130, 1190),
        "Saída do\nestagiamento",
        FUSION,
        font=FONT_SMALL,
    )
    arrow(draw, (950, 1140), (1130, 1140))
    arrow(draw, (1130, 1140), (1180, 1140))

    img.save(OUT, quality=95)


if __name__ == "__main__":
    make_image()
