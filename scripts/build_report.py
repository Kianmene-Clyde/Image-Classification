"""Génère le rapport PDF à partir du texte et des résultats synthétiques vérifiés."""
from argparse import ArgumentParser
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String, PolyLine
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#162D49")
BLUE = colors.HexColor("#2463C5")
TEAL = colors.HexColor("#087F72")
GRAY = colors.HexColor("#53637B")
PALE = colors.HexColor("#EDF3FC")
LINE = colors.HexColor("#D4DEED")
INK = colors.HexColor("#18283C")
WHITE = colors.white


def typography():
    fonts = [
        (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in fonts:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Report", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            return "Report", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


FONT, BOLD = typography()


def style(name, size, leading, color=INK, bold=False, **kwargs):
    return ParagraphStyle(name, fontName=BOLD if bold else FONT, fontSize=size,
                          leading=leading, textColor=color, **kwargs)


LABEL = style("label", 10, 14, BLUE, True, spaceAfter=10)
TITLE = style("title", 31, 38, NAVY, True, spaceAfter=20)
CHAPTER = style("chapter", 27, 33, NAVY, True, spaceAfter=23)
SUBTITLE = style("subtitle", 13, 21, GRAY, spaceAfter=21)
BODY = style("body", 12, 20, INK, spaceAfter=14)
CAPTION = style("caption", 9.5, 14, GRAY, spaceBefore=7, spaceAfter=16)
TABLE_BODY = style("table body", 9, 12)
TABLE_HEAD = style("table head", 9, 12, WHITE, True)


def para(text, design=BODY):
    escaped = escape(text).replace("—", "-").replace("–", "-").replace("‑", "-")
    formatted = re.sub(r"`([^`]+)`", r'<font color="#2463C5">\1</font>', escaped)
    return Paragraph(formatted, design)


def table(rows, widths, compact=False):
    values = [[para(str(value), TABLE_HEAD if index == 0 else TABLE_BODY)
               for value in row] for index, row in enumerate(rows)]
    result = Table(values, colWidths=widths, hAlign="LEFT", repeatRows=1)
    padding = 1.5 if compact else 6
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
    ]))
    return result


def text_at(drawing, x, y, text, size=10, color=INK, bold=False):
    drawing.add(String(x, y, text, fontName=BOLD if bold else FONT, fontSize=size, fillColor=color))


def flow_figure():
    drawing = Drawing(470, 92)
    for i, (name, detail) in enumerate([("Images", "Décodage et labels"), ("Groupes", "Scènes et doublons"), ("Partitions", "Train / validation / test")]):
        x = 156 * i
        drawing.add(Rect(x, 16, 144, 58, rx=7, ry=7, fillColor=PALE, strokeColor=LINE))
        text_at(drawing, x + 10, 50, name, 13, NAVY, True)
        text_at(drawing, x + 10, 31, detail, 8.6, GRAY)
        if i < 2:
            drawing.add(Line(x + 145, 44, x + 155, 44, strokeColor=BLUE, strokeWidth=2))
    return drawing


def xor_figure():
    drawing = Drawing(470, 196)
    drawing.add(Rect(0, 8, 460, 178, rx=7, ry=7, fillColor=PALE, strokeColor=LINE))
    drawing.add(Line(36, 33, 169, 33, strokeColor=GRAY))
    drawing.add(Line(36, 33, 36, 157, strokeColor=GRAY))
    for x, y, color in ((55, 52, BLUE), (141, 139, BLUE), (55, 139, TEAL), (141, 52, TEAL)):
        drawing.add(Circle(x, y, 7, fillColor=color, strokeColor=None))
    text_at(drawing, 215, 137, "XOR sur quatre points", 13, NAVY, True)
    text_at(drawing, 215, 106, "Régression logistique   0,50", 11)
    text_at(drawing, 215, 84, "SVM à noyau RBF          1,00", 11)
    text_at(drawing, 215, 55, "Accuracy d'entraînement du notebook", 9, GRAY)
    return drawing


def benchmark_figure(metrics):
    rows = metrics["comparison"]
    final = {row["model"]: row["metrics"]["macro_f1"] for row in metrics["test"]}
    data = [["Modèle", "Validation F1", "Test F1", "Ajustement s"]]
    for row in rows:
        data.append([row["model"], f"{row['validation_macro_f1_mean']:.3f} ± {row['validation_macro_f1_std']:.3f}",
                     f"{final[row['model']]:.3f}", f"{row['fit_seconds_mean']:.3f}"])
    return table(data, [155, 116, 80, 119], compact=True)


def learning_figure(metrics):
    grouped = defaultdict(list)
    for item in metrics["mlp_learning_curve"]:
        if item["candidate"] == 0:
            for part in ("train", "validation"):
                grouped[(item["n_train"], part)].append(item[part]["macro_f1"])
    sizes = sorted({n for n, _ in grouped})
    drawing = Drawing(470, 230)
    x0, y0, width, height = 48, 50, 392, 145
    for i in range(6):
        y = y0 + i * height / 5
        drawing.add(Line(x0, y, x0 + width, y, strokeColor=LINE))
        text_at(drawing, 12, y - 3, f"{i/5:.1f}", 9, GRAY)
    for i, n in enumerate(sizes):
        x = x0 + width * i / max(1, len(sizes) - 1)
        text_at(drawing, x - 5, 34, str(n), 9, GRAY)
    for index, (part, color) in enumerate((("train", BLUE), ("validation", TEAL))):
        points = []
        for i, n in enumerate(sizes):
            values = grouped[(n, part)]
            x = x0 + width * i / max(1, len(sizes) - 1)
            y = y0 + height * sum(values) / len(values)
            points.append((x, y))
        drawing.add(PolyLine(points, strokeColor=color, strokeWidth=1.6, strokeDashArray=[4, 3] if index else None))
        for x, y in points:
            drawing.add(Circle(x, y, 4.2 if index == 0 else 2.4, fillColor=color, strokeColor=None))
        text_at(drawing, 52 + index * 110, 208, part.capitalize(), 10, color, True)
    text_at(drawing, 12, 208, "F1", 9, GRAY)
    text_at(drawing, 154, 13, "Nombre d'images d'entraînement", 9, GRAY)
    return drawing


def representations_figure(metrics):
    models = ["logistic", "decision_tree", "mlp"]
    rows = [["Représentation", "Logistique", "Arbre", "MLP"]]
    for row in metrics["representation_comparison"]:
        scores = {item["model"]: item["validation_macro_f1_mean"] for item in row["comparison"]}
        rows.append([f"{row['features'].upper()} {row['size']} × {row['size']}"] + [f"{scores[name]:.3f}" for name in models])
    return table(rows, [174, 104, 96, 96])


def figure(name, metrics):
    partitions = metrics["partition_sizes"]
    builders = {
        "flow": (flow_figure, "Figure 1. Les groupes sont établis avant l'affectation aux partitions."),
        "xor": (xor_figure, "Figure 2. Vérification de capacité sur quatre observations déterministes."),
        "benchmark": (lambda: benchmark_figure(metrics),
                      f"Tableau 1. Données synthétiques. Train {partitions['train']}, validation {partitions['validation']}, "
                      f"test {partitions['test']}. {metrics['trial_count']} essais de validation."),
        "learning": (lambda: learning_figure(metrics),
                     "Figure 3. Macro-F1 moyen sur trois graines, premier réglage du MLP. "
                     "Les marqueurs superposés indiquent un score identique."),
        "representations": (lambda: representations_figure(metrics),
                            "Tableau 2. Macro-F1 de validation. Mêmes images, une graine, premier réglage."),
    }
    build, caption = builders[name]
    return [build(), para(caption, CAPTION)]


def decorate(canvas, document):
    page = document.page
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 8, width, 8, fill=1, stroke=0)
    canvas.setFont(BOLD, 8.5)
    canvas.setFillColor(BLUE)
    canvas.drawString(60, height - 31, "SIGNLAB   /   PROJET ANNUEL IABD")
    canvas.setStrokeColor(LINE)
    canvas.line(60, 44, width - 60, 44)
    canvas.setFont(FONT, 8.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(60, 29, "RAPPORT TECHNIQUE ET EXPÉRIMENTAL")
    canvas.drawRightString(width - 60, 29, f"{page:02d}")


def generate(output):
    source = (ROOT / "docs/RAPPORT.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "docs/verification-synthetique.json").read_text(encoding="utf-8"))
    if metrics["scope"] != "synthetic":
        raise ValueError("Le rapport attend les mesures de contrôle synthétiques.")
    cover, *chapters = re.split(r"^## ", source, flags=re.MULTILINE)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = SimpleDocTemplate(str(output), pagesize=A4, title="SignLab - Rapport technique et expérimental",
                               author="Clyde Kianmène Takuefou", leftMargin=60, rightMargin=60, topMargin=62, bottomMargin=62)
    story = [Spacer(1, 95), para("VISION PAR ORDINATEUR ET APPRENTISSAGE SUPERVISÉ", LABEL),
             para("Classification de panneaux", TITLE), para("Rapport technique et expérimental", SUBTITLE)]
    for paragraph in cover.strip().split("\n\n")[1:]:
        story.append(para(paragraph))
    story.extend([Spacer(1, 16), table([["Périmètre expérimental", "État"], ["Catalogue Python", "23 modèles et variantes"],
                                      ["Vérification des algorithmes", f"{metrics['trial_count']} essais synthétiques"],
                                      ["Photos réelles", "Répertoires réservés au dataset"]], [255, 215]), PageBreak()])
    story.extend([para("NAVIGATION", LABEL), para("Sommaire", CHAPTER)])
    toc = [["Chapitre", "Page"]]
    for index, section in enumerate(chapters, 3):
        heading = section.splitlines()[0]
        toc.append([heading, str(index)])
    story.append(table(toc, [402, 68], compact=True))
    for chapter in chapters:
        heading, _, content = chapter.strip().partition("\n")
        number, _, title = heading.partition(" — ")
        story.extend([PageBreak(), para(f"CHAPITRE {number}", LABEL), para(title, CHAPTER)])
        for paragraph in content.strip().split("\n\n"):
            marker = re.fullmatch(r"\[figure:(\w+)\]", paragraph.strip())
            if marker:
                story.extend(figure(marker.group(1), metrics))
            else:
                story.append(para(paragraph))
    report.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(output.resolve())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/Rapport-SignLab.pdf")
    generate(parser.parse_args().output)
