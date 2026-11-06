"""Figures SVG autonomes ; aucune dépendance graphique requise."""

from html import escape
from pathlib import Path


def line_plot(path, series, title, y_label="Valeur"):
    colors = ["#2463eb", "#e36d30", "#14846d", "#ab4cbd"]
    points = [(i, float(y)) for values in series.values() for i, y in enumerate(values)]
    if not points:
        return
    xmax = max(1, max(x for x, _ in points))
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    span = max(ymax - ymin, 0.01)
    ymin -= span * 0.05
    ymax += span * 0.05
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="440" viewBox="0 0 850 440">',
           '<rect width="850" height="440" fill="white"/>',
           f'<text x="65" y="28" font-family="sans-serif" font-size="19">{escape(title)}</text>',
           '<path d="M65 55V360H810" fill="none" stroke="#667085"/>']
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        yp = 360 - 300 * i / 4
        svg.append(f'<text x="7" y="{yp:.1f}" font-family="sans-serif" font-size="12">{y:.3f}</text>')
    for idx, (name, values) in enumerate(series.items()):
        points_text = " ".join(f"{65 + 740 * i / xmax:.1f},{360 - 300 * (float(y) - ymin) / (ymax - ymin):.1f}" for i, y in enumerate(values))
        color = colors[idx % len(colors)]
        svg.extend([f'<polyline points="{points_text}" fill="none" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{70 + (idx % 3) * 250}" y="{402 + (idx // 3) * 18}" fill="{color}" font-family="sans-serif" font-size="12">{escape(name)}</text>'])
    svg.append(f'<text x="65" y="385" font-family="sans-serif" font-size="12">Étape / taille croissante</text><text x="65" y="49" font-family="sans-serif" font-size="12">{escape(y_label)}</text></svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def confusion_plot(path, confusion, names, title):
    size = len(names)
    maximum = max(1, max(max(row) for row in confusion))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="500">',
           '<rect width="650" height="500" fill="white"/>',
           f'<text x="25" y="28" font-family="sans-serif" font-size="18">{escape(title)}</text>']
    for i, name in enumerate(names):
        svg.append(f'<text x="12" y="{123 + i * 90}" font-family="sans-serif" font-size="13">{escape(name)}</text>')
        svg.append(f'<text x="{195 + i * 125}" y="65" font-family="sans-serif" font-size="12">{escape(name)}</text>')
        for j, value in enumerate(confusion[i]):
            opacity = 0.1 + 0.8 * value / maximum
            svg.append(f'<rect x="{175 + j * 125}" y="{80 + i * 90}" width="123" height="88" fill="#2463eb" fill-opacity="{opacity:.3f}"/>')
            svg.append(f'<text x="{226 + j * 125}" y="{130 + i * 90}" font-family="sans-serif" font-size="22" fill="{"white" if opacity > 0.5 else "#172033"}">{value}</text>')
    svg.append('<text x="175" y="420" font-family="sans-serif" font-size="13">Lignes : vérité ; colonnes : prédiction</text></svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")
