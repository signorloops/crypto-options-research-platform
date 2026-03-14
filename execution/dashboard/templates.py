"""HTML template functions for server-side rendered dashboard pages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- CSS ---

DASHBOARD_CSS = """\
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #1f2937; background: #f4f6f8; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.card { background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); padding: 20px; margin-bottom: 20px; }
h1 { margin: 0 0 16px; font-size: 28px; }
h2 { margin: 0 0 12px; font-size: 20px; }
.toolbar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
select, button { padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; background: white; }
button { background: #111827; color: white; cursor: pointer; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }
.nav { background: #111827; padding: 0 24px; display: flex; gap: 0; }
.nav a { color: #9ca3af; text-decoration: none; padding: 14px 20px; font-size: 14px; font-weight: 500; }
.nav a:hover { color: white; background: #1f2937; }
.nav a.active { color: white; border-bottom: 2px solid #3b82f6; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }
.metric-card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); padding: 16px; text-align: center; }
.metric-card .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 24px; font-weight: 700; margin: 4px 0; }
.metric-card .subtitle { font-size: 12px; color: #9ca3af; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.badge-normal { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-restricted { background: #fed7aa; color: #9a3412; }
.badge-halted { background: #fecaca; color: #991b1b; }
.badge-low { background: #dbeafe; color: #1e40af; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-high { background: #fecaca; color: #991b1b; }
@media (max-width: 768px) { .container { padding: 12px; } h1 { font-size: 22px; } .chart-grid { grid-template-columns: 1fr; } }
"""

NAV_ITEMS = [
    ("Overview", "/"),
    ("Backtest", "/backtest"),
    ("Strategy", "/strategy"),
    ("Risk", "/risk"),
]


def nav_html(active: str) -> str:
    """Render navigation bar with the given page marked as active."""
    links = []
    for label, href in NAV_ITEMS:
        cls = ' class="active"' if label == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


def base_layout(title: str, nav_active: str, body_html: str) -> str:
    """Full HTML page wrapper with CSS, Plotly CDN, and navigation."""
    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} — CORP Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{DASHBOARD_CSS}</style>
  </head>
  <body>
    {nav_html(nav_active)}
    <div class="container">
      {body_html}
    </div>
  </body>
</html>"""


def metric_card(label: str, value: str, subtitle: Optional[str] = None) -> str:
    """Render a single KPI metric card."""
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{sub}</div>'


def data_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Render a generic HTML table."""
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'


def file_selector(options: List[Dict[str, str]], selected: str, action_url: str) -> str:
    """Render a file selector dropdown form."""
    opts = "\n".join(
        f'<option value="{o["path"]}" {"selected" if o["path"] == selected else ""}>'
        f'{o["subdir"]}/{o["name"]}</option>'
        for o in options
    )
    return f"""<div class="card">
  <form method="get" action="{action_url}" class="toolbar">
    <label for="file">Result File</label>
    <select id="file" name="file">{opts}</select>
    <button type="submit">Load</button>
  </form>
</div>"""
