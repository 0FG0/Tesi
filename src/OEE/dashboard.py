"""
dashboard_oee.py
────────────────
Genera una dashboard HTML interattiva dall'output di oee_calculator.

Come usarlo:
    python src/dashboard_oee.py
    → outputs/dashboard_oee.html  (apribile nel browser senza server)

Dipendenze: pandas, jinja2  (pip install jinja2)
Grafici: Chart.js via CDN (nessuna installazione)
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OEE_calculator import calcola_oee, genera_alert, SOGLIA_ACCETTABILE, SOGLIA_OTTIMO

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
INPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_HTML  = os.path.join(PROJECT_ROOT, "outputs", "dashboard_oee.html")


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepara_dati(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df = calcola_oee(df)
    df = df.dropna(subset=["OEE"])
    if "Data_Ora_Fine" in df.columns:
        df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"], errors="coerce")
        df["Mese"]      = df["Data_Ora_Fine"].dt.to_period("M").astype(str)
        df["Settimana"] = df["Data_Ora_Fine"].dt.to_period("W").astype(str)
    return df


def trend_mensile(df: pd.DataFrame) -> dict:
    if "Mese" not in df.columns:
        return {}
    g = df.groupby("Mese")[["OEE","OEE_Disponibilita","OEE_Performance","OEE_Qualita"]].mean().round(4)
    return {
        "labels": g.index.tolist(),
        "oee":    g["OEE"].tolist(),
        "disp":   g["OEE_Disponibilita"].tolist(),
        "perf":   g["OEE_Performance"].tolist(),
        "qual":   g["OEE_Qualita"].tolist(),
    }


def top_wo_critici(df: pd.DataFrame, n: int = 10) -> list:
    alert = genera_alert(df)
    cols = [c for c in ["WO", "FASE", "ARTICOLO", "Data_Ora_Fine", "OEE", "OEE_Classe",
                        "OEE_Disponibilita", "OEE_Performance", "OEE_Qualita", "Alert_Motivo"] if c in alert.columns]
    out = alert[cols].dropna(subset=["OEE"]).sort_values("OEE").head(n).round(4).copy()
    # convert non-JSON-serializable columns to string
    for col in ["Data_Ora_Fine", "OEE_Classe"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out.to_dict(orient="records")


def distribuzione_classi(df: pd.DataFrame) -> dict:
    if "OEE_Classe" not in df.columns:
        return {}
    vc = df["OEE_Classe"].astype(str).value_counts()
    order = ["Ottimo", "Accettabile", "Critico"]
    labels = [l for l in order if l in vc.index]
    values = [int(vc.get(l, 0)) for l in labels]
    return {"labels": labels, "values": values}


def oee_per_articolo(df: pd.DataFrame, n: int = 15) -> dict:
    if "ARTICOLO" not in df.columns:
        return {}
    g = df.groupby("ARTICOLO")["OEE"].mean().dropna().sort_values().head(n).round(4)
    return {"labels": g.index.tolist(), "values": g.values.tolist()}


def kpi_globali(df: pd.DataFrame) -> dict:
    return {
        "oee_medio":   round(df["OEE"].mean() * 100, 1)               if "OEE" in df.columns else "-",
        "disp_media":  round(df["OEE_Disponibilita"].mean() * 100, 1) if "OEE_Disponibilita" in df.columns else "-",
        "perf_media":  round(df["OEE_Performance"].mean() * 100, 1)   if "OEE_Performance" in df.columns else "-",
        "qual_media":  round(df["OEE_Qualita"].mean() * 100, 1)        if "OEE_Qualita" in df.columns else "-",
        "n_critici":   int((df["OEE"] < SOGLIA_ACCETTABILE).sum())     if "OEE" in df.columns else 0,
        "n_wc":        int((df["OEE"] >= SOGLIA_OTTIMO).sum())          if "OEE" in df.columns else 0,
        "totale_wo":   len(df),
        "generato_il": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OEE Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0d0f14;
    --surface:  #141720;
    --border:   #1e2130;
    --accent:   #00e5a0;
    --warn:     #f5a623;
    --danger:   #e5004c;
    --text:     #e8eaf0;
    --muted:    #5a6070;
    --radius:   12px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }}

  header {{
    padding: 28px 40px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 16px;
  }}
  header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }}
  header span {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }}

  .grid {{
    display: grid;
    gap: 16px;
    padding: 24px 40px;
  }}

  .kpi-row {{ grid-template-columns: repeat(4, 1fr); }}
  .charts-row {{ grid-template-columns: 2fr 1fr; }}
  .bottom-row {{ grid-template-columns: 1fr 1fr; }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
  }}

  .kpi-card {{
    position: relative;
    overflow: hidden;
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
  }}
  .kpi-card.warn::before  {{ background: var(--warn); }}
  .kpi-card.danger::before {{ background: var(--danger); }}

  .kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-family: 'DM Mono', monospace;
    font-size: 36px;
    font-weight: 500;
    line-height: 1;
    color: var(--accent);
  }}
  .kpi-card.warn  .kpi-value  {{ color: var(--warn); }}
  .kpi-card.danger .kpi-value {{ color: var(--danger); }}
  .kpi-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}

  .card-title {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }}

  canvas {{ width: 100% !important; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th {{
    text-align: left;
    padding: 6px 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }}
  .badge-critico   {{ background: rgba(229,0,76,0.15);  color: var(--danger); }}
  .badge-accettabile{{ background: rgba(245,166,35,0.15); color: var(--warn); }}
  .badge-wc        {{ background: rgba(0,229,160,0.15);  color: var(--accent); }}

  .oee-bar {{
    display: inline-block;
    height: 6px;
    border-radius: 3px;
    background: var(--danger);
    vertical-align: middle;
    margin-right: 6px;
    transition: width 0.3s;
  }}
  .oee-bar.ok {{ background: var(--accent); }}
  .oee-bar.mid {{ background: var(--warn); }}

  footer {{
    padding: 16px 40px;
    color: var(--muted);
    font-size: 11px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <h1>OEE Dashboard</h1>
  <span>Generato il {generato_il} &nbsp;·&nbsp; {totale_wo} WO analizzati</span>
</header>

<!-- KPI -->
<div class="grid kpi-row">
  <div class="card kpi-card">
    <div class="kpi-label">OEE Medio</div>
    <div class="kpi-value">{oee_medio}%</div>
    <div class="kpi-sub">Disponibilità × Performance × Qualità</div>
  </div>
  <div class="card kpi-card">
    <div class="kpi-label">Disponibilità</div>
    <div class="kpi-value">{disp_media}%</div>
    <div class="kpi-sub">Uptime effettivo / programmato</div>
  </div>
  <div class="card kpi-card warn">
    <div class="kpi-label">Performance</div>
    <div class="kpi-value">{perf_media}%</div>
    <div class="kpi-sub">Velocità reale / velocità teorica</div>
  </div>
  <div class="card kpi-card">
    <div class="kpi-label">Qualità</div>
    <div class="kpi-value">{qual_media}%</div>
    <div class="kpi-sub">Pezzi buoni / tot pezzi</div>
  </div>
</div>

<!-- TREND + DISTRIBUZIONE -->
<div class="grid charts-row">
  <div class="card">
    <div class="card-title">Trend OEE — dettaglio mensile</div>
    <div id="trendContainer" style="display:flex;flex-direction:column;gap:20px;max-height:330px;overflow-y:auto;padding-right:6px;scrollbar-width:thin;scrollbar-color:#1e2130 transparent;"></div>
  </div>
  <div class="card">
    <div class="card-title">Distribuzione classi OEE</div>
    <canvas id="distChart" height="110"></canvas>
  </div>
</div>

<!-- ARTICOLI + ALERT TABLE -->
<div class="grid bottom-row">
  <div class="card">
    <div class="card-title">Articoli con OEE più basso</div>
    <canvas id="artChart" height="180"></canvas>
  </div>
  <div class="card">
    <div class="card-title">WO critici — OEE sotto soglia</div>
    <div style="overflow-x:auto;max-height:168px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#1e2130 transparent;">
      <table>
        <thead>
          <tr>
            <th>WO</th><th>Fase</th><th>OEE</th><th>D</th><th>P</th><th>Q</th><th>Classe</th><th>Alert</th>
          </tr>
        </thead>
        <tbody id="alertTable"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ALERT SUMMARY -->
<div class="grid" style="grid-template-columns:1fr;">
  <div class="card" style="background:#0d0f14;border-color:#1e2130;">
    <div class="card-title" style="color:#e5004c;">⚠ Alert automatici</div>
    <div id="alertSummary" style="font-size:12px;line-height:1.8;color:#c0c4cc;"></div>
  </div>
</div>

<footer>OEE = Disponibilità × Performance × Qualità &nbsp;|&nbsp; Ottimo ≥ 85% &nbsp;|&nbsp; Accettabile ≥ 65% &nbsp;|&nbsp; Critico &lt; 65%</footer>

<script>
const TREND = {trend_json};
const DIST  = {dist_json};
const ARTS  = {art_json};
const CRITICI = {critici_json};
const KPI  = {kpi_json};

// ── Trend Charts (mensile) ─────────────────────────────────────────────────
if (TREND.labels && TREND.labels.length) {{
  const trendContainer = document.getElementById('trendContainer');
  for (let i = 0; i < TREND.labels.length; i++) {{
    const mese = TREND.labels[i];
    const wrapper = document.createElement('div');
    const periodLabel = document.createElement('div');
    periodLabel.style.cssText = 'font-size:10px;color:#5a6070;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;';
    periodLabel.textContent = mese;
    const canvas = document.createElement('canvas');
    canvas.height = 280;
    wrapper.appendChild(periodLabel);
    wrapper.appendChild(canvas);
    trendContainer.appendChild(wrapper);
    const vals = [TREND.oee[i], TREND.disp[i], TREND.perf[i], TREND.qual[i]].filter(v => v != null);
    const yMin = Math.max(0, Math.floor(Math.min(...vals) * 20) / 20 - 0.03);
    const yMax = Math.min(1, Math.ceil(Math.max(...vals) * 20) / 20 + 0.02);
    new Chart(canvas, {{
      type: 'bar',
      data: {{
        labels: ['OEE', 'Disponibilità', 'Performance', 'Qualità'],
        datasets: [{{
          data: [TREND.oee[i], TREND.disp[i], TREND.perf[i], TREND.qual[i]],
          backgroundColor: ['rgba(0,229,160,0.7)', 'rgba(91,138,245,0.7)', 'rgba(245,166,35,0.7)', 'rgba(199,125,255,0.7)'],
          borderColor:     ['#00e5a0', '#5b8af5', '#f5a623', '#c77dff'],
          borderWidth: 1,
          borderRadius: 4,
        }}]
      }},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks:{{ color:'#8890a4', font:{{size:11}} }}, grid:{{ display:false }} }},
          y: {{ min: yMin, max: yMax, ticks:{{ color:'#5a6070', font:{{size:10}}, callback: v => (v*100).toFixed(0)+'%' }}, grid:{{ color:'#1e2130' }} }}
        }}
      }}
    }});
  }}
}}

// ── Distribuzione Classi ───────────────────────────────────────────────────
// label order from Python is always: Ottimo, Accettabile, Critico
if (DIST.labels && DIST.labels.length) {{
  new Chart(document.getElementById('distChart'), {{
    type: 'doughnut',
    data: {{
      labels: DIST.labels,
      datasets: [{{ data: DIST.values, backgroundColor: ['#00e5a0','#f5a623','#e5004c'], borderWidth: 0 }}]
    }},
    options: {{
      plugins: {{
        legend: {{ position:'bottom', labels:{{ color:'#8890a4', font:{{size:11}}, padding:16 }} }}
      }},
      cutout: '65%'
    }}
  }});
}}

// ── Articoli chart ─────────────────────────────────────────────────────────
if (ARTS.labels && ARTS.labels.length) {{
  const colors = ARTS.values.map(v => v < 0.65 ? '#e5004c' : v < 0.85 ? '#f5a623' : '#00e5a0');
  new Chart(document.getElementById('artChart'), {{
    type: 'bar',
    data: {{
      labels: ARTS.labels,
      datasets: [{{ label: 'OEE medio', data: ARTS.values, backgroundColor: colors, borderRadius: 4, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y',
      plugins: {{ legend:{{ display:false }} }},
      scales: {{
        x: {{ min:0, max:1, ticks:{{ color:'#5a6070', font:{{size:10}}, callback: v => (v*100)+'%' }}, grid:{{ color:'#1e2130' }} }},
        y: {{ ticks:{{ color:'#8890a4', font:{{size:10}} }}, grid:{{ display:false }} }}
      }}
    }}
  }});
}}

// ── Tabella WO critici ─────────────────────────────────────────────────────
const tbody = document.getElementById('alertTable');
CRITICI.forEach(r => {{
  const oee = r.OEE ?? '-';
  const pct = v => typeof v === 'number' ? (v*100).toFixed(1)+'%' : '-';
  const classe = (r.OEE_Classe||'').toLowerCase();
  const badge = (classe === 'ottimo' || classe.includes('world')) ? 'badge-wc' : classe === 'accettabile' ? 'badge-accettabile' : 'badge-critico';
  const barW = typeof oee === 'number' ? Math.round(oee*60) : 0;
  const barClass = oee >= 0.85 ? 'ok' : oee >= 0.65 ? 'mid' : '';
  tbody.innerHTML += `<tr>
    <td>${{r.WO||'-'}}</td>
    <td>${{r.FASE||'-'}}</td>
    <td><span class="oee-bar ${{barClass}}" style="width:${{barW}}px"></span>${{pct(oee)}}</td>
    <td>${{pct(r.OEE_Disponibilita)}}</td>
    <td>${{pct(r.OEE_Performance)}}</td>
    <td>${{pct(r.OEE_Qualita)}}</td>
    <td><span class="badge ${{badge}}">${{r.OEE_Classe||'-'}}</span></td>
    <td style="color:#8890a4;font-size:10px">${{r.Alert_Motivo||'-'}}</td>
  </tr>`;
}});

// ── Alert summary ──────────────────────────────────────────────────────────
const alertDiv = document.getElementById('alertSummary');
const msgs = [];
if (KPI.n_critici > 0)
  msgs.push(`<b style="color:#e5004c">${{KPI.n_critici}} WO con OEE critico</b> (sotto il 65%) richiedono attenzione immediata.`);
if (KPI.perf_media < 70)
  msgs.push(`Performance media al <b>${{KPI.perf_media}}%</b> — verificare tempi teorici AS400 o cause di rallentamento.`);
if (KPI.disp_media < 80)
  msgs.push(`Disponibilità media al <b>${{KPI.disp_media}}%</b> — analizzare le cause dei fermi macchina.`);
if (KPI.qual_media < 95)
  msgs.push(`Qualità media al <b>${{KPI.qual_media}}%</b> — revisione scarti e rilavorazioni consigliata.`);
if (msgs.length === 0)
  msgs.push('Nessun alert critico. OEE globale nei parametri accettabili.');
alertDiv.innerHTML = msgs.map(m => '→ ' + m).join('<br>');
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def genera_dashboard(path_input: str = INPUT_CSV, path_output: str = OUTPUT_HTML):
    print(f"Caricamento dati da: {path_input}")
    df = prepara_dati(path_input)

    kpi      = kpi_globali(df)
    trend    = trend_mensile(df)
    dist     = distribuzione_classi(df)
    arts     = oee_per_articolo(df)
    critici  = top_wo_critici(df)

    html = HTML_TEMPLATE.format(
        **kpi,
        trend_json   = json.dumps(trend),
        dist_json    = json.dumps(dist),
        art_json     = json.dumps(arts),
        critici_json = json.dumps(critici),
        kpi_json     = json.dumps(kpi),
    )

    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard generata: {path_output}")


if __name__ == "__main__":
    genera_dashboard()