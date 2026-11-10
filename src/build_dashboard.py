"""reports/figures/*.json dagi grafiklardan bitta statik HTML hisobot quradi.

Ishga tushirish (loyiha ildizidan):  .venv/bin/python -m src.build_dashboard
Avval uchta notebook ishga tushirilgan bo'lishi kerak (ular grafiklarni saqlaydi).
"""
from __future__ import annotations

import html

import plotly.io as pio

from .data import ROOT, load_data
from .style import FIG_DIR

OUT = ROOT / "docs" / "index.html"

# (bo'lim sarlavhasi, [(grafik nomi, izoh), ...])
SECTIONS = [
    ("1. Ma'lumot va asosiy ko'rsatkichlar", [
        ("01_kpi_by_year", "2023 va 2024 deyarli bir xil. 2025 dagi pastroq qiymat trend emas: 2025 da faqat qish va bahor oylari bor."),
        ("01_rice_daily_by_year", "Har yili bir xil shakl takrorlanadi: qishda past, yozda cho'qqi. Kunlik tebranishning asosiy manbai hafta kuni."),
    ]),
    ("2. Mavsumiylik", [
        ("01_heatmap_weekday_month", "Haftalik (Dushanba → Shanba) va yillik (qish → yoz) mavsumiylik bir-birini kuchaytiradi."),
    ]),
    ("3. Omillarning ta'siri", [
        ("02_coef_plot", "Ko'p omilli regressiya (log y, HC3, 95% CI, train 2023–2024). Guruch va tushum uchun ta'sirlar deyarli bir xil."),
        ("02_raw_vs_adjusted", "Xom farq va sof ta'sir orasidagi tafovut confounding'ni ko'rsatadi: masalan, qorli kunlar asosan qishga to'g'ri keladi."),
    ]),
    ("4. Bashorat modeli", [
        ("03_model_comparison", "Test: 2025-yanvar – 2025-iyun (169 kun). Ridge regressiya baseline'lardan 2.3–2.6 baravar aniqroq va daraxtli modellardan ham yaxshiroq."),
        ("03_test_forecast", "Ridge bashorati va P10–P90 oralig'i. Test kunlarining 80–84% i shu oraliqqa tushadi."),
        ("03_shap", "Ikkala model ham hafta kunini eng muhim omil deb biladi. Mavsum signali Ridge'da oyga, Gradient Boosting'da haroratga tushadi (multikollinearlik)."),
    ]),
    ("5. Kelgusi hafta bashorati (ssenariy)", [
        ("03_next_week", "Ssenariy: harorat — oldingi yillarning shu kunlaridagi o'rtacha, yog'ingarchilik va aksiya yo'q. Zaxirani P90 bo'yicha rejalashtirish tavsiya etiladi."),
    ]),
]

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; color: #222; background: #fff; margin: 0; line-height: 1.55; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.7em; margin-bottom: 0.2em; }
h2 { font-size: 1.25em; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 2.2em; }
.meta { color: #666; font-size: 0.92em; }
table { border-collapse: collapse; margin: 12px 0; font-size: 0.93em; width: 100%; }
th, td { border-bottom: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { border-bottom: 2px solid #444; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 18px 0 28px; }
figcaption { font-size: 0.9em; color: #444; margin-top: 4px; }
.note { font-size: 0.9em; color: #444; background: #f6f6f6; padding: 10px 14px; border-left: 3px solid #2f5d8a; }
ul { padding-left: 1.2em; }
"""


def _summary_table(df) -> str:
    rows = [
        ("Davr", f"{df.date.min():%Y-%m-%d} — {df.date.max():%Y-%m-%d} ({len(df)} kun)"),
        ("Kunlik o'rtacha guruch sarfi", f"{df.rice_kg.mean():.1f} kg"),
        ("Kunlik o'rtacha tushum", f"{df.revenue_uzs.mean() / 1e6:.1f} mln so'm"),
        ("Kunlik o'rtacha mijozlar", f"{df.customers.mean():.0f}"),
        ("Bir mijozga tushum", f"{df.revenue_uzs.sum() / df.customers.sum():,.0f} so'm".replace(",", " ")),
    ]
    body = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{html.escape(v)}</td></tr>" for k, v in rows)
    return f"<table><tbody>{body}</tbody></table>"


MODEL_TABLE = """
<table>
<thead><tr><th>Model (test, 2025)</th><th>Guruch MAPE</th><th>Guruch R²</th><th>Tushum MAPE</th><th>Tushum R²</th></tr></thead>
<tbody>
<tr><td>Baseline: o'tgan hafta</td><td class='num'>27.7%</td><td class='num'>−0.24</td><td class='num'>28.7%</td><td class='num'>−0.30</td></tr>
<tr><td>Baseline: hafta kuni × oy</td><td class='num'>19.7%</td><td class='num'>0.35</td><td class='num'>20.1%</td><td class='num'>0.31</td></tr>
<tr><td><b>Ridge (log y)</b></td><td class='num'><b>8.5%</b></td><td class='num'><b>0.89</b></td><td class='num'><b>7.6%</b></td><td class='num'><b>0.90</b></td></tr>
<tr><td>Random Forest</td><td class='num'>11.4%</td><td class='num'>0.71</td><td class='num'>10.7%</td><td class='num'>0.71</td></tr>
<tr><td>Gradient Boosting</td><td class='num'>9.8%</td><td class='num'>0.80</td><td class='num'>9.0%</td><td class='num'>0.82</td></tr>
</tbody></table>
"""


def build() -> None:
    df = load_data()
    parts, n, first = [], 0, True
    for title, figs in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        if title.startswith("4."):
            parts.append(MODEL_TABLE)
        for name, caption in figs:
            n += 1
            fig = pio.read_json(FIG_DIR / f"{name}.json")
            div = pio.to_html(fig, full_html=False, include_plotlyjs=first, config={"displaylogo": False, "responsive": True})
            first = False
            parts.append(f"<figure>{div}<figcaption><b>Rasm {n}.</b> {html.escape(caption)}</figcaption></figure>")

    page = f"""<!doctype html>
<html lang="uz"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Savdo tahlili va bashorat</title><style>{CSS}</style></head>
<body><main>
<h1>Oshxona savdosi: tahlil va guruch/tushum bashorati</h1>
<p class="meta">Ma'lumot: data/sales.csv · Kod va batafsil tahlil: <a href="https://github.com/faxriddinprof/sales-analysis">github.com/faxriddinprof/sales-analysis</a></p>

<h2>Qisqacha</h2>
{_summary_table(df)}
<ul>
<li>Talabni asosan <b>hafta kuni</b> (Shanba ≈ Dushanba × 1.6) va <b>mavsum</b> (yoz ≈ qish × 1.5) belgilaydi.</li>
<li>Kuchli omillar (sof ta'sir): Hayit +53%, aksiya +34%, bayram +27%, maosh haftasi +20%; Ramazon −36%, qor −32%, yomg'ir −15%.</li>
<li>Guruch va tushum mijozlar soniga deyarli chiziqli bog'liq (r ≈ 0.99), shuning uchun omillar ularga bir xil ta'sir qiladi.</li>
<li>Ridge regressiya 1–7 kun oldinga guruchni ~8.5%, tushumni ~7.6% o'rtacha xato bilan bashorat qiladi.</li>
</ul>
<p class="note">Metodologiya: feature tanlash va giperparametrlar faqat 2023–2024 ma'lumotida (walk-forward CV); 2025-yil yakuniy test.
Bashorat oralig'i (P10–P90) CV qoldiqlaridan olingan. Cheklov: ob-havo ustunlari haqiqiy kuzatuv, real ishlatishda prognoz bo'ladi.</p>
{''.join(parts)}
</main></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} yozildi ({OUT.stat().st_size / 1e6:.1f} MB, {n} ta rasm)")


if __name__ == "__main__":
    build()
