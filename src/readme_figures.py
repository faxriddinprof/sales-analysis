"""README uchun statik PNG grafiklar (GitHub interaktiv Plotly grafiklarni ko'rsatmaydi).

Ishga tushirish (loyiha ildizidan):  .venv/bin/python -m src.readme_figures
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV

from . import style
from .data import ROOT, TARGETS, WEEKDAYS_UZ, MONTHS_UZ, load_data
from .features import build_features, train_test_split_time
from .models import CV, SeasonalNaive, metrics, model_specs, oof_log_residuals

warnings.filterwarnings("ignore")
IMG = ROOT / "reports" / "img"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ececec", "axes.edgecolor": "#444",
    "axes.titleweight": "bold", "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlelocation": "left", "figure.dpi": 150,
})


def _save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_data()
    train, test = train_test_split_time(df)
    X_tr, X_te = build_features(train), build_features(test)

    # 1. Hafta kuni × oy heatmap
    pivot = df.pivot_table(index="weekday", columns="month", values="rice_kg", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(12), MONTHS_UZ)
    ax.set_yticks(range(7), WEEKDAYS_UZ)
    ax.grid(False)
    for i in range(7):
        for j in range(12):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > np.nanpercentile(pivot.values, 70) else "#222")
    fig.colorbar(im, ax=ax, label="kg/kun", fraction=0.03)
    ax.set_title("Average rice usage, kg/day: weekday × month  |  O'rtacha guruch sarfi")
    _save(fig, "heatmap_weekday_month")

    # 2. Sof ta'sirlar (OLS, log y, train)
    X = sm.add_constant(build_features(train, features="all").astype(float))
    m = sm.OLS(np.log(train["rice_kg"]), X).fit(cov_type="HC3")
    names = {"eid": "Eid / Hayit", "promotion": "Promotion / Aksiya", "is_holiday": "Holiday / Bayram",
             "local_event": "Local event / Tadbir", "salary_week": "Salary week / Maosh haftasi",
             "football_match": "Football / Futbol", "rain": "Rain / Yomg'ir", "snow": "Snow / Qor",
             "ramadan": "Ramadan / Ramazon"}
    ci = m.conf_int()
    eff = {k: ((np.exp(m.params[k]) - 1) * 100, (np.exp(ci.loc[k, 0]) - 1) * 100, (np.exp(ci.loc[k, 1]) - 1) * 100)
           for k in names}
    order = sorted(eff, key=lambda k: eff[k][0])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(order))
    vals = np.array([eff[k] for k in order])
    colors = [style.GREEN if v > 0 else style.RED for v in vals[:, 0]]
    ax.barh(y, vals[:, 0], color=colors, height=0.6)
    ax.errorbar(vals[:, 0], y, xerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="none", ecolor="#333", capsize=2, lw=1)
    for yi, (v, lo, hi) in zip(y, vals):
        ax.text(hi + 2 if v > 0 else lo - 2, yi, f"{v:+.0f}%", va="center", ha="left" if v > 0 else "right", fontsize=9)
    ax.set_yticks(y, [names[k] for k in order])
    ax.axvline(0, color="#444", lw=1)
    ax.grid(axis="x", color="#ececec")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-55, 80)
    ax.set_xlabel("Effect on rice usage, % (95% CI)  |  Guruch sarfiga sof ta'sir, %")
    ax.set_title("Adjusted effects (OLS on log y, other factors held constant)")
    _save(fig, "effects")

    # 3. Modellarni solishtirish va test bashorati
    order_models = ["Baseline: o'tgan hafta", "Baseline: hafta kuni × oy", "Random Forest", "Gradient Boosting", "Ridge"]
    labels = {"Baseline: o'tgan hafta": "Baseline: last week", "Baseline: hafta kuni × oy": "Baseline: weekday × month",
              "Random Forest": "Random Forest", "Gradient Boosting": "Gradient Boosting", "Ridge": "Ridge (log y)"}
    mape, best = {t: {} for t in TARGETS}, {}
    for t in TARGETS:
        y_log = np.log(train[t])
        mape[t]["Baseline: o'tgan hafta"] = metrics(test[t], SeasonalNaive().predict(df, t, test.index))["MAPE, %"]
        for name, (est, grid) in model_specs().items():
            gs = GridSearchCV(est, grid, cv=CV, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_tr, y_log)
            mape[t][name] = metrics(test[t], np.exp(gs.best_estimator_.predict(X_te)))["MAPE, %"]
            if name == "Ridge":
                best[t] = gs.best_estimator_

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, t, title in zip(axes, TARGETS, ["Rice / Guruch", "Revenue / Tushum"]):
        v = [mape[t][k] for k in order_models]
        c = [style.GREY, style.GREY, "#a9c1dd", "#a9c1dd", style.BLUE]
        ax.barh(range(5), v, color=c, height=0.6)
        for i, x in enumerate(v):
            ax.text(x + 0.4, i, f"{x:.1f}%", va="center", fontsize=9)
        ax.set_yticks(range(5), [labels[k] for k in order_models])
        ax.set_xlim(0, max(v) * 1.2)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#ececec")
        ax.set_title(f"{title}: test MAPE (2025)")
    _save(fig, "model_comparison")

    t = "rice_kg"
    r = oof_log_residuals(best[t], X_tr, np.log(train[t]))
    p = best[t].predict(X_te)
    p50, p10, p90 = np.exp(p), np.exp(p + np.quantile(r, 0.1)), np.exp(p + np.quantile(r, 0.9))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(test.date, p10, p90, color="#8c8c8c", alpha=0.25, lw=0, label="P10–P90")
    ax.plot(test.date, test[t], color="#222", lw=0.9, label="Actual / Haqiqiy")
    ax.plot(test.date, p50, color=style.BLUE, lw=1.8, label="Forecast / Bashorat (Ridge)")
    mm = metrics(test[t], p50)
    ax.set_title(f"Rice usage, test period 2025 (MAPE {mm['MAPE, %']:.1f}%, R² {mm['R²']:.2f})  |  Guruch sarfi, test davri")
    ax.set_ylabel("kg/day")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    _save(fig, "test_forecast")
    print("PNG:", sorted(p.name for p in IMG.glob("*.png")))


if __name__ == "__main__":
    main()
