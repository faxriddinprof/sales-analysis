"""Modellar, baseline'lar, metrikalar va yakuniy bashorat funksiyasi.

Barcha modellar log(y) ni o'rganadi (2-bosqich: ta'sirlar multiplikativ), bashorat exp() bilan qaytariladi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT
from .features import FEATURES, build_features

MODEL_DIR = ROOT / "models"

# Walk-forward CV: har bir foldda train kamida 1 yil (oy dummy'lari to'liq bo'lishi uchun), validatsiya ~2.5 oy
CV = TimeSeriesSplit(n_splits=5, test_size=73)
RANDOM_STATE = 42


# ---------------------------------------------------------------- metrikalar
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE, %": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R²": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true),
    }


# ---------------------------------------------------------------- baseline'lar
class SeasonalNaive:
    """y(t) = y(t − 7): o'tgan haftaning shu kuni. 1–7 kunlik ufqda ishlatsa bo'ladi."""

    name = "Baseline: o'tgan hafta"

    def predict(self, full: pd.DataFrame, target: str, index) -> np.ndarray:
        return full[target].shift(7).loc[index].to_numpy()


class CalendarMean(BaseEstimator, RegressorMixin):
    """Hafta kuni × oy bo'yicha o'rtacha (log shkalada). Faqat kalendarga asoslangan oddiy model."""

    def fit(self, X: pd.DataFrame, y):
        key = self._key(X)
        self.table_ = pd.Series(np.asarray(y)).groupby(key.to_numpy()).mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame):
        return self._key(X).map(self.table_).fillna(self.global_).to_numpy()

    @staticmethod
    def _key(X):
        wd = X.filter(regex=r"^wd_\d").to_numpy().argmax(1) + X.filter(regex=r"^wd_\d").to_numpy().any(1)
        mo = X.filter(regex=r"^m_\d").to_numpy().argmax(1) + X.filter(regex=r"^m_\d").to_numpy().any(1)
        return pd.Series(wd * 100 + mo, index=X.index)


# ---------------------------------------------------------------- modellar va giperparametr to'rlari
def model_specs() -> dict:
    """Nom → (estimator, param_grid). Grid kichik: ma'lumot 731 kun."""
    return {
        "Baseline: hafta kuni × oy": (CalendarMean(), {}),
        "Ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": [0.01, 0.1, 1, 3, 10, 30, 100]}),
        "Random Forest": (RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
                          {"min_samples_leaf": [1, 3, 5], "max_features": [0.33, 0.6, 1.0]}),
        "Gradient Boosting": (HistGradientBoostingRegressor(random_state=RANDOM_STATE, early_stopping=False),
                              {"learning_rate": [0.03, 0.1], "max_iter": [200, 500],
                               "max_leaf_nodes": [7, 15], "l2_regularization": [0.0, 1.0]}),
    }


# ---------------------------------------------------------------- bashorat oralig'i (split-conformal)
def oof_log_residuals(estimator, X: pd.DataFrame, y_log: pd.Series, cv=CV) -> np.ndarray:
    """Walk-forward CV dagi out-of-fold qoldiqlar (log shkalada): haqiqiy − bashorat."""
    res = []
    for tr, va in cv.split(X):
        m = clone(estimator).fit(X.iloc[tr], y_log.iloc[tr])
        res.append(y_log.iloc[va].to_numpy() - m.predict(X.iloc[va]))
    return np.concatenate(res)


@dataclass
class Forecaster:
    """Yakuniy model: har bir target uchun log-model + empirik bashorat oralig'i (P10–P90)."""

    estimators: dict = field(default_factory=dict)       # target → fitted estimator
    resid_q: dict = field(default_factory=dict)          # target → {0.1: q, 0.9: q}
    features: list = field(default_factory=lambda: list(FEATURES))

    def predict(self, days: pd.DataFrame) -> pd.DataFrame:
        """days: date, temp_c, rain, snow, kalendar va tadbir belgilari bo'lgan jadval (bir yoki bir necha kun)."""
        days = complete_calendar(days)
        X = build_features(days)[self.features]
        out = pd.DataFrame({"date": days["date"].to_numpy()})
        for t, est in self.estimators.items():
            p = est.predict(X)
            out[f"{t}_P50"] = np.exp(p)
            out[f"{t}_P10"] = np.exp(p + self.resid_q[t][0.1])
            out[f"{t}_P90"] = np.exp(p + self.resid_q[t][0.9])
        return out

    def save(self, path: Path | str = MODEL_DIR / "forecaster.joblib") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str = MODEL_DIR / "forecaster.joblib") -> "Forecaster":
        return joblib.load(path)


def complete_calendar(days: pd.DataFrame) -> pd.DataFrame:
    """Sanadan weekday, month va salary_week ni to'ldiradi; berilmagan belgilarni False deb oladi."""
    d = days.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weekday"] = d["date"].dt.weekday
    d["month"] = d["date"].dt.month
    if "salary_week" not in d:
        dom = d["date"].dt.day
        d["salary_week"] = dom.between(3, 7) | dom.between(18, 22)
    for c in ["rain", "snow", "is_holiday", "ramadan", "eid", "promotion", "local_event", "football_match"]:
        if c not in d:
            d[c] = False
    return d.reset_index(drop=True)
