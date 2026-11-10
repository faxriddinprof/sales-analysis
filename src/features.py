"""Feature engineering: 2- va 3-bosqichlar bir xil featurelardan foydalanishi uchun yagona joy.

Qarorlar (asoslanishi 02_feature_analysis.ipynb da):
- customers, salad_count va boshqa target X ga kirmaydi: ular kun oxirida ma'lum bo'ladi (leakage).
- is_weekend olib tashlanadi: u weekday dan to'liq kelib chiqadi (mukammal multikollinearlik).
- season olib tashlanadi: u month dan to'liq kelib chiqadi.
- humidity va wind olib tashlanadi: harorat va oy nazorat qilinganda ta'siri ahamiyatsiz.
- weekday va month one-hot ko'rinishda (bazaviy: Dushanba, Yanvar).
- Lag featurelar asosiy modelga kirmaydi: qoldiqlarda avtokorrelyatsiya yo'q.
  Shu sababli bitta model 1 kunlik va 7 kunlik ufq uchun bir xil ishlaydi.
"""
from __future__ import annotations

import pandas as pd

from .data import CALENDAR_BIN, EVENTS_BIN, WEATHER_BIN, WEATHER_NUM

BINARY_FEATURES = WEATHER_BIN + CALENDAR_BIN + EVENTS_BIN
NUMERIC_CANDIDATES = WEATHER_NUM            # 2-bosqichda tekshirilgan barcha sonli nomzodlar
NUMERIC_FEATURES = ["temp_c"]               # yakuniy tanlov


def build_features(df: pd.DataFrame, features: str = "final") -> pd.DataFrame:
    """Modelga kiradigan X matritsasini quradi (faqat oldindan ma'lum bo'ladigan ustunlar).

    features="final" — yakuniy 27 feature; features="all" — humidity va wind ham qo'shilgan nomzodlar.
    """
    numeric = NUMERIC_FEATURES if features == "final" else NUMERIC_CANDIDATES
    X = pd.DataFrame(index=df.index)
    for c in numeric:
        X[c] = df[c].astype(float)
    for c in BINARY_FEATURES:
        X[c] = df[c].astype(int)
    # Barcha kategoriyalar har doim mavjud bo'lishi uchun Categorical ishlatiladi
    wd = pd.Categorical(df["weekday"], categories=range(7))
    mo = pd.Categorical(df["month"], categories=range(1, 13))
    X = X.join(pd.get_dummies(wd, prefix="wd", drop_first=True, dtype=int).set_index(df.index))
    X = X.join(pd.get_dummies(mo, prefix="m", drop_first=True, dtype=int).set_index(df.index))
    return X


FEATURES = (NUMERIC_FEATURES + BINARY_FEATURES
            + [f"wd_{i}" for i in range(1, 7)] + [f"m_{i}" for i in range(2, 13)])


def add_lag_features(df: pd.DataFrame, target: str, min_lag: int = 7) -> pd.DataFrame:
    """Tajriba uchun lag featurelar. min_lag=7 bo'lsa 7 kungacha oldindan bashoratda ham ishlatsa bo'ladi."""
    s = df[target]
    out = pd.DataFrame(index=df.index)
    out[f"{target}_lag{min_lag}"] = s.shift(min_lag)
    out[f"{target}_lag{min_lag + 7}"] = s.shift(min_lag + 7)
    out[f"{target}_roll7_lag{min_lag}"] = s.shift(min_lag).rolling(7).mean()
    return out


def train_test_split_time(df: pd.DataFrame, test_start: str = "2025-01-01"):
    """Vaqt bo'yicha bo'lish: test_start dan oldingisi train, keyingisi test."""
    mask = df["date"] < pd.Timestamp(test_start)
    return df[mask].copy(), df[~mask].copy()
