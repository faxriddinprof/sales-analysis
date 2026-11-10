"""Ma'lumotni yuklash, ustunlar lug'ati va sifat tekshiruvlari."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sales.csv"

# Bashorat qilinadigan ko'rsatkichlar
TARGETS = ["rice_kg", "revenue_uzs"]

# Kun oxirida ma'lum bo'ladigan ustunlar: bashoratda feature sifatida ishlatib bo'lmaydi (leakage)
OUTCOMES = ["customers", "rice_kg", "salad_count", "revenue_uzs"]

WEATHER_NUM = ["temp_c", "humidity", "wind"]
WEATHER_BIN = ["rain", "snow"]
CALENDAR = ["weekday", "is_weekend", "month", "season"]
CALENDAR_BIN = ["is_holiday", "ramadan", "eid", "salary_week"]
EVENTS_BIN = ["promotion", "local_event", "football_match"]
BINARY = WEATHER_BIN + CALENDAR_BIN + EVENTS_BIN

WEEKDAYS_UZ = ["Dush", "Sesh", "Chor", "Pay", "Juma", "Shan", "Yak"]
MONTHS_UZ = ["Yan", "Fev", "Mar", "Apr", "May", "Iyun", "Iyul", "Avg", "Sen", "Okt", "Noy", "Dek"]
SEASONS_UZ = {"winter": "Qish", "spring": "Bahor", "summer": "Yoz", "autumn": "Kuz"}

# Ustunlar lug'ati: guruh, tur, birlik, ma'no
DICTIONARY = {
    "date": ("Sana", "sana", "—", "Kuzatuv kuni"),
    "customers": ("Natija", "butun son", "kishi", "Kunlik mijozlar soni"),
    "rice_kg": ("Natija (target)", "son", "kg", "Kunlik guruch sarfi"),
    "salad_count": ("Natija", "butun son", "dona", "Sotilgan salatlar soni"),
    "revenue_uzs": ("Natija (target)", "butun son", "so'm", "Kunlik tushum"),
    "temp_c": ("Ob-havo", "son", "°C", "Havo harorati"),
    "humidity": ("Ob-havo", "son", "%", "Nisbiy namlik"),
    "wind": ("Ob-havo", "son", "m/s", "Shamol tezligi"),
    "rain": ("Ob-havo", "binary", "—", "Yomg'ir yog'dimi"),
    "snow": ("Ob-havo", "binary", "—", "Qor yog'dimi"),
    "weekday": ("Kalendar", "kategoriya", "0–6", "Hafta kuni (0 = Dushanba)"),
    "is_weekend": ("Kalendar", "binary", "—", "Dam olish kuni (Shan/Yak)"),
    "month": ("Kalendar", "kategoriya", "1–12", "Oy"),
    "season": ("Kalendar", "kategoriya", "—", "Fasl"),
    "is_holiday": ("Kalendar", "binary", "—", "Davlat bayrami"),
    "ramadan": ("Kalendar", "binary", "—", "Ramazon oyi"),
    "eid": ("Kalendar", "binary", "—", "Hayit kunlari"),
    "salary_week": ("Kalendar", "binary", "—", "Maosh haftasi (oyning 3–7 va 18–22 kunlari)"),
    "promotion": ("Tadbir", "binary", "—", "Aksiya o'tkazildimi"),
    "local_event": ("Tadbir", "binary", "—", "Mahalliy tadbir"),
    "football_match": ("Tadbir", "binary", "—", "Futbol o'yini"),
}


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """CSV ni o'qiydi, turlarni to'g'rilaydi va sana bo'yicha tartiblaydi."""
    df = pd.read_csv(path, parse_dates=["date"])
    for c in BINARY + ["is_weekend"]:
        df[c] = df[c].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    return df


def column_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Har bir ustun uchun: guruh, tur, birlik, ma'no va haqiqiy qiymatlar oralig'i."""
    rows = []
    for col, (group, kind, unit, meaning) in DICTIONARY.items():
        s = df[col]
        if kind == "binary":
            rng = f"True: {int(s.sum())} kun ({s.mean():.1%})"
        elif col == "season":
            rng = ", ".join(sorted(s.unique()))
        elif col == "date":
            rng = f"{s.min():%Y-%m-%d} … {s.max():%Y-%m-%d}"
        else:
            rng = f"{s.min():,.1f} … {s.max():,.1f}"
        rows.append({"ustun": col, "guruh": group, "tur": kind, "birlik": unit,
                     "ma'nosi": meaning, "qiymatlar": rng})
    return pd.DataFrame(rows)


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Sifat va mantiqiy tekshiruvlar. Har bir qator: tekshiruv, natija, holat."""
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    checks = [
        ("Bo'sh qiymatlar (NaN)", int(df.isna().sum().sum()), 0),
        ("Takroriy sanalar", int(df["date"].duplicated().sum()), 0),
        ("Tushib qolgan kunlar", len(full_range.difference(df["date"])), 0),
        ("weekday sanaga mos emas", int((df["date"].dt.weekday != df["weekday"]).sum()), 0),
        ("month sanaga mos emas", int((df["date"].dt.month != df["month"]).sum()), 0),
        ("is_weekend weekday ga mos emas", int(((df["weekday"] >= 5) != df["is_weekend"]).sum()), 0),
        ("Manfiy natija qiymatlari", int((df[OUTCOMES] < 0).sum().sum()), 0),
        ("Namlik 0–100% dan tashqarida", int((~df["humidity"].between(0, 100)).sum()), 0),
        ("Qor, lekin harorat > 5°C", int((df["snow"] & (df["temp_c"] > 5)).sum()), 0),
        ("Bir kunda yomg'ir va qor", int((df["rain"] & df["snow"]).sum()), None),
    ]
    out = []
    for name, value, expected in checks:
        if expected is None:
            status = "ℹ️ ma'lumot uchun"
        else:
            status = "✅" if value == expected else "⚠️"
        out.append({"tekshiruv": name, "natija": value, "holat": status})
    return pd.DataFrame(out)
