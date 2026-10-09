"""Exports docs/data.json and docs/districts.geojson for the dashboard.

Run from the repo root (after prepare_reviews.py):  python scripts/build_dashboard_data.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
WIESN_END_2026 = pd.Timestamp("2026-10-04")   # keep in sync with the notebook's wiesn() rule
LANGS = {"de": "German", "en": "English", "fr": "French", "zh": "Chinese"}
THEMES = {"location": "Location", "transport": "Public transport", "host": "Host & hospitality",
          "cleanliness": "Cleanliness", "quiet": "Quiet", "noise_complaint": "Noise complaint",
          "heat": "Heat / air con", "oktoberfest": "Oktoberfest"}

listings = pd.read_csv(ROOT / "data/raw/listings.csv.gz")
calendar = pd.read_csv(ROOT / "data/raw/calendar.csv.gz", parse_dates=["date"])
reviews = pd.read_csv(ROOT / "data/processed/review_features.csv.gz", parse_dates=["date"])
listings["active"] = listings["number_of_reviews_ltm"] > 0

# Price quotes and the hedonic model (same specification as the notebook)
q = listings.dropna(subset=["price_quote_price_per_night", "review_scores_rating"]).copy()
q["checkin"] = pd.to_datetime(q["price_quote_checkin_date"])
q["nights"] = (pd.to_datetime(q["price_quote_checkout_date"]) - q["checkin"]).dt.days
q["ppn"] = q["price_quote_price_per_night"]
q = q[q["ppn"].between(15, 3000) & (q["nights"] <= 7) & (q["checkin"] < "2026-12-01")]
q["wiesn"] = q["checkin"] < WIESN_END_2026
q["weekend_checkin"] = q["checkin"].dt.dayofweek >= 4
q["bedrooms"] = q["bedrooms"].fillna(1)
q = q[q["room_type"].isin(["Entire home/apt", "Private room"])]
model = smf.ols("np.log(ppn) ~ wiesn + weekend_checkin + C(room_type) + np.log(accommodates) + bedrooms"
                " + C(neighbourhood_cleansed) + review_scores_rating + np.log1p(number_of_reviews) + nights",
                data=q).fit()
# Retransformation from log scale: multiply by the mean of exp(residuals) (Duan's smearing estimator)
smearing = float(np.mean(np.exp(model.resid)))
checks = q.sample(5, random_state=2)
reference = [{"district": r.neighbourhood_cleansed, "room_type": r.room_type, "guests": int(r.accommodates),
              "bedrooms": float(r.bedrooms), "rating": float(r.review_scores_rating), "reviews": int(r.number_of_reviews),
              "nights": int(r.nights), "wiesn": bool(r.wiesn), "weekend": bool(r.weekend_checkin),
              "expected": float(np.exp(model.predict(checks.loc[[i]]).iloc[0]))} for i, r in checks.iterrows()]

# District stats
active = listings[listings["active"]]
normal = q[~q["wiesn"]].groupby("neighbourhood_cleansed")["ppn"].median()
districts = pd.DataFrame({
    "active_listings": active.groupby("neighbourhood_cleansed").size(),
    "entire_share": active.assign(e=active["room_type"].eq("Entire home/apt")).groupby("neighbourhood_cleansed")["e"].mean(),
    "normal_price": normal,
}).fillna(0)

# Occupancy (active listings that aren't blocked all year)
cal = calendar[calendar["listing_id"].isin(active["id"])].copy()
cal["booked"] = cal["available"].eq("f")
blocked = cal.groupby("listing_id")["booked"].mean() > 0.98
cal = cal[~cal["listing_id"].isin(blocked[blocked].index)]
occ = cal.groupby("date")["booked"].mean()
occ = occ["2026-09-27":"2026-11-30"]

# Languages
core = reviews[reviews["lang"].isin(LANGS)]
rates = core.groupby("lang")[list(THEMES)].mean()
dated = reviews[reviews["lang"] != "short"]
mix = pd.crosstab(dated["date"].dt.year, dated["lang"], normalize="index").loc[2015:2026]

out = {
    "districts": {k: {c: round(float(v), 3) for c, v in row.items()} for k, row in districts.iterrows()},
    "occupancy": {"date": occ.index.strftime("%Y-%m-%d").tolist(), "share": occ.round(4).tolist(),
                  "wiesn_end": WIESN_END_2026.strftime("%Y-%m-%d")},
    "languages": {
        "names": LANGS, "themes": THEMES,
        "rates": {lang: {t: round(float(rates.loc[lang, t]), 4) for t in THEMES} for lang in LANGS},
        "counts": core["lang"].value_counts().to_dict(),
        "mix": {"year": mix.index.tolist(), **{l: mix[l].round(4).tolist() for l in ["en", "de", "fr", "es", "it", "zh"]}},
    },
    "model": {"params": model.params.round(6).to_dict(), "smearing": round(smearing, 4), "r2": round(model.rsquared, 3),
              "premium": round(float(np.exp(model.params["wiesn[T.True]"]) - 1), 3), "reference": reference,
              "rating_median": float(q["review_scores_rating"].median()),
              "reviews_median": int(q["number_of_reviews"].median())},
    "headline": {"listings": int(len(listings)), "active": int(listings["active"].sum()), "reviews": int(len(reviews))},
}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/data.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
shutil.copy(ROOT / "data/raw/neighbourhoods.geojson", ROOT / "docs/districts.geojson")
print(f"docs/data.json written | premium {out['model']['premium']:+.0%} | R² {model.rsquared:.2f}")
