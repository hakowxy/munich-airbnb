"""Turns the raw review export into a compact, text-free feature table.

Detects each review's language and flags which themes it mentions, then writes
data/processed/review_features.csv.gz. The raw reviews.csv.gz (28 MB, with reviewer
names) stays out of the repo: download it from insideairbnb.com to rebuild.

Run from the repo root:  python scripts/prepare_reviews.py
"""
import re
from pathlib import Path

import pandas as pd
import py3langid as langid

ROOT = Path(__file__).resolve().parent.parent

# Keyword stems per theme and language. Chinese has no word boundaries, so it is matched as substrings.
THEMES = {
    "location": {
        "en": r"\blocation|located|central\b|close to|walking distance",
        "de": r"\blage\b|zentral|gelegen|nähe|fußläufig",
        "fr": r"emplacement|situé|central|proche|à pied",
        "zh": r"位置|地段|交通便利|离.{0,4}近",
    },
    "cleanliness": {
        "en": r"\bclean|spotless|tidy",
        "de": r"sauber|reinlich|gepflegt",
        "fr": r"propre|propreté|impeccable",
        "zh": r"干净|整洁|卫生",
    },
    "host": {
        "en": r"\bhost|responsive|helpful|friendly",
        "de": r"gastgeber|vermieter|freundlich|hilfsbereit",
        "fr": r"\bhôte|accueil|sympathique|réactif",
        "zh": r"房东|热情|友好|回复",
    },
    "transport": {
        "en": r"\bu-?bahn|s-?bahn|subway|metro|tram|train station|public transport",
        "de": r"\bu-?bahn|s-?bahn|tram|straßenbahn|bahnhof|öffentlich",
        "fr": r"\bu-?bahn|s-?bahn|métro|tram|gare|transports? en commun",
        "zh": r"地铁|轻轨|电车|火车站|公交",
    },
    "quiet": {
        "en": r"\bquiet|peaceful",
        "de": r"\bruhig",
        "fr": r"\bcalme|paisible|tranquille",
        "zh": r"安静|清静",
    },
    "noise_complaint": {
        "en": r"(?<!no )(?<!not )(?<!not too )\bnois[ey]|(?<!not )\bloud\b|thin walls",
        "de": r"lärm|(?<!nicht )(?<!nicht zu )\blaut\b(?!\s+(?:beschreibung|anzeige|inserat|bildern|fotos|angabe))|hellhörig",
        "fr": r"(?<!pas de )(?<!aucun )(?<!sans )\bbruit|(?<!pas )\bbruyant",
        "zh": r"(?<!不)(?<!没有)吵|(?<!没有)噪音",
    },
    "heat": {
        "en": r"air ?con|aircon|\ba/?c\b|too hot|very hot|heat ?wave|stuffy|\bfans?\b(?! of)",
        "de": r"klimaanlage|zu warm|zu heiß|sehr heiß|hitze|ventilator|stickig",
        "fr": r"climatisation|\bclim\b|trop chaud|très chaud|canicul|ventilateur",
        "zh": r"空调|太热|很热(?!情|心)|闷热|风扇",
    },
    "oktoberfest": {
        "en": r"oktoberfest|wiesn",
        "de": r"oktoberfest|wiesn",
        "fr": r"oktoberfest|wiesn|fête de la bière",
        "zh": r"啤酒节|oktoberfest",
    },
}
LANGUAGES = ["en", "de", "fr", "zh"]
COMPILED = {t: {lang: re.compile(p, re.IGNORECASE) for lang, p in by_lang.items()} for t, by_lang in THEMES.items()}


def main():
    reviews = pd.read_csv(ROOT / "data/raw/reviews.csv.gz", usecols=["listing_id", "id", "date", "comments"])
    text = reviews["comments"].fillna("").astype(str).str.replace(r"<br\s*/?>", " ", regex=True)
    reviews["chars"] = text.str.len()
    reviews["lang"] = [langid.classify(t)[0] if len(t.strip()) >= 15 else "short" for t in text]
    for theme, patterns in COMPILED.items():
        reviews[theme] = [bool(patterns[lang].search(t)) if lang in patterns else False
                          for t, lang in zip(text, reviews["lang"])]
    out = reviews.drop(columns=["comments"])
    out.to_csv(ROOT / "data/processed/review_features.csv.gz", index=False)
    print(f"{len(out):,} reviews processed")
    print(out["lang"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
