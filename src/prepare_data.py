from __future__ import annotations

import argparse
import json
import random
import re

import pandas as pd

from .utils import DATA

SEED = 42
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
N_TRAIN, N_VAL, N_TEST, N_HISTORY = 200, 40, 40, 8000

PRIORITIES = ["low", "medium", "high", "urgent"]
BASE_PRIORITY = {
    "payment_issue": "high", "complaint": "high", "get_refund": "medium", "track_refund": "medium",
    "cancel_order": "medium", "change_order": "medium", "change_shipping_address": "medium",
    "track_order": "medium", "recover_password": "medium", "registration_problems": "medium",
    "delete_account": "medium", "contact_human_agent": "medium",
}
URGENCY_EN = re.compile(
    r"\b(urgent|urgently|asap|immediately|right now|now!|at once|emergency|furious|"
    r"unacceptable|ridiculous|worst|scam|fraud|lawyer|sue)\b|!!", re.IGNORECASE)
URGENCY_AR = re.compile(r"(عاجل|فورا|فوراً|حالا|حالاً|بسرعة|الحين|!!|شي ما يصير|سيئة)")


def priority_for(intent: str, text: str, offensive: bool = False) -> str:
    level = PRIORITIES.index(BASE_PRIORITY.get(intent, "low"))
    if offensive or URGENCY_EN.search(text) or URGENCY_AR.search(text):
        level = min(level + 1, len(PRIORITIES) - 1)
    return PRIORITIES[level]


CITIES = ["Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Abha", "Tabuk", "Khobar"]
COUNTRIES = ["Saudi Arabia", "UAE", "Kuwait", "Bahrain", "Oman", "Qatar"]
NAMES = ["Sara", "Fahad", "Noura", "Abdullah", "Reem", "Khalid", "Lama", "Omar", "Hessa", "Yousef"]
ACCOUNT_TYPES = ["Premium", "Business", "Gold", "Standard", "Platinum", "Basic"]


def fill_message_placeholders(text: str, rng: random.Random, values: dict) -> str:
    gens = {
        "order number": lambda: f"A-{rng.randint(10000, 99999)}",
        "invoice number": lambda: f"INV-2024-{rng.randint(1, 99999):05d}",
        "account type": lambda: rng.choice(ACCOUNT_TYPES),
        "account category": lambda: rng.choice(ACCOUNT_TYPES),
        "person name": lambda: rng.choice(NAMES),
        "delivery city": lambda: rng.choice(CITIES),
        "delivery country": lambda: rng.choice(COUNTRIES),
        "refund amount": lambda: str(rng.choice([49, 120, 199, 250, 349, 799, 1250])),
        "currency symbol": lambda: "SAR",
    }

    def repl(m: re.Match) -> str:
        key = m.group(1).strip().lower()
        if key not in values and key in gens:
            values[key] = gens[key]()
        return values.get(key, m.group(1))

    return re.sub(r"\{\{([^}]+)\}\}", repl, text)


RESPONSE_VALUES = {
    "online order interaction": "order management page", "customer support phone number": "920000000",
    "website url": "demomart.example", "customer support hours": "8:00 to 22:00, Saturday to Thursday",
    "online company portal info": "DemoMart account", "tracking number": "the tracking number in your confirmation SMS",
    "date range": "the last 30 days", "client last name": "", "salutation": "", "settings": "Settings",
    "profile": "Profile", "account change": "account change", "upgrade account": "Upgrade account",
    "login page url": "demomart.example/login", "store location": "the nearest pickup point",
    "forgot password": "Forgot password", "order status": "Order status",
    "customer support email": "support@demomart.example", "profile type": "account type",
    "order tracking": "Order tracking", "client name": "", "client first name": "", "client full name": "",
    "account recovery page url": "demomart.example/recover", "money amount": "the amount",
    "company name": "DemoMart", "company": "DemoMart", "live chat support": "live chat",
}


def fill_response_placeholders(text: str, values: dict) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1).strip().lower()
        if key in values:
            return values[key]
        if key in RESPONSE_VALUES:
            return RESPONSE_VALUES[key]
        return m.group(1).strip()

    out = re.sub(r"\{\{([^}]+)\}\}", repl, text)
    return re.sub(r"[ \t]{2,}", " ", out).replace(" ,", ",")


def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\b(a|inv)-[\d-]+\b", " <id> ", t)
    t = re.sub(r"\d+", " <n> ", t)
    t = re.sub(r"\b(" + "|".join(n.lower() for n in NAMES + CITIES + ACCOUNT_TYPES) + r")\b", " <x> ", t)
    t = re.sub(r"[^\w<>\u0600-\u06FF]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stratified_take(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(n_per_class, len(g)), random_state=seed) for _, g in df.groupby("intent")]
    return pd.concat(parts).reset_index(drop=True)


def load_raw(synthetic: bool) -> pd.DataFrame:
    if synthetic:
        return synthetic_raw()
    from datasets import load_dataset

    print(f"Downloading {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID, split="train").to_pandas().rename(columns={"instruction": "text"})
    return raw


def synthetic_raw(per_intent: int = 450) -> pd.DataFrame:
    intents = sorted(pd.read_csv(DATA / "arabic_tickets.csv").intent.unique())
    rng = random.Random(SEED)
    filler = ["please", "help", "urgent", "today", "quick", "thanks", "hello", "need", "my", "order", "now"]
    rows = []
    for i, it in enumerate(intents):
        cat = it.split("_")[-1].upper()
        for j in range(per_intent):
            words = it.replace("_", " ").split() + rng.sample(filler, 3)
            rng.shuffle(words)
            rows.append({"text": " ".join(words) + f" ref{j} {{{{Order Number}}}}", "intent": it, "category": cat,
                         "flags": "B", "response": f"Thanks. Regarding {it.replace('_', ' ')}: see help center."})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="fake data for smoke tests only")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    raw = load_raw(args.synthetic)
    n_raw = len(raw)

    raw["text"] = raw["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)

    rng = random.Random(SEED)
    texts, responses = [], []
    for t, r in zip(raw["text"], raw["response"]):
        values: dict = {}
        texts.append(fill_message_placeholders(t, rng, values))
        responses.append(fill_response_placeholders(r, values))
    raw["text"], raw["response"] = texts, responses

    raw["norm"] = raw["text"].map(normalise)
    before = len(raw)
    raw = raw.drop_duplicates("norm").reset_index(drop=True)
    print(f"Near-duplicate removal: {before} -> {len(raw)} tickets")

    raw["offensive"] = raw["flags"].str.contains("W")
    raw["priority"] = [priority_for(i, t, o) for i, t, o in zip(raw["intent"], raw["text"], raw["offensive"])]
    raw["lang"] = "en"
    raw["ticket_id"] = range(1, len(raw) + 1)

    test = stratified_take(raw, N_TEST, SEED)
    rest = raw[~raw.ticket_id.isin(test.ticket_id)]
    val = stratified_take(rest, N_VAL, SEED)
    rest = rest[~rest.ticket_id.isin(val.ticket_id)]
    train = stratified_take(rest, N_TRAIN, SEED)
    rest = rest[~rest.ticket_id.isin(train.ticket_id)]
    history = rest.sample(n=min(N_HISTORY, len(rest)), random_state=SEED)

    cols = ["ticket_id", "text", "intent", "category", "priority", "lang"]
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df[cols].sample(frac=1, random_state=SEED).to_csv(DATA / f"tickets_{name}.csv", index=False)
    history[cols + ["response"]].to_csv(DATA / "tickets_history.csv", index=False)

    intent2cat = raw.groupby("intent")["category"].first().to_dict()
    ar = pd.read_csv(DATA / "arabic_tickets.csv")
    hard = pd.read_csv(DATA / "hard_en_tickets.csv")
    for name, df in [("arabic_tickets.csv", ar), ("hard_en_tickets.csv", hard)]:
        unknown = set(df["intent"]) - set(intent2cat)
        assert not unknown, f"Unknown intents in {name}: {unknown}"
    ar.insert(0, "ticket_id", range(900001, 900001 + len(ar)))
    ar["category"] = ar["intent"].map(intent2cat)
    ar["priority"] = [priority_for(i, t) for i, t in zip(ar["intent"], ar["text"])]
    ar["lang"] = "ar"
    ar[cols + ["dialect"]].to_csv(DATA / "tickets_ar_test.csv", index=False)

    hard.insert(0, "ticket_id", range(800001, 800001 + len(hard)))
    hard["category"] = hard["intent"].map(intent2cat)
    hard["priority"] = [priority_for(i, t) for i, t in zip(hard["intent"], hard["text"])]
    hard["lang"] = "en"
    hard["dialect"] = "hard_en"
    hard[cols + ["dialect"]].to_csv(DATA / "tickets_hard_test.csv", index=False)

    labels = {
        "intents": sorted(intent2cat), "categories": sorted(set(intent2cat.values())), "priorities": PRIORITIES,
        "intent_to_category": dict(sorted(intent2cat.items())),
    }
    (DATA / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    sets = {n: set(d["text"].map(normalise)) for n, d in [("train", train), ("val", val), ("test", test)]}
    overlap = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    report = {
        "synthetic": args.synthetic, "raw_rows": n_raw, "after_near_dup_removal": len(raw),
        "sizes": {"train": len(train), "val": len(val), "test": len(test), "history": len(history),
                  "ar_test": len(ar), "hard_test": len(hard)},
        "min_train_per_intent": int(train.intent.value_counts().min()),
        "min_test_per_intent": int(test.intent.value_counts().min()),
        "masked_sentence_overlap": overlap,
    }
    (DATA / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert all(v == 0 for v in overlap.values()), f"leakage detected: {overlap}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
