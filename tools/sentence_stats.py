"""How often do two findings share the sentences that supervise their grounding?

The alignment target of finding c is the mean grounding of the tokens of the sentences that assert c (Section 3.4), so the
target is fixed by the set of asserting sentences. Two statistics are reported, which are not the same thing:

  exposed      at least one sentence asserting c also asserts another finding, so part of c's token set is shared
  identical    the set of sentences asserting c is exactly the set asserting some other finding, so the two targets coincide

Only the thirteen findings the classifier models are counted; the rule labeler also marks fracture, which is outside that
space and is reported separately.

python tools/sentence_stats.py --manifest manifests/rexgradient.csv --split train --n 20000 --out results/sentence_stats.csv
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
from collections import Counter

import pandas as pd
import yaml

from evialign.data import evidence_sentences
from evialign.findings import FINDINGS
from evialign.labeler import positive_sentence_spans

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--config", default="configs/main.yaml", help="supplies the modelled class list")
ap.add_argument("--out", default="results/sentence_stats.csv")
a = ap.parse_args()

ALL = list(FINDINGS)
try:
    CLASSES = yaml.safe_load(open(a.config))["data"]["classes"]
except Exception:
    CLASSES = [c for c in ALL if c != "fracture"]
EXTRA = [c for c in ALL if c not in CLASSES]

df = pd.read_csv(a.manifest, low_memory=False)
df = df[df["split"] == a.split] if "split" in df.columns else df
df = df.drop_duplicates("study_id")
df = df[df["report"].fillna("").str.len() > 0].sample(n=min(a.n, len(df)), random_state=0)

n_sent = Counter()
pair_total = pair_exposed = pair_identical = 0
per_finding = Counter()
per_exposed = Counter()
per_identical = Counter()
cooccur = Counter()
reports_exposed = reports_identical = 0
extra_pairs = 0

for text in df["report"]:
    target = evidence_sentences(text, ALL)
    spans = positive_sentence_spans(target, ALL)
    extra_pairs += sum(1 for c in EXTRA if spans[c])
    spans = {c: sp for c, sp in spans.items() if c in CLASSES}
    by_span = {}
    for c, sp in spans.items():
        for s in sp:
            by_span.setdefault(s, []).append(c)
    for s, cs in by_span.items():
        n_sent[len(cs)] += 1
        if len(cs) > 1:
            for i, c in enumerate(cs):
                for d in cs[i + 1:]:
                    cooccur[tuple(sorted((c, d)))] += 1
    present = [c for c, sp in spans.items() if sp]
    sets = {c: frozenset(spans[c]) for c in present}
    exposed = identical = False
    for c in present:
        pair_total += 1
        per_finding[c] += 1
        if any(len(by_span[s]) > 1 for s in spans[c]):
            pair_exposed += 1
            per_exposed[c] += 1
            exposed = True
        if any(d != c and sets[d] == sets[c] for d in present):
            pair_identical += 1
            per_identical[c] += 1
            identical = True
    reports_exposed += int(exposed)
    reports_identical += int(identical)

q = lambda n, d: round(n / max(1, d), 4)
rows = [{"quantity": "reports", "value": len(df)},
        {"quantity": "sentences asserting one modelled finding", "value": n_sent[1]},
        {"quantity": "sentences asserting more than one modelled finding", "value": sum(v for k, v in n_sent.items() if k > 1)},
        {"quantity": "share of asserting sentences with more than one finding",
         "value": q(sum(v for k, v in n_sent.items() if k > 1), sum(n_sent.values()))},
        {"quantity": "alignment pairs", "value": pair_total},
        {"quantity": "pairs exposed to multi-finding sentences", "value": pair_exposed},
        {"quantity": "share exposed", "value": q(pair_exposed, pair_total)},
        {"quantity": "pairs whose sentence set is identical to another finding's", "value": pair_identical},
        {"quantity": "share identical", "value": q(pair_identical, pair_total)},
        {"quantity": "reports with at least one exposed pair", "value": reports_exposed},
        {"quantity": "reports with at least one identical pair", "value": reports_identical},
        {"quantity": "labeler-only findings outside the modelled classes (pairs)", "value": extra_pairs},
        {"quantity": "classes outside the modelled space", "value": ";".join(EXTRA) or "none"}]
for c in CLASSES:
    if per_finding[c]:
        rows.append({"quantity": f"exposed share, {c}", "value": q(per_exposed[c], per_finding[c])})
        rows.append({"quantity": f"identical share, {c}", "value": q(per_identical[c], per_finding[c])})
for (c, d), v in cooccur.most_common(8):
    rows.append({"quantity": f"co-asserted in one sentence, {c} + {d}", "value": v})

out = pd.DataFrame(rows)
_os.makedirs(_os.path.dirname(a.out) or ".", exist_ok=True)
out.to_csv(a.out, index=False)
print(out.to_string(index=False))
