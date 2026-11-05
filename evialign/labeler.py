"""Transparent rule-based finding-mention labeler (NegEx-style).

It is used for three purposes:
  1. weak finding labels for report-only datasets (optional; CheXbert can replace it),
  2. sentence/token masks that tell the alignment loss which tokens of a report talk about
     which finding,
  3. extracting the findings asserted by *generated* explanations at evaluation time
     (prediction-explanation consistency, hallucination and omission rates).

Status codes: 1 = positive mention, 0 = negated, -1 = uncertain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .findings import FINDINGS, NORMAL_PATTERNS

PRE_NEG = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\bfree of\b", r"\bclear of\b",
    r"\babsence of\b", r"\bno (evidence|sign|signs) of\b", r"\brules? out\b", r"\bruled out\b",
    r"\bnor\b", r"\bresolution of\b", r"\bresolved\b",
]
POST_NEG = [r"\b(is|are|was|were) (not|no longer) (seen|present|identified|evident|visualized)\b",
            r"\bhas resolved\b", r"\bhave resolved\b", r"\b(is|are) absent\b", r"\bresolved\b"]
UNCERTAIN = [r"\bmay\b", r"\bmight\b", r"\bpossible\b", r"\bpossibly\b", r"\bprobable\b", r"\blikely\b",
             r"\bcannot be (excluded|ruled out)\b", r"\bcan not be (excluded|ruled out)\b",
             r"\bsuggest\w*\b", r"\bquestionable\b", r"\bversus\b", r"\bvs\.?\b", r"\bconcern for\b",
             r"\bcould (represent|reflect)\b", r"\?", r"\bprobabl[ey]\b", r"\bborderline\b", r"\bsuspicio\w+\b", r"\bsuspect\w*\b"]
PSEUDO_NEG = [r"\bno (significant )?(change|interval change|increase|decrease)\b", r"\bnot only\b",
              r"\bwithout (significant )?change\b", r"\bno new\b"]
SCOPE_END = r"\b(but|however|although|though|which|aside from|apart from|except)\b|[;:]"

_FIND_RE = {k: re.compile("|".join(f"(?:{p})" for p in v["patterns"]), re.I) for k, v in FINDINGS.items()}
_PRE = re.compile("|".join(PRE_NEG), re.I)
_POST = re.compile("|".join(POST_NEG), re.I)
_UNC = re.compile("|".join(UNCERTAIN), re.I)
_PSEUDO = re.compile("|".join(PSEUDO_NEG), re.I)
_SCOPE = re.compile(SCOPE_END, re.I)
_NORMAL = re.compile("|".join(NORMAL_PATTERNS), re.I)
_SENT = re.compile(r"[^.!?\n]+[.!?]?", re.M)


@dataclass
class Mention:
    finding: str
    status: int          # 1 positive, 0 negated, -1 uncertain
    sent_start: int      # character span of the sentence containing the mention
    sent_end: int


def _status(sentence: str, m_start: int, m_end: int) -> int:
    before = sentence[:m_start]
    # restrict pre-negation scope to the clause that contains the mention
    cuts = [c.end() for c in _SCOPE.finditer(before)]
    clause = before[cuts[-1]:] if cuts else before
    clause_wo_pseudo = _PSEUDO.sub(" ", clause)
    after = sentence[m_end:]
    after_cut = _SCOPE.search(after)
    after_clause = after[: after_cut.start()] if after_cut else after
    if _PRE.search(clause_wo_pseudo) or _POST.search(after_clause):
        return 0
    if _UNC.search(clause) or _UNC.search(after_clause):
        return -1
    return 1


def extract_mentions(text: str) -> list[Mention]:
    out: list[Mention] = []
    if not isinstance(text, str) or not text.strip():
        return out
    for sm in _SENT.finditer(text):
        s, e = sm.start(), sm.end()
        sent = text[s:e]
        for finding, rx in _FIND_RE.items():
            best = None
            for m in rx.finditer(sent):
                st = _status(sent, m.start(), m.end())
                # a positive mention dominates an uncertain one, which dominates a negated one
                rank = {1: 2, -1: 1, 0: 0}[st]
                if best is None or rank > best[0]:
                    best = (rank, st)
            if best is not None:
                out.append(Mention(finding, best[1], s, e))
    return out


def label_report(text: str, classes: list[str], uncertain_as: int = 1) -> list[int]:
    """Report-level labels in {1, 0, -1(missing)}.

    A finding is positive if any sentence mentions it positively; uncertain mentions are mapped to
    ``uncertain_as`` (U-Ones by default, following CheXpert practice); findings never mentioned are 0.
    """
    status = {c: None for c in classes}
    for m in extract_mentions(text):
        if m.finding not in status:
            continue
        v = m.status if m.status != -1 else uncertain_as
        cur = status[m.finding]
        status[m.finding] = v if cur is None else max(cur, v)
    return [0 if status[c] is None else int(status[c]) for c in classes]


def is_normal(text: str) -> bool:
    return bool(_NORMAL.search(text or ""))


def positive_sentence_spans(text: str, classes: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Character spans of the sentences that positively assert each finding."""
    spans: dict[str, list[tuple[int, int]]] = {c: [] for c in classes}
    for m in extract_mentions(text):
        if m.finding in spans and m.status == 1:
            spans[m.finding].append((m.sent_start, m.sent_end))
    return spans


if __name__ == "__main__":
    demo = ("Moderate right pleural effusion with adjacent atelectasis. No pneumothorax. "
            "Heart size is enlarged. Possible left lower lobe pneumonia. No change in the left basilar opacity.")
    for m in extract_mentions(demo):
        print(m)
