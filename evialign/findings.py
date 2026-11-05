"""Canonical finding vocabulary shared by every dataset adapter, the report labeler,
and the evaluation code.

Each canonical finding has (i) regular-expression synonyms used by the rule-based
report labeler and (ii) an anatomical prior region used by the anatomical-plausibility
loss ("lungs", "heart", "mediastinum", or "any").
"""
from __future__ import annotations

FINDINGS: dict[str, dict] = {
    "atelectasis": {
        "patterns": [r"atelecta\w*", r"(lobar|segmental|subsegmental) collapse", r"collapse of the \w+ lobe"],
        "region": "lungs",
    },
    "cardiomegaly": {
        "patterns": [
            r"cardiomegaly",
            r"cardiac enlargement(?! (?:is )?(?:within normal|absent))",
            r"(?:heart|cardiac|cardiopericardial) (?:size |silhouette |shadow |contours? |borders? )?"
            r"(?:(?:is|are|was|were|remains?|appears?|has become|that is|which is|again|now|also|still|stable|unchanged) )*"
            r"(?:(?!not\b|no\b|without\b|nor\b)\w+ ){0,2}(?:enlarged|enlargement(?! (?:is )?(?:within normal|absent))|prominent|increased in size|bulky|globular|dilated)",
            r"(?:enlarged|enlargement of(?: the)?|enlarging|prominent|globular|dilated) (?:(?!not\b|no\b|without\b|nor\b)\w+ ){0,2}(?:heart|cardiac|cardiopericardial)\b",
            r"borderline (?:heart|cardiac)(?: size| silhouette)?",
        ],
        "region": "heart",
    },
    "consolidation": {"patterns": [r"consolidat\w*"], "region": "lungs"},
    "edema": {
        "patterns": [r"o?edema", r"vascular congestion", r"interstitial o?edema", r"fluid overload"],
        "region": "lungs",
    },
    "pleural_effusion": {
        "patterns": [r"effusions?", r"pleural fluid", r"blunting of the (\w+ )?costophrenic"],
        "region": "lungs",
    },
    "pneumonia": {"patterns": [r"pneumonia", r"bronchopneumonia", r"infectious process"], "region": "lungs"},
    "pneumothorax": {"patterns": [r"pneumothora\w*", r"\bptx\b"], "region": "lungs"},
    "nodule_mass": {
        "patterns": [r"nodules?", r"nodular opacit\w*", r"\bmass(es)?\b", r"tumou?r", r"neoplasm"],
        "region": "lungs",
    },
    "lung_opacity": {
        "patterns": [r"opacit(y|ies)", r"opacification", r"infiltrat\w*", r"air ?space (?:disease|opacit\w*|consolidation|process)",
                     r"haziness", r"ground[- ]glass", r"(?:increased|ill-defined|patchy|hazy|veiling|focal|subtle) (?:\w+ )?densit(?:y|ies)"],
        "region": "lungs",
    },
    "emphysema": {"patterns": [r"emphysema\w*", r"hyperinflat\w*", r"\bcopd\b"], "region": "lungs"},
    "fibrosis": {"patterns": [r"fibros\w*", r"fibrotic", r"\bscarring\b"], "region": "lungs"},
    "pleural_thickening": {"patterns": [r"pleural thickening", r"pleural plaques?"], "region": "lungs"},
    "hernia": {"patterns": [r"hernia\w*"], "region": "any"},
    "fracture": {"patterns": [r"fractures?"], "region": "any"},
}

NO_FINDING = "no_finding"
NORMAL_PATTERNS = [
    r"no acute (cardiopulmonary|intrathoracic) (abnormality|process|disease)",
    r"lungs are clear", r"normal chest", r"no active (cardiopulmonary )?disease",
]

# Default class list used for training when a dataset provides all of them.
DEFAULT_CLASSES = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "pleural_effusion",
    "pneumonia", "pneumothorax", "nodule_mass", "lung_opacity", "emphysema",
    "fibrosis", "pleural_thickening", "hernia",
]

# Short natural-language names, used by the template-explanation baseline.
DISPLAY = {
    "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly", "consolidation": "consolidation",
    "edema": "pulmonary edema", "pleural_effusion": "pleural effusion", "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax", "nodule_mass": "a nodule or mass", "lung_opacity": "a lung opacity",
    "emphysema": "emphysema", "fibrosis": "fibrosis", "pleural_thickening": "pleural thickening",
    "hernia": "a hernia", "fracture": "a fracture",
}
