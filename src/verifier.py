from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT EVIDENCE"

@dataclass
class ScoredEvidence:
    evidence: object
    entail: float
    contradict: float
    neutral: float
    stance: str

@dataclass
class Verdict:
    label: str
    explanation: str
    scored: list

_tokenizer = None
_model = None
_label_map = None

def _load_model():
    global _tokenizer, _model, _label_map
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
        _model.eval()
        raw = _model.config.id2label
        _label_map = {}
        for idx, name in raw.items():
            name = str(name).lower()
            if "entail" in name:
                _label_map["entailment"] = int(idx)
            elif "contrad" in name:
                _label_map["contradiction"] = int(idx)
            elif "neutral" in name:
                _label_map["neutral"] = int(idx)
        # Some checkpoints use LABEL_0 etc.; the known nli-deberta-v3-small order is
        # contradiction, entailment, neutral. Keep this fallback explicit.
        if not all(k in _label_map for k in ("entailment", "contradiction", "neutral")):
            _label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return _tokenizer, _model, _label_map

def score_pair(premise, hypothesis):
    tokenizer, model, label_map = _load_model()
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
    return (
        float(probs[label_map["entailment"]]),
        float(probs[label_map["contradiction"]]),
        float(probs[label_map["neutral"]])
    )

def verify_claim(claim, evidence_list, entail_t=0.70, contradict_t=0.70):
    scored = []
    for evidence in evidence_list:
        # NLI premise = retrieved evidence; hypothesis = claim.
        entail, contradict, neutral = score_pair(evidence.text, claim)
        if entail >= entail_t and entail > contradict:
            stance = SUPPORTED
        elif contradict >= contradict_t and contradict > entail:
            stance = CONTRADICTED
        else:
            stance = INSUFFICIENT
        scored.append(ScoredEvidence(evidence, entail, contradict, neutral, stance))

    # A single strong piece of evidence can decide the result. If strong evidence
    # points both ways, do not silently pick one side.
    has_support = any(s.stance == SUPPORTED for s in scored)
    has_contradiction = any(s.stance == CONTRADICTED for s in scored)
    if has_support and has_contradiction:
        label = INSUFFICIENT
        explanation = "Retrieved passages conflict: some support the claim while others contradict it. Review the sources before deciding."
    elif has_support:
        label = SUPPORTED
        best = max((s for s in scored if s.stance == SUPPORTED), key=lambda s: s.entail)
        explanation = f"The retrieved passage from '{best.evidence.source}' supports the claim (entailment {best.entail:.1%}). Check that the source itself is trustworthy."
    elif has_contradiction:
        label = CONTRADICTED
        best = max((s for s in scored if s.stance == CONTRADICTED), key=lambda s: s.contradict)
        explanation = f"The retrieved passage from '{best.evidence.source}' contradicts the claim (contradiction {best.contradict:.1%}). Check that the source itself is trustworthy."
    else:
        label = INSUFFICIENT
        explanation = "The retrieved evidence does not cross the configured support or contradiction threshold. This is not proof that the claim is false."

    return Verdict(label=label, explanation=explanation, scored=scored)
