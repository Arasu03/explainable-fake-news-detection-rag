import re
from dataclasses import dataclass

from src.ingest import retrieve_evidence
from src.web_search import search_web_evidence
from src.verifier import verify_claim

@dataclass
class ClaimResult:
    claim: str
    verdict: object
    explanation: str

def extract_claims(text):
    text = re.sub(r"^\\s*Claim\\s*:\\s*", "", text.strip(), flags=re.IGNORECASE)
    # Basic baseline sentence splitting; complex articles may need a stronger
    # claim extraction model.
    parts = re.split(r"(?<=[.!?])\\s+", text)
    claims = []
    for part in parts:
        part = part.strip(" \\n\\t\\\"'")
        if len(part.split()) >= 4:
            claims.append(part)
    return claims

def verify_text(text, top_k=4, entail_t=0.70, contradict_t=0.70, use_web_search=True):
    results = []
    for claim in extract_claims(text):
        # Search local trusted documents first.
        evidence = retrieve_evidence(claim, top_k=top_k)

        # Optionally add public-web snippets to the local evidence.
        if use_web_search:
            web_evidence = search_web_evidence(claim, top_k=top_k)
            known = {(item.source, item.text) for item in evidence}
            evidence.extend(
                item for item in web_evidence
                if (item.source, item.text) not in known
            )

        verdict = verify_claim(
            claim, evidence, entail_t=entail_t, contradict_t=contradict_t
        )
        results.append(
            ClaimResult(claim=claim, verdict=verdict, explanation=verdict.explanation)
        )
    return results
