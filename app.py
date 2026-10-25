import streamlit as st
import config
from src.ingest import build_index, get_collection
from src.pipeline import verify_text
from src.verifier import SUPPORTED, CONTRADICTED

st.set_page_config(page_title="Explainable Fact Checker", page_icon="🔎", layout="wide")
st.title("🔎 Explainable Fake News Detection & Claim Verification")
st.caption("Verdicts are based on NLI scores between claims and retrieved evidence—not LLM confidence.")

with st.sidebar:
    st.header("Trusted local knowledge base")
    st.caption("Uploading documents is optional when web search is enabled.")
    uploads = st.file_uploader(
        "Upload trusted reference documents (.txt / .md)",
        type=["txt", "md"], accept_multiple_files=True
    )
    if uploads:
        config.KB_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (config.KB_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploads)} document(s). Click Rebuild index.")

    if st.button("Rebuild local index", type="secondary"):
        with st.spinner("Embedding and indexing local documents..."):
            count = build_index()
        st.success(f"Indexed {count} passages.")

    try:
        st.metric("Local passages indexed", get_collection().count())
    except Exception:
        st.metric("Local passages indexed", 0)

    st.header("Evidence retrieval")
    use_web_search = st.checkbox("Search the web automatically", value=True)
    st.caption("Web search requires an internet connection. Search results are candidate evidence, not guaranteed-trustworthy sources.")

    st.header("Verification settings")
    top_k = st.slider("Passages per source type", 1, 10, config.TOP_K)
    entail_t = st.slider("Entailment threshold", 0.50, 0.95, config.ENTAIL_THRESHOLD, 0.05)
    contradict_t = st.slider("Contradiction threshold", 0.50, 0.95, config.CONTRADICT_THRESHOLD, 0.05)

text = st.text_area(
    "Enter a news article or claim. Prefix with 'Claim:' to verify one claim.",
    height=180,
    placeholder="Claim: Apollo 11 landed humans on the Moon in July 1969."
)

if st.button("Verify claim(s)", type="primary"):
    if not text.strip():
        st.warning("Enter a claim or article first.")
    else:
        with st.spinner("Extracting claims, retrieving local/web evidence, and running NLI..."):
            results = verify_text(
                text, top_k=top_k, entail_t=entail_t,
                contradict_t=contradict_t, use_web_search=use_web_search
            )

        if not results:
            st.warning("No claims could be extracted.")
        for i, result in enumerate(results, start=1):
            label = result.verdict.label
            color = "green" if label == SUPPORTED else "red" if label == CONTRADICTED else "orange"
            st.markdown(f"### Claim {i}")
            st.write(result.claim)
            st.markdown(f"**Prediction:** :{color}[{label}]")
            st.write(result.explanation)

            with st.expander(f"Retrieved evidence ({len(result.verdict.scored)})", expanded=True):
                if not result.verdict.scored:
                    st.info("No relevant passages were retrieved. This does not prove the claim is false.")
                for scored in result.verdict.scored:
                    ev = scored.evidence
                    st.markdown(f"**Source:** {ev.source}  ·  **Similarity:** {ev.similarity:.3f}  ·  **Stance:** {scored.stance}")
                    st.write(ev.text)
                    st.progress(float(max(0.0, min(scored.entail, 1.0))), text=f"Entailment: {scored.entail:.1%}")
                    st.progress(float(max(0.0, min(scored.contradict, 1.0))), text=f"Contradiction: {scored.contradict:.1%}")
                    st.progress(float(max(0.0, min(scored.neutral, 1.0))), text=f"Neutral: {scored.neutral:.1%}")
                    st.divider()
