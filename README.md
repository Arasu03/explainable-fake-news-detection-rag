# Explainable Fake News Detection & Claim Verification

A local evidence-based fact-checking prototype. It retrieves passages from trusted `.txt` and `.md` documents, then uses a 3-way Natural Language Inference (NLI) Transformer to estimate entailment, contradiction, and neutral scores.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection on first run to download the embedding and NLI models
- CPU works; first model load can take several minutes

## Run on Windows / VS Code
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and use:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Use
1. Open the app in the browser.
2. Upload trusted `.txt` or `.md` reference documents in the sidebar.
3. Click **Rebuild index**.
4. Enter a factual claim and click **Verify claim(s)**.
5. Review the verdict, retrieved passages, source names, similarity, and NLI scores.

## Important limitations
- The initial claim extractor uses simple sentence splitting, not a trained claim-extraction model.
- This searches only documents you add to the local knowledge base; it does not automatically browse the internet.
- NLI scores are model estimates, not calibrated probabilities that a claim is objectively true.
- Source reliability and evidence coverage matter. `INSUFFICIENT EVIDENCE` means the system could not make a confident decision from retrieved passages, not that the claim is false.
- Verify model label order and behavior when swapping the NLI model.

## Hybrid retrieval update

The app can now combine:
- Local evidence from `.txt` and `.md` documents indexed in ChromaDB.
- Optional public-web search snippets, enabled by default in the sidebar.

Install the updated dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

No web-search API key is required for the `ddgs` metasearch library, but results can be temporarily unavailable and may vary. Treat search snippets as candidate evidence. Verify source credibility, dates, and context before trusting a verdict. The NLI model scores whether a passage supports or contradicts a claim; it does not independently establish source reliability. Claim extraction is still a basic sentence splitter.
