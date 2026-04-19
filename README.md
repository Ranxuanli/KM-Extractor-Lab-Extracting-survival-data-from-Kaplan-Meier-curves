# KM Extractor Lab — High-score Version for Option 2

This project is a polished website:

> Design a website that uses LLM APIs to extract event times and event indicators from Kaplan–Meier curves, and then computes log-rank test results based on the extracted data. For extra credit, identify biomedical articles comparing treatment A versus B and treatment B versus C, and use your tool to indirectly compare treatment A versus C.

## What is included

This version goes beyond a minimal demo and is designed to look like a complete methods prototype.

### Main workflow page

The main page supports:

1. **Live extraction from a user-uploaded Kaplan–Meier image**
2. **Cached demo mode** for safe screenshots without any API key
3. **Automatic JSON caching** of extracted results
4. **Approximate pseudo-IPD reconstruction** from extracted KM step curves
5. **Approximate log-rank test**
6. **Approximate Cox proportional-hazards model** to derive a pairwise hazard ratio and 95% CI
7. **A cache gallery** showing stored runs and snapshots
8. **A polished explanation layer** describing workflow, assumptions, and limitations

### Extra-credit page

The extra-credit page supports:

1. A dedicated **indirect comparison dashboard**
2. Two cached pairwise demo studies: **A vs B** and **B vs C**
3. Approximate pairwise hazard-ratio estimation from reconstructed pseudo survival data
4. **Bucher-style anchored indirect comparison** for **A vs C**
5. A **manual calculator** where you can type in your own log(HR) and SE values
6. Reporting text you can reuse in your final write-up

## Why cached mode matters

The course instruction explicitly says **do not upload API keys** and instead demonstrate the site with **cached runs and snapshots**.

This project is built around that requirement:

- each run can be saved as JSON in `static/cache/`
- rendered plots are stored in `static/plots/`
- the cached demo mode works with no API key at all

That means you can run the website, click the demo button, and take screenshots that clearly prove the site works.

## Statistical note

This website performs an **approximate** reconstruction. A Kaplan–Meier image usually does not reveal:

- the exact censoring pattern
- the complete risk table
- the full original event-time data

Therefore:

- the pairwise log-rank test is approximate
- the Cox hazard ratio is approximate
- the indirect A–C comparison is approximate

For a course project, that is acceptable as long as you describe it honestly as a proof-of-concept or computational demo.

## Project structure

```text
km_llm_website/
├── app.py
├── README.md
├── requirements.txt
├── data/
├── uploads/
├── static/
│   ├── style.css
│   ├── cache/
│   └── plots/
└── templates/
    ├── base.html
    ├── index.html
    └── indirect.html
```

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"   # optional, only needed for live extraction
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Recommended demo sequence for submission

### Minimum safe route

1. Open the homepage
2. Click **Load cached demo run**
3. Screenshot the resulting analysis cards and extracted JSON
4. Open the **Extra credit: indirect comparison** page
5. Screenshot the indirect A–C estimate and the manual calculator
6. Optionally show the cache gallery or one cached JSON file

This directly satisfies the request for **cached runs and snapshots**.

## Suggested wording for your report

You can adapt the paragraph below:

> We built a Flask-based website that uses a multimodal LLM API to extract two Kaplan–Meier curves from uploaded biomedical figures. The extracted curves are normalized into structured JSON and cached for reproducible demonstration without exposing API credentials. The website then reconstructs approximate pseudo individual patient data from the extracted step functions and computes an approximate log-rank test and Cox proportional-hazards estimate. In addition, the site contains an extra-credit module for indirect comparison, combining reconstructed A–B and B–C evidence to derive an anchored A–C hazard-ratio estimate under Bucher-style assumptions. Because censoring patterns and complete risk tables are not fully recoverable from image data alone, all inferential results are reported as approximate and proof-of-concept.

## How to turn this into an even stronger final submission

If you want to go further, you can extend this code to include:

- OCR of number-at-risk tables under the KM plot
- explicit extraction of censoring tick marks
- article metadata fields such as PMID / DOI / journal / year
- multi-study comparison panels
- forest-plot rendering for the indirect-comparison module

