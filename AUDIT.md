# Phase 0: Project Audit & Findings

## 1. System & Environment
- **Operating System**: Windows (AMD64)
- **Active Python**: Python 3.14.7 (located in `.venv/Scripts/python.exe`)
- **Torch & Sentence-Transformers**: `torch 2.14.0+cpu` and `sentence-transformers 6.0.1` are fully functional in Python 3.14.7 with pre-downloaded weights for `all-MiniLM-L6-v2` producing 384-dimensional embeddings.
- **Additional Libraries**: `streamlit 1.66.1`, `plotly 7.1.0`, `scikit-learn 1.9.1`, `scipy 1.18.1`, `pandas 3.0.5`, `numpy 2.5.3`, `jsonschema 4.26.0`.

## 2. Inventory of Repo Files & Health Assessment

| File / Path | Current Purpose | Status & Issues Identified |
|---|---|---|
| `.gitignore` | Ignore list for git | Incomplete: ignores all `.csv` and `.json` files by default or fails to explicitly preserve dataset and report CSVs; `.venv` and `__pycache__` had artifacts tracked in git history. |
| `requirements.txt` | Dependency specifications | Missing `sentence-transformers`, `torch`, `jsonschema`, `streamlit`, `plotly`. Only had 6 baseline dependencies. |
| `shared/column_profile_schema.json` | JSON Schema for column profile contracts | Minimal draft schema. `value_profile` is marked as generic `object` without property constraints. |
| `shared/contracts.py` | Schema validation function | Bare script with hardcoded test object. Prints non-ASCII character (`\u2713`) that crashes Windows cp1252 consoles with `UnicodeEncodeError`. Missing formal dataclasses for typed profile exchange. |
| `person_a/normalize_names.py` | Column name tokenizer and abbreviation expander | Passes 6 existing tests, but abbreviation list is narrow (26 entries). Tokenization lacks acronym handling like `CustID` or dotted notations. |
| `person_a/name_embedder.py` | SentenceTransformer wrapper for token embeddings | Works, but instantiates model per class call (not a lazy singleton), prints `\u2713` crashing Windows cp1252 consoles, uses deprecated `get_sentence_embedding_dimension` method, lacks offline fallback (TF-IDF char n-gram). |
| `person_a/name_similarity.py` | Cosine similarity between embeddings | Uses O(N*M) Python loops for matrix computation rather than vectorized matrix dot products. Rescales cosine `[-1, 1]` to `[0, 1]` via `(x+1)/2` rather than standard clipping. |
| `person_a/fusion.py` | Adaptive weighting between name and value signals | Incomplete and untuned: `alpha` is not profile-aware (does not inspect null ratios, sample counts, or dtype agreement). Overly penalizes signal disagreement by resetting alpha to 0.5. Does not return structured dataclass with `reasons` or explanations. |
| `person_a/mock_value_profiles.py` | Mock profile vectors for person_a testing | Broken contract: generates hand-crafted 21-dimensional vectors with arbitrary index placements, completely incompatible with the real 22-dimensional profile vectors from `value_pipeline.profiler`. |
| `person_a/integration_test.py` | Integration test script | Broken: uses relative imports `from normalize_names import ...` which crash when executed as a package or from repo root (`python -m person_a.integration_test`). |
| `person_a/mock_end_to_end_test.py` | Integration test with mock value profiles | Broken: uses bare relative imports and obsolete mock value profiles. |
| `person_a/evaluate.py` | Metrics calculation (TP, FP, FN, P, R, F1) | Redundant with `evaluation/metrics.py`. Uses loose tuples rather than structured matcher outputs. |
| `person_a/test_normalize.py` | 6 unit tests for normalization | Passing, but uses bare import `from normalize_names import ...` instead of root-relative package import. |
| `value_pipeline/sampler.py` | Sampling values from CSV or column arrays | Functional and tested. Handles null filtering and deterministic random seeding. |
| `value_pipeline/profiler.py` | 22-dimensional statistical profiling & dtype inference | Comprehensive and tested (numeric, date, categorical, id_code, string dtypes). Output vector layout is strictly 22-dimensional. |
| `value_pipeline/similarity.py` | Weighted Euclidean and JSD similarities | Functional and tested with 22-dimensional feature weights. |
| `matching/matcher.py` | Hungarian algorithm (linear_sum_assignment) | Works for 1-to-1 matching. Needs integration with adaptive alpha per-pair matrix and structured Match records. |
| `evaluation/metrics.py` | Precision, recall, F1 calculation | Works and tested. Filters on `accepted=True`. |
| `evaluation/baselines.py` | Name-only, value-only, fixed alpha baseline matchers | Works, but uses loose dictionary profiles and fallback token Jaccard for empty embeddings. |
| `evaluation/benchmark.py` | Synthetic benchmark harness | Evaluates JSON files from `data/synthetic`. Does not write required CSV reports (`reports/comparison_table.csv`, `reports/results_long.csv`, `reports/per_match_details.csv`). |
| `generate_synthetic.py` | Benchmark pair generator | Emits 5 JSON pairs with synthetic samples. Does not emit proper multi-file CSV directories (`data/test_schemas/schema_pair_*/{source.csv, target.csv, ground_truth.csv}`). |
| `person_b/` | Planned value pipeline folder | Empty directory. Dead artifact from split project planning. |
| `notebooks/`, `reports/` | Empty folders | Need standard structure, reports folder needs `.gitkeep` and CSV reports generated. |

## 3. Test Baseline Results
- `pytest -v`: **29 passed** in 6.02s.
- `evaluation.benchmark`:
  - Name-Only Baseline: Precision 0.6667, Recall 0.5917, F1 0.5538
  - Value-Only Baseline: Precision 0.9000, Recall 1.0000, F1 0.9333
  - Fixed-Fusion (alpha=0.5): Precision 0.8667, Recall 1.0000, F1 0.9000
- Windows cp1252 issue discovered and confirmed: `\u2713` in print statements causes fatal crash on default Windows console. Must use ASCII output or robust UTF-8 handling everywhere.
