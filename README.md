# Schema Matching Using Adaptive Fusion

A Python project for matching columns across schemas by combining column-name similarity with value-profile similarity. The matcher adapts the relative weight of these signals and uses globally optimal one-to-one assignment to produce schema correspondences.

## Overview

The project evaluates whether columns from two different schemas represent the same underlying data. It combines:

- **Name similarity** based on normalized column names and cosine similarity between embeddings.
- **Value profiling** based on inferred data types, cardinality, null ratios, length statistics, numeric statistics, and common patterns such as emails, phone numbers, IDs, and dates.
- **Adaptive fusion** that changes the name/value weighting based on name quality, similarity, and disagreement between signals.
- **Global matching** using the Hungarian algorithm from SciPy.

## Repository structure

```text
data/
  synthetic/       Generated schema-pair benchmark data
  test_schemas/    Schema fixtures
  valentine/       Placeholder for external Valentine benchmark datasets

evaluation/
  baselines.py     Baseline name-only, value-only, and fixed-fusion matchers
  benchmark.py     Benchmark loading and comparison harness
  metrics.py       Precision, recall, and F1 calculations

matching/
  matcher.py       Similarity matrix construction and Hungarian matching

person_a/
  fusion.py        Adaptive name/value fusion
  name_similarity.py
                   Cosine-based name similarity helpers
  normalize_names.py
                   Column-name normalization utilities

shared/
  contracts.py     Shared profile contracts
  column_profile_schema.json
                   Column profile schema definition

value_pipeline/
  profiler.py      Column value profiling and 22-dimensional vectorization
  sampler.py       Value sampling utilities
  similarity.py    Value-profile similarity calculations

tests/             Unit tests for the pipeline and matcher
generate_synthetic.py
                   Synthetic benchmark-pair generator
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Generate benchmark data

The synthetic generator creates five schema-pair scenarios, including exact matches, renamed columns, conflicting data types, mixed matchability, and a customer/client example:

```bash
python generate_synthetic.py
```

Generated pairs are written to `data/synthetic/`.

## Run the benchmark

```bash
python -m evaluation.benchmark
```

The benchmark compares name-only, value-only, and fixed-fusion baselines and reports mean precision, recall, and F1 scores.

## Run tests

```bash
pytest
```

## Main matching flow

A column profile contains identifiers, normalized name tokens, optional name embeddings, sampled values, and a structured `value_profile`. The value profiler converts each profile into a fixed 22-dimensional vector. Name and value similarities are combined into a similarity matrix, and `matching.matcher.match_schemas()` applies the Hungarian algorithm to select one-to-one matches above the configured confidence threshold.

## Dependencies

The project uses NumPy, SciPy, pandas, scikit-learn, python-dateutil, and pytest. See [`requirements.txt`](requirements.txt) for the version constraints.
