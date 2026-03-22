# Changelog

All notable changes to TrustGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-22

### Added

- Full certification pipeline: sample, canonicalize, calibrate, certify
- Self-consistency sampling with configurable K
- Sequential stopping via Hoeffding bounds (saves ~50% API costs)
- Conformal prediction calibration with formal coverage guarantees
- Five built-in canonicalizers: MCQ, numeric, code execution, LLM-as-judge, embedding clustering
- Custom canonicalizer plugin system with `@register_canonicalizer` decorator
- Multi-model comparison with pairwise reliability deltas
- Built-in dataset loaders: GSM8K, MMLU, TruthfulQA
- Generic endpoint support: agents, RAG pipelines, custom APIs via `request_template` / `response_path`
- Optional temperature (`null` = endpoint controls its own randomness)
- Pre-flight cost estimation with cost/reliability arbitrage table
- Human calibration with blind review (randomized answer order, no frequency/rank info)
- Exportable HTML questionnaire for cross-organization calibration delegation
- Local Flask-based calibration web UI with admin panel
- Profile quality diagnostic: automatic detection of canonicalization failures
- `TrustGate` runtime trust layer with passthrough and sampled modes
- CLI commands: `certify`, `calibrate` (with `--serve` and `--export`), `compare`, `sample`, `cache`
- JSON and CSV export for certification results
- Rich terminal output with formatted tables
- Disk-based response caching with SHA-256 keys (re-runs are free)
- YAML configuration file support with CLI overrides
- `py.typed` marker for PEP 561 compliance
- MkDocs Material documentation site with GitHub Pages deployment
- GitHub Actions CI workflow (Python 3.10–3.13)
- Comprehensive test suite (442 tests)
