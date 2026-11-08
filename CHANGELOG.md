# Changelog

All notable changes to TrustGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.11] - 2026-11-08

### Changed

- Claims in the README, docs and console report now state the guarantee as it is proven: a finite-sample, distribution-free *marginal* coverage guarantee for the top-M\* prediction set at a confidence level fixed in advance. The reliability level is presented as a diagnostic, not as a lower bound on future accuracy. New "What the guarantee covers" section (conditions, label reference, scope)
- FAQ definition of the reliability level aligned with the code (`|{mode acceptable}| / (n + 1)` on calibration items)
- New `guarantee_applies` field on `CertificationResult` (and in the JSON report): `false` when no finite prediction set reaches the target. The console report then warns instead of describing the fallback set as meeting the target
- Codebase formatted with `ruff format` (no behavior change)

## [0.3.10] - 2026-09-10

### Added

- Publish workflow: PyPI releases now ship via GitHub Actions on published releases
- `CITATION.cff` citing the arXiv paper
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`

### Fixed

- LICENSE replaced with the canonical Apache-2.0 text so GitHub detects the license (terms unchanged; copyright moved to `NOTICE`)

### Changed

- CI toolchain pinned to verified versions (ruff 0.15.x, mypy 1.19.x, pytest <10; numpy <2.5 in dev extra only)

> The entries below for 0.1.1–0.3.9 were reconstructed on 2026-09-10 from the commit history.

## [0.3.9] - 2026-03-28

### Security

- Removed `code_exec` support, fixed XSS in the HTML questionnaire, sanitized error output

### Fixed

- 7 documentation issues found in re-audit

## [0.3.8] - 2026-03-28

### Fixed

- 5 audit issues: documentation, dead code, default mismatch

## [0.3.7] - 2026-03-28

### Added

- Deterministic-endpoint warning during calibration
- LLM-powered RAG agent example

## [0.3.6] - 2026-03-28

### Changed

- Improved calibration UI for single-answer questions (CLI and HTML questionnaire)
- Made example hard questions genuinely challenging

## [0.3.5] - 2026-03-28

### Added

- RAG agent example: real agent with retriever + calculator tools, hard-question dataset
- Configurable PASS/WEAK/FAIL thresholds
- Local proxy pattern for complex endpoints (docs)

## [0.3.4] - 2026-03-27

### Changed

- Documented custom auth headers for `judge_endpoint`

## [0.3.3] - 2026-03-27

### Fixed

- Parse custom auth headers in `judge_endpoint` config

## [0.3.2] - 2026-03-27

### Added

- Custom auth headers on all adapters (LiteLLM, Azure, etc.)

### Changed

- Provider-agnostic `LLM_API_KEY` everywhere (no longer OpenAI-specific)

## [0.3.1] - 2026-03-27

### Changed

- Flask moved to core dependencies, simplified install
- End-to-end guide rewritten for PyPI installs, Windows commands added

## [0.3.0] - 2026-03-26

### Changed

- Reliability level now uses the exact paper formula (Definition 2.4)
- M* computed at user-specified `--alpha` (default 0.05 = 95% confidence)

## [0.2.1] - 2026-03-26

### Changed

- Human reviewers see only the `n_cal` calibration questions, not all questions
- Ground-truth generation fallback in the compare step

## [0.2.0] - 2026-03-26

### Added

- LLM semantic canonicalizer and `--auto-judge` LLM-as-judge calibration
- Semantic (LLM-based) calibration matching of ground truth
- `--min-reliability` for CI/CD gating
- `--concurrency` flag; dynamic time estimation measuring real API latency

### Changed

- Parallelized canonicalization; throttled and cached LLM calls
- Pre-flight and spinner UX (K adjustable from pre-flight, enriched MCQ answers)

## [0.1.1] - 2026-03-24

### Added

- 120-question test dataset and complete end-to-end test guide
- "Getting Your Questions" guide (cold-start problem)
- Progress spinner during certification and calibration

## [0.1.0] - 2026-03-22

### Added

- Full certification pipeline: sample, canonicalize, calibrate, certify
- Self-consistency sampling with configurable K
- Sequential stopping via Hoeffding bounds (saves ~50% API costs)
- Conformal prediction calibration with formal coverage guarantees
- Five built-in canonicalizers: MCQ, numeric, LLM semantic, LLM-as-judge, embedding clustering
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
