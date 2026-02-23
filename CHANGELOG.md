# Changelog

All notable changes to TrustGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-23

### Added

- Full certification pipeline: sample, canonicalize, calibrate, certify
- Self-consistency sampling with configurable K and temperature
- Sequential stopping via Hoeffding bounds (saves ~50% API costs)
- Conformal prediction calibration with formal coverage guarantees
- Five built-in canonicalizers: MCQ, numeric, code execution, LLM-as-judge, embedding clustering
- Custom canonicalizer plugin system with `@register_canonicalizer` decorator
- Multi-model comparison with pairwise reliability deltas
- Built-in dataset loaders: GSM8K, MMLU, TruthfulQA
- Local human calibration web UI (Flask-based)
- CLI with commands: certify, compare, sample, cache stats/clear, version
- JSON and CSV export for certification results
- Rich terminal output with formatted tables
- Disk-based response caching (re-runs are free)
- YAML configuration file support
- Support for OpenAI, Anthropic, Together, and any OpenAI-compatible endpoint
- GitHub Actions CI workflow (Python 3.10-3.13)
- Comprehensive test suite (330+ tests)
- Full documentation (concepts, configuration, CLI, API reference, FAQ)
