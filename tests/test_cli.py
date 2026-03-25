"""Tests for the Click CLI."""

from __future__ import annotations

import json

from click.testing import CliRunner

from theaios.trustgate.cli import main


class TestCLIHelp:
    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "certify" in result.output
        assert "compare" in result.output
        assert "cache" in result.output
        assert "version" in result.output

    def test_certify_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["certify", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--questions" in result.output
        assert "--ground-truth" in result.output
        assert "--output" in result.output

    def test_compare_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output

    def test_cache_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "--help"])
        assert result.exit_code == 0
        assert "stats" in result.output
        assert "clear" in result.output

    def test_sample_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["sample", "--help"])
        assert result.exit_code == 0
        assert "--questions" in result.output


class TestVersion:
    def test_version_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "trustgate" in result.output
        assert "trustgate 0." in result.output  # version present, not hardcoded


class TestCacheCommands:
    def test_cache_stats(self, tmp_path: object) -> None:
        import pathlib

        cache_dir = str(pathlib.Path(str(tmp_path)) / "cache")
        runner = CliRunner()
        result = runner.invoke(main, ["cache", "stats", "--cache-dir", cache_dir])
        assert result.exit_code == 0
        assert "Total entries: 0" in result.output

    def test_cache_clear(self, tmp_path: object) -> None:
        import pathlib

        cache_dir = str(pathlib.Path(str(tmp_path)) / "cache")
        runner = CliRunner()
        # --yes to skip confirmation prompt
        result = runner.invoke(main, ["cache", "clear", "--cache-dir", cache_dir, "--yes"])
        assert result.exit_code == 0
        assert "Cache cleared" in result.output


class TestCertifyErrors:
    def test_missing_config_no_endpoint(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["certify", "--config", "/nonexistent.yaml"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_invalid_questions_file(self, tmp_path: object) -> None:
        import pathlib

        cfg = pathlib.Path(str(tmp_path)) / "config.yaml"
        cfg.write_text(
            "endpoint:\n  url: https://api.openai.com/v1/chat/completions\n"
        )
        runner = CliRunner()
        result = runner.invoke(main, [
            "certify",
            "--config", str(cfg),
            "--questions", "/nonexistent_questions.csv",
        ])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestCertifyOutput:
    """Test that certify works end-to-end with mock data using output formats."""

    def _run_certify(
        self, tmp_path: object, output_format: str, output_file: str | None = None,
    ) -> object:
        """Run certify with a fully mocked pipeline via patching."""
        import pathlib
        from unittest.mock import patch

        from theaios.trustgate.types import CertificationResult

        mock_result = CertificationResult(
            reliability_level=0.90,
            m_star=1,
            coverage=0.95,
            conditional_coverage=0.98,
            capability_gap=0.02,
            n_cal=50,
            n_test=50,
            k_used=10,
            api_cost_estimate=5.00,
            alpha_coverage={0.10: 0.95},
        )

        p = pathlib.Path(str(tmp_path))
        cfg = p / "config.yaml"
        cfg.write_text(
            "endpoint:\n"
            "  url: https://api.openai.com/v1/chat/completions\n"
            "  model: gpt-4.1-mini\n"
            "  api_key_env: TEST_KEY\n"
        )

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        args = ["certify", "--config", str(cfg)]
        if output_file:
            out_path = str(p / output_file)
            args += ["--output", output_format, "--output-file", out_path]
        else:
            args += ["--output", output_format]

        with patch("theaios.trustgate.cli.certify", return_value=mock_result):
            return runner.invoke(main, args)

    def test_json_output(self, tmp_path: object) -> None:
        result = self._run_certify(tmp_path, "json")
        assert result.exit_code == 0  # type: ignore[union-attr]
        data = json.loads(result.output)  # type: ignore[union-attr]
        assert data["reliability_level"] == 0.90

    def test_csv_output(self, tmp_path: object) -> None:
        result = self._run_certify(tmp_path, "csv")
        assert result.exit_code == 0  # type: ignore[union-attr]
        assert "reliability_level" in result.output  # type: ignore[union-attr]

    def test_console_output(self, tmp_path: object) -> None:
        result = self._run_certify(tmp_path, "console")
        assert result.exit_code == 0  # type: ignore[union-attr]

    def test_json_output_file(self, tmp_path: object) -> None:
        import pathlib

        result = self._run_certify(tmp_path, "json", output_file="result.json")
        assert result.exit_code == 0  # type: ignore[union-attr]
        p = pathlib.Path(str(tmp_path)) / "result.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["reliability_level"] == 0.90
