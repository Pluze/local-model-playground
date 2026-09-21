"""Tests for the unified launcher and storage controls."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from playground import cli
from playground import storage


class CliTests(unittest.TestCase):
    def test_help(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--help"]), 0)
        self.assertIn("Run one model at a time", output.getvalue())

    def test_model_shortcut_routes_to_expected_backend(self):
        with patch.object(cli, "_run", return_value=0) as run:
            self.assertEqual(cli.main(["flux2", "--no-browser"]), 0)
        run.assert_called_once_with("image", "flux2", ["--no-browser"])

    def test_benchmark_command_routes_to_benchmark_runner(self):
        with patch.object(cli.benchmark, "main", return_value=0) as runner:
            self.assertEqual(cli.main(["benchmark", "--model", "flux2"]), 0)
        runner.assert_called_once_with(["--model", "flux2"])

    def test_live_session_blocks_second_model(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cli, "SESSION", Path(directory) / "session.json"), \
             patch.object(cli, "ensure_local_layout"), \
             patch.object(cli, "_pid_alive", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            cli.SESSION.write_text('{"pid": 42, "kind": "llm", "model": "compact"}')
            self.assertFalse(cli._claim_session("image", "flux2"))

    def test_human_size(self):
        self.assertEqual("1.0 GiB", storage.human_size(1024**3))


if __name__ == "__main__":
    unittest.main()
