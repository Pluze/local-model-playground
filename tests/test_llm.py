"""Fast checks for platform routing and rolling-memory boundaries."""
import json
import contextlib
import gzip
import io
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playground.llm import assets as install
from playground.llm import launcher as launch
from playground.llm.proxy import RollingMemory, ResponseRollover, make_handler


class AssetRoutingTests(unittest.TestCase):
    def test_supported_platforms(self):
        for system, machine, expected in [
            ("Darwin", "arm64", "darwin-arm64"),
            ("Darwin", "x86_64", "darwin-x64"),
            ("Linux", "aarch64", "linux-arm64"),
            ("Linux", "AMD64", "linux-x64"),
            ("Windows", "AMD64", "windows-x64"),
        ]:
            with self.subTest(expected=expected), patch("platform.system", return_value=system), patch("platform.machine", return_value=machine):
                self.assertEqual(install.platform_key(), expected)

    def test_archive_path_rejection(self):
        for path in ("../escape", "/absolute", "nested/../../escape"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                install.safe_member(path)

    def test_profiles_have_pinned_models(self):
        self.assertEqual(install.profile_names(), ("compact", "uncensored", "direct"))
        for name in install.profile_names():
            _, selected = install.profile(name)
            self.assertEqual(len(selected["model"]["sha256"]), 64)
            self.assertGreater(selected["model"]["size"], 5_000_000_000)
            defaults = selected["defaults"]
            self.assertGreaterEqual(defaults["context"], 4096)
            self.assertEqual(1, defaults["parallel"])
            self.assertEqual("q8_0", defaults["kv_cache"])

    def test_custom_model_does_not_inherit_profile_adapter(self):
        with patch.dict(os.environ, {"LOCAL_LLM_MODEL": "/tmp/custom.gguf"}, clear=True):
            _, model, adapter, _, _ = install.paths("compact")
        self.assertEqual(model, Path("/tmp/custom.gguf"))
        self.assertIsNone(adapter)

    def test_gated_download_explains_first_run_token(self):
        _, selected = install.profile("uncensored")
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(install, "HF_TOKEN_PATHS", ()), \
             patch.object(install, "huggingface_token", return_value=None), \
             self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            install.download(selected["model"]["url"], Path(directory) / "model.gguf",
                             selected["model"])

    def test_standard_huggingface_token_cache_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token"
            token_path.write_text("hf_cached\n")
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(install, "HF_TOKEN_PATHS", (token_path,)), \
                 patch.object(install.platform, "system", return_value="Linux"):
                self.assertEqual(install.huggingface_token(), "hf_cached")

    def test_macos_keychain_token_is_used_without_printing_it(self):
        result = type("Result", (), {"returncode": 0, "stdout": "hf_keychain\n"})()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(install.platform, "system", return_value="Darwin"), \
             patch.object(install.subprocess, "run", return_value=result):
            self.assertEqual(install.huggingface_token(), "hf_keychain")

    def test_legacy_project_environment_is_not_read(self):
        with patch.dict(os.environ, {"ORCA_PROFILE": "direct"}, clear=True):
            self.assertEqual(install.default_profile(), "compact")


class MemoryTests(unittest.TestCase):
    def test_compression_keeps_instructions_and_latest_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=4000, compact_at=70, keep_tokens=25)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            calls = []

            def answer(path, payload, timeout=120):
                calls.append((path, payload))
                return {"choices": [{"message": {"content": "Earlier facts summarized."}}]}

            memory.post = answer
            messages = [{"role": "system", "content": "Keep this instruction."}]
            messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 20} for i in range(5)]
            messages += [{"role": "user", "content": "Latest question stays."}]
            result = memory.compact({"messages": messages})
            self.assertEqual(result["messages"][0], messages[0])
            self.assertEqual(result["messages"][-1], messages[-1])
            self.assertIn("Earlier facts summarized.", json.dumps(result["messages"]))
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(directory) / "rolling-memory.json").is_file())

    def test_tool_messages_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, 100, 70, 25)
            payload = {"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "result"}]}
            self.assertIs(memory.compact(payload), payload)

    def test_single_large_message_is_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, 4096, 1000, 400)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            with self.assertRaisesRegex(ValueError, "latest message"):
                memory.compact({"messages": [{"role": "user", "content": "word " * 3200}]})

    def test_prefix_digest_matches_existing_cache_keys(self):
        history = [{"role": "user", "content": "\u4f60\u597d"}, {"role": "assistant", "content": "fine"}]
        self.assertEqual(RollingMemory.prefix_digests(history),
                         [RollingMemory.digest(history[:i]) for i in (1, 2)])

    def test_large_requested_output_reduces_input_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=5000, compact_at=4000, keep_tokens=600)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            memory.post = lambda path, payload, timeout=120: {"choices": [{"message": {"content": "short memory"}}]}
            messages = [{"role": "user", "content": "old " * 800},
                        {"role": "assistant", "content": "older " * 800},
                        {"role": "user", "content": "new " * 300}]
            result = memory.compact({"messages": messages, "max_tokens": 4000})
            self.assertLess(memory.count(result["messages"]), 1000)

    def test_summary_splits_when_archived_text_does_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=4000, compact_at=900, keep_tokens=400)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            calls = []
            def summarize(path, payload, timeout=120):
                calls.append(payload)
                return {"choices": [{"message": {"content": "short memory"}}]}
            memory.post = summarize
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 450}
                        for i in range(9)]
            result = memory.compact({"messages": messages})
            self.assertGreater(len(calls), 1)
            self.assertLess(memory.count(result["messages"]), 900)

    def test_retained_history_starts_with_a_user_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=5000, compact_at=80, keep_tokens=45)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            memory.post = lambda path, payload, timeout=120: {"choices": [{"message": {"content": "Short memory."}}]}
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 20}
                        for i in range(5)]
            result = memory.compact({"messages": messages})
            self.assertEqual([m["role"] for m in result["messages"]], ["system", "user"])
            self.assertEqual(result["messages"][-1], messages[-1])

    def test_failed_compaction_does_not_commit_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=4000, compact_at=900, keep_tokens=400)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            calls = []

            def summarize(path, payload, timeout=120):
                calls.append(payload)
                if len(calls) == 2:
                    raise RuntimeError("summary failed")
                return {"choices": [{"message": {"content": "Short memory."}}]}

            memory.post = summarize
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 450}
                        for i in range(9)]
            with self.assertRaisesRegex(RuntimeError, "summary failed"):
                memory.compact({"messages": messages})
            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(memory.cache, {})
            self.assertFalse(memory.path.exists())

    def test_one_oversized_older_message_is_summarized_in_pieces(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 9999, context=4000, compact_at=900, keep_tokens=300)
            memory.count = lambda messages, original=None: sum(len(m["content"].split()) for m in messages)
            archived = "word " * 6500
            pieces = []

            def summarize(path, payload, timeout=120):
                source = json.loads(payload["messages"][1]["content"])
                pieces.extend(item["content"] for item in source["archived_messages"])
                return {"choices": [{"message": {"content": "Short memory."}}]}

            memory.post = summarize
            messages = [{"role": "user", "content": archived},
                        {"role": "assistant", "content": "Earlier reply."},
                        {"role": "user", "content": "Latest question."}]
            result = memory.compact({"messages": messages})
            self.assertGreater(len(pieces), 1)
            self.assertEqual("".join(pieces), archived)
            self.assertEqual(result["messages"][-1], messages[-1])
            self.assertIn("Short memory.", result["messages"][0]["content"])


class LaunchArgumentsTests(unittest.TestCase):
    def test_numbered_startup_choice(self):
        with patch("builtins.input", side_effect=["oops", "2"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.prompt_choice("Context window", launch.CONTEXT_PRESETS, 32768), 8192)

    def test_enter_keeps_startup_default(self):
        with patch("builtins.input", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.prompt_choice("Context window", launch.CONTEXT_PRESETS, 32768), 32768)

    def test_gated_model_access_opens_pages_and_accepts_hidden_token(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(install, "paths", return_value=("key", Path(directory) / "missing.gguf", None, None, None)), \
             patch.object(install, "huggingface_token", return_value=None), \
             patch.object(install, "set_session_huggingface_token") as set_token, \
             patch.object(launch.webbrowser, "open") as open_page, \
             patch.object(launch.getpass, "getpass", return_value="hf_interactive"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(launch.prepare_model_access("uncensored"))
            set_token.assert_called_once_with("hf_interactive")
            self.assertEqual(open_page.call_count, 2)

    def test_empty_gated_token_returns_to_model_selection(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(install, "paths", return_value=("key", Path(directory) / "missing.gguf", None, None, None)), \
             patch.object(install, "huggingface_token", return_value=None), \
             patch.object(launch.webbrowser, "open"), \
             patch.object(launch.getpass, "getpass", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(launch.prepare_model_access("uncensored"))

    def test_cancelled_gated_access_restores_compact_menu_default(self):
        with patch.object(sys, "argv", ["launch.py", "--context", "8192", "--no-browser"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(launch, "existing_proxy", return_value=None), \
             patch.object(launch, "prompt_choice", side_effect=["uncensored", "compact"]) as choose, \
             patch.object(launch, "prepare_model_access", side_effect=[False, True]), \
             patch.object(install, "install", side_effect=RuntimeError("setup reached")), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.main(), 1)
        self.assertEqual([call.args[2] for call in choose.call_args_list], ["compact", "compact"])

    def test_window_defaults_for_8k(self):
        self.assertEqual(launch.window_settings(8192, 1), (8192, 6144, 3031))

    def test_small_window_keeps_proportional_output_room(self):
        self.assertEqual(launch.window_settings(4096, 1), (4096, 3072, 1515))

    def test_context_can_be_set_on_command_line(self):
        with patch.object(sys, "argv", ["launch.py", "--check", "--context", "16384"]), \
             patch.object(install, "check", return_value=True):
            self.assertEqual(launch.main(), 0)

    def test_model_can_be_set_on_command_line(self):
        with patch.object(sys, "argv", ["launch.py", "--check", "--model", "direct"]), \
             patch.object(install, "check", return_value=True) as checker:
            self.assertEqual(launch.main(), 0)
            checker.assert_called_once_with("direct")

    def test_parallel_slots_require_enough_context_each(self):
        with patch.object(sys, "argv", ["launch.py", "--check", "--context", "8192"]), \
             patch.dict(os.environ, {"LOCAL_LLM_PARALLEL": "3"}), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit):
            launch.main()

    def test_launcher_leaves_reasoning_to_webui(self):
        command = launch.build_command("server", "model", "adapter", 8081, 8192)
        for option in ("--reasoning", "--reasoning-budget", "--reasoning-effort"):
            self.assertNotIn(option, command)

    def test_model_without_adapter_omits_lora_option(self):
        command = launch.build_command("server", "model", None, 8081, 8192)
        self.assertNotIn("--lora", command)

    def test_compact_model_keeps_lora_option(self):
        command = launch.build_command("server", "model", "adapter", 8081, 8192)
        self.assertEqual(command[command.index("--lora") + 1], "adapter")

    def test_host_prompt_cache_has_bounded_memory(self):
        command = launch.build_command("server", "model", "adapter", 8081, 8192)
        self.assertEqual(command[command.index("--cache-ram") + 1], "512")

    def test_profile_defaults_control_context_and_kv_cache(self):
        self.assertEqual(16384, launch.profile_defaults("compact")["context"])
        self.assertEqual(8192, launch.profile_defaults("direct")["context"])
        command = launch.build_command("server", "model", None, 8081, 8192, "direct")
        self.assertEqual(command[command.index("-ctk") + 1], "q8_0")
        self.assertEqual(command[command.index("-ctv") + 1], "q8_0")

    def test_second_default_launch_reuses_running_service(self):
        with patch.object(sys, "argv", ["launch.py", "--no-prompt"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(launch, "existing_proxy", return_value={
                 "url": "http://127.0.0.1:8080", "port": 8080, "pid": 123,
                 "active": False, "started": 0,
             }), \
             patch.object(install, "install") as installer, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(), 0)
            installer.assert_not_called()

    def test_interactive_launch_can_stop_existing_instance(self):
        instance = {"url": "http://127.0.0.1:8080", "port": 8080, "pid": 123,
                    "active": False, "started": 0}
        with patch.object(sys, "argv", ["launch.py", "--model", "compact", "--context", "8192"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(launch, "existing_proxy", return_value=instance), \
             patch("builtins.input", return_value="y"), \
             patch.object(launch, "stop_existing_instance") as stop, \
             patch.object(install, "install", side_effect=RuntimeError("setup reached")), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.main(), 1)
            stop.assert_called_once_with(instance)

    def test_stop_existing_instance_uses_status_pid(self):
        instance = {"url": "http://127.0.0.1:8080", "port": 8080, "pid": 123}
        with patch.object(launch.os, "kill") as kill, \
             patch.object(launch, "existing_proxy", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            launch.stop_existing_instance(instance)
        kill.assert_called_once_with(123, launch.signal.SIGTERM)

    def test_legacy_instance_pid_comes_from_listener(self):
        completed = type("Completed", (), {"stdout": "456\n"})()
        with patch.object(launch.subprocess, "run", return_value=completed) as run:
            self.assertEqual(launch.listener_pid(53973), 456)
        self.assertIn("-iTCP:53973", run.call_args.args[0])

    def test_existing_proxy_checks_recorded_dynamic_port(self):
        with tempfile.TemporaryDirectory() as directory:
            active_port = Path(directory) / "active-port"
            active_port.write_text("53973")
            with patch.object(launch, "ACTIVE_PORT", active_port), \
                 patch.object(launch.urllib.request, "urlopen") as open_url:
                open_url.return_value.__enter__.return_value.read.return_value = b'{"active": false, "started": 0}'
                self.assertEqual(launch.existing_proxy()["url"], "http://127.0.0.1:53973")
                self.assertEqual(launch.existing_proxy()["port"], 53973)
                self.assertEqual(open_url.call_args.args[0], "http://127.0.0.1:53973/orca/status")


class ProxyStreamingTests(unittest.TestCase):
    def test_webui_replay_lookup_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 1, 4096, 3072, 1515)
            event = memory.begin_stream("chat-1")
            memory.append_stream("chat-1", event, b"data: first\n\n")
            memory.append_stream("chat-1", event, b"data: [DONE]\n\n")
            memory.end_stream("chat-1", event)
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(memory))
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{proxy.server_port}"
            try:
                with urllib.request.urlopen(base + "/v1/stream?conv_id=chat-1&from=0") as response:
                    self.assertEqual(response.read(), b"data: first\n\ndata: [DONE]\n\n")
                lookup = urllib.request.Request(base + "/v1/streams/lookup",
                                                data=b'{"conversation_ids":["chat-1"]}',
                                                headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(lookup) as response:
                    self.assertEqual(json.load(response)[0]["total_bytes"], 27)
                delete = urllib.request.Request(base + "/v1/stream?conv_id=chat-1", method="DELETE")
                with urllib.request.urlopen(delete) as response:
                    self.assertEqual(response.status, 204)
                self.assertIsNone(memory.replay_stream("chat-1"))
            finally:
                proxy.shutdown()
                proxy.server_close()

    def test_unmanaged_stream_routes_still_reach_llama(self):
        calls = []

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                calls.append(("GET", self.path))
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"live")

            def do_DELETE(self):
                calls.append(("DELETE", self.path))
                self.send_response(204)
                self.end_headers()

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, backend.server_port, 4096, 3072, 1515)
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(memory))
            for server in (backend, proxy):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{proxy.server_port}/v1/stream?conv_id=bounded"
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.read(), b"live")
                with urllib.request.urlopen(urllib.request.Request(url, method="DELETE")) as response:
                    self.assertEqual(response.status, 204)
                self.assertEqual(calls, [("GET", "/v1/stream?conv_id=bounded"),
                                         ("DELETE", "/v1/stream?conv_id=bounded")])
            finally:
                for server in (proxy, backend):
                    server.shutdown()
                    server.server_close()

    def test_stream_cancel_tracks_current_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 1, 4096, 3072, 1515)
            first = memory.begin_stream("chat-1")
            second = memory.begin_stream("chat-1")
            memory.end_stream("chat-1", first)
            memory.cancel_stream("chat-1")
            self.assertTrue(first.is_set())
            self.assertTrue(second.is_set())
            memory.end_stream("chat-1", second)
            self.assertEqual(memory.stream_cancellations, {})

    def test_response_checkpoint_keeps_current_user_and_exact_answer_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 1, 4096, 3072, 1515)
            messages = [{"role": "system", "content": "Keep facts exact"},
                        {"role": "user", "content": "Explain the result"}]
            with patch.object(memory, "count", return_value=100), \
                 patch.object(memory, "post", return_value={"choices": [{"message": {"content": "Useful checkpoint"}}]}):
                continuation, tail = memory.checkpoint_response(messages, "work", "ABCDEF")
            self.assertTrue(continuation[0]["content"].startswith(messages[0]["content"]))
            self.assertIn("Useful checkpoint", continuation[0]["content"])
            self.assertEqual(continuation[-2], messages[-1])
            self.assertEqual((continuation[-1]["content"], tail), ("ABCDEF", "ABCDEF"))
            with patch.object(memory, "count", return_value=100), \
                 patch.object(memory, "post", return_value={"choices": [{"message": {"content": ""}}]}):
                with self.assertRaisesRegex(ValueError, "empty response checkpoint"):
                    memory.checkpoint_response(messages, "work", "ABCDEF")
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                memory.checkpoint_response(messages, "work", "ABCDEF", cancelled)

    def test_second_checkpoint_carries_previous_summary_once(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = RollingMemory(directory, 1, 4096, 3072, 1515)
            messages = [{"role": "system", "content": "Original rule\n\nCheckpoint from the unfinished answer:\nEarlier result"},
                        {"role": "user", "content": "Continue"},
                        {"role": "assistant", "content": "partial"}]
            sources = []

            def summarize(path, payload, timeout=120):
                sources.append(payload["messages"][1]["content"])
                return {"choices": [{"message": {"content": "Updated result"}}]}

            with patch.object(memory, "count", return_value=100), patch.object(memory, "post", side_effect=summarize):
                continuation, _ = memory.checkpoint_response(messages, "", "later")
            self.assertIn("Earlier result", "".join(sources))
            self.assertEqual(continuation[0]["content"].count("Checkpoint from the unfinished answer:"), 1)
            self.assertIn("Updated result", continuation[0]["content"])

    def test_rollover_compacts_only_at_step_boundary_and_preserves_webui_budget(self):
        class Memory:
            context = 4096

            def __init__(self):
                self.checkpoints = 0

            def count(self, messages, original=None):
                return 3500

            def checkpoint_response(self, messages, reasoning, content, cancelled):
                self.checkpoints += 1
                self.reasoning = reasoning
                self.content = content
                return ([{"role": "system", "content": "checkpoint"},
                         {"role": "user", "content": "question"},
                         {"role": "assistant", "content": content[-3:]}], content[-3:])

            def set_status(self, message):
                pass

        memory = Memory()
        payload = {"messages": [{"role": "user", "content": "question"}], "stream": True,
                   "thinking_budget_tokens": 512, "chat_template_kwargs": {"enable_thinking": True}}
        state = ResponseRollover(memory, payload, threading.Event())
        self.assertEqual(state.first_request()["max_tokens"], 1024)
        state.ingest({"reasoning_content": "thinking"})
        state.ingest({"content": "abcdef"})
        step = state.next_request()
        self.assertEqual(memory.checkpoints, 1)
        self.assertEqual((memory.reasoning, memory.content), ("thinking", "abcdef"))
        self.assertEqual(step["messages"][-1]["content"], "def")
        self.assertEqual(step["thinking_budget_tokens"], 0)
        self.assertFalse(step["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(ResponseRollover.eligible(dict(payload, max_tokens=100)))
        self.assertFalse(ResponseRollover.eligible(dict(payload, tools=[{"type": "function"}])))

    def test_unlimited_response_rolls_over_without_duplicate_or_false_finish(self):
        calls = []

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(payload)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                part = "ABC" if len(calls) == 1 else "DEF"
                finish = "length" if len(calls) == 1 else "stop"
                for frame in ({"choices": [{"delta": {"content": part}, "finish_reason": None}]},
                              {"choices": [{"delta": {}, "finish_reason": finish}]}):
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        class Memory:
            context = 4096

            def __init__(self, base):
                self.base = base

            def compact(self, payload):
                return payload

            def count(self, messages, original=None):
                return 100

            def set_status(self, message):
                pass

            def clear_status(self):
                pass

            def begin_stream(self, conversation_id):
                return threading.Event()

            def append_stream(self, conversation_id, event, data):
                pass

            def end_stream(self, conversation_id, event):
                pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Memory(
            f"http://127.0.0.1:{backend.server_port}")))
        for server in (backend, proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            payload = {"messages": [{"role": "user", "content": "letters"}],
                       "stream": True, "max_tokens": -1, "thinking_budget_tokens": 512}
            request = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3) as response:
                lines = [line[6:].strip() for line in response if line.startswith(b"data: ")]
            frames = [json.loads(line) for line in lines if line != b"[DONE]"]
            content = "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames)
            finishes = [frame["choices"][0]["finish_reason"] for frame in frames
                        if frame["choices"][0]["finish_reason"]]
            self.assertEqual(content, "ABCDEF")
            self.assertEqual(finishes, ["stop"])
            self.assertEqual(lines.count(b"[DONE]"), 1)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[1]["continue_final_message"])
            self.assertFalse(calls[1]["add_generation_prompt"])
            self.assertEqual(calls[1]["messages"][-1]["content"], "ABC")
            self.assertEqual(calls[1]["thinking_budget_tokens"], 0)
        finally:
            for server in (proxy, backend):
                server.shutdown()
                server.server_close()

    def test_webui_reasoning_fields_and_stream_are_forwarded(self):
        first_chunk_sent = threading.Event()
        allow_second_chunk = threading.Event()
        received = {}

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"step":1}\n\n')
                self.wfile.flush()
                first_chunk_sent.set()
                if allow_second_chunk.wait(2):
                    self.wfile.write(b'data: {"step":2}\n\n')
                    self.wfile.flush()

        class Memory:
            def __init__(self, base):
                self.base = base
                self.message = ""

            def compact(self, payload):
                return payload

            def set_status(self, message):
                self.message = message

            def clear_status(self):
                self.message = ""

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        memory = Memory(f"http://127.0.0.1:{backend.server_port}")
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(memory))
        threads = [threading.Thread(target=server.serve_forever, daemon=True)
                   for server in (backend, proxy)]
        for thread in threads:
            thread.start()
        payload = {"messages": [{"role": "user", "content": "hi"}],
                   "chat_template_kwargs": {"enable_thinking": False},
                   "thinking_budget_tokens": 0, "reasoning_effort": "none", "stream": True,
                   "max_tokens": 5}
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertTrue(first_chunk_sent.wait(1))
                self.assertEqual(response.readline(), b'data: {"step":1}\n')
                self.assertEqual(memory.message, "Model responding...")
                allow_second_chunk.set()
                self.assertEqual(response.readline(), b"\n")
                self.assertEqual(response.readline(), b'data: {"step":2}\n')
            self.assertEqual(received, payload)
        finally:
            allow_second_chunk.set()
            for server in (proxy, backend):
                server.shutdown()
                server.server_close()

    def test_status_is_visible_without_changing_model_stream(self):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                page = gzip.compress(b"<html><body>Upstream WebUI</body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        with tempfile.TemporaryDirectory() as directory:
            backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
            memory = RollingMemory(directory, backend.server_port, 4096, 3072, 1515)
            proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(memory))
            threads = [threading.Thread(target=server.serve_forever, daemon=True)
                       for server in (backend, proxy)]
            for thread in threads:
                thread.start()
            try:
                base = f"http://127.0.0.1:{proxy.server_port}"
                with urllib.request.urlopen(base + "/", timeout=3) as response:
                    page = response.read().decode()
                self.assertIn("Upstream WebUI", page)
                self.assertIn("/orca/status", page)
                memory.set_status("Compacting context")
                with urllib.request.urlopen(base + "/orca/status", timeout=3) as response:
                    status = json.load(response)
                self.assertTrue(status["active"])
                self.assertIn("Compacting", status["message"])
                self.assertEqual(status["service"], "Local Model Playground")
                self.assertEqual(status["pid"], os.getpid())
                memory.clear_status()
                with urllib.request.urlopen(base + "/orca/status", timeout=3) as response:
                    self.assertFalse(json.load(response)["active"])
            finally:
                for server in (proxy, backend):
                    server.shutdown()
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
