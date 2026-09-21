import json
import socket
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from playground.image import launcher as MODULE
from playground.image import webui


class ManifestTests(unittest.TestCase):
    def test_models_have_complete_pinned_assets(self):
        self.assertEqual({"flux2", "qwen21"}, set(MODULE.MODELS))
        for model in MODULE.MODELS.values():
            self.assertGreater(model["steps"], 0)
            self.assertGreater(model["cfg"], 0)
            self.assertEqual({"fast", "balanced", "detail"}, set(model["presets"]))
            self.assertTrue(model["sizes"])
            self.assertTrue(model["samplers"])
            self.assertTrue(model["schedulers"])
            self.assertEqual({"steps", "cfg"}, set(model["parameter_guides"]))
            self.assertEqual({"sampler", "scheduler"}, set(model["option_guides"]))
            self.assertEqual({"diffusion", "text_encoder", "vae"}, set(model["assets"]))
            for asset in model["assets"].values():
                self.assertTrue(asset["url"].startswith("https://huggingface.co/"))
                self.assertIn("/resolve/", asset["url"])
                self.assertGreater(asset["size"], 0)
                self.assertEqual(64, len(asset["sha256"]))

    def test_runtime_downloads_are_pinned(self):
        self.assertEqual(40, len(MODULE.COMFY_COMMIT))
        self.assertEqual(40, len(MODULE.GGUF_COMMIT))
        for asset in MODULE.RUNTIME_FILES.values():
            self.assertTrue(asset["url"].startswith("https://"))
            self.assertGreater(asset["size"], 0)
            self.assertEqual(64, len(asset["sha256"]))

    def test_verified_rejects_wrong_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"wrong")
            spec = {"size": 5, "sha256": "0" * 64}
            self.assertFalse(MODULE.verified(path, spec, full=True))

    def test_commands_bind_only_to_loopback(self):
        command = MODULE.build_command(1234)
        self.assertIn("127.0.0.1", command)
        self.assertNotIn("0.0.0.0", command)

    def test_benchmark_workflows_use_expected_loaders(self):
        paths = {
            "diffusion": Path("diffusion.gguf"),
            "text_encoder": Path("text.gguf"),
            "vae": Path("vae.safetensors"),
        }
        for model, diffusion_loader, text_loader in (("flux2", "UnetLoaderGGUF", "CLIPLoaderGGUF"), ("qwen21", "UNETLoader", "CLIPLoader")):
            graph = MODULE.api_workflow(model, paths, "test", 512, 512, 2, 1)
            classes = {node["class_type"] for node in graph.values()}
            self.assertIn(diffusion_loader, classes)
            self.assertIn(text_loader, classes)
            self.assertIn("SaveImage", classes)

    def test_workflows_use_profile_cfg(self):
        paths = {name: Path(name) for name in ("diffusion", "text_encoder", "vae")}
        flux = MODULE.api_workflow("flux2", paths, "test", 512, 512, 20, 1)
        qwen = MODULE.api_workflow("qwen21", paths, "test", 512, 512, 25, 1)
        self.assertEqual(4.0, flux["10"]["inputs"]["cfg"])
        self.assertEqual(1.0, qwen["5"]["inputs"]["cfg"])

    def test_workflow_accepts_ui_parameters(self):
        paths = {name: Path(name) for name in ("diffusion", "text_encoder", "vae")}
        flux = MODULE.api_workflow(
            "flux2", paths, "positive", 512, 512, 9, 123, cfg=3.5,
            negative_prompt="negative", sampler="dpmpp_2m", scheduler="flux2",
            preset="custom",
        )
        self.assertEqual("negative", flux["5"]["inputs"]["text"])
        self.assertEqual("dpmpp_2m", flux["8"]["inputs"]["sampler_name"])
        self.assertEqual(3.5, flux["10"]["inputs"]["cfg"])

        qwen = MODULE.api_workflow(
            "qwen21", paths, "positive", 512, 512, 20, 123, cfg=1,
            negative_prompt="negative", sampler="euler_ancestral", scheduler="karras",
        )
        self.assertEqual("negative", qwen["4"]["inputs"]["negative_prompt"])
        self.assertEqual("euler_ancestral", qwen["5"]["inputs"]["sampler_name"])
        self.assertEqual("karras", qwen["5"]["inputs"]["scheduler"])

    def test_workflow_nodes_include_top_level_and_subgraphs(self):
        workflow = {
            "nodes": [{"id": 1}],
            "definitions": {"subgraphs": [{"nodes": [{"id": 2}]}]},
        }
        self.assertEqual([1, 2], [node["id"] for node in MODULE.workflow_nodes(workflow)])

    def test_job_workflow_updates_outer_and_inner_parameters(self):
        template = {
            "nodes": [{"id": 1, "type": "group", "widgets_values": ["old", "", 1, 25, 1024, 1024, "euler", "simple", 0],
                       "widgets_values_named": {"prompt": "old", "steps": 25, "width": 1024, "height": 1024, "seed": 0}}],
            "definitions": {"subgraphs": [{"nodes": [
                {"id": 2, "type": "TextEncodeQwenImage21", "widgets_values": ["old", "", 1024]},
                {"id": 3, "type": "KSampler", "widgets_values": [0, "fixed", 25, 1, "euler", "simple", 1]},
            ]}]},
        }
        params = {"prompt": "new", "negative_prompt": "avoid", "width": 512, "height": 512,
                  "steps": 8, "cfg": 1.5, "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 9}
        with mock.patch.object(MODULE, "prepared_workflow", return_value=template):
            workflow = MODULE.job_workflow("qwen21", {}, **params)
        self.assertEqual(["new", "avoid", 1.5, 8, 512, 512, "dpmpp_2m", "karras", 9],
                         workflow["nodes"][0]["widgets_values"][:9])
        inner = workflow["definitions"]["subgraphs"][0]["nodes"]
        self.assertEqual(["new", "avoid", 512], inner[0]["widgets_values"])
        self.assertEqual([9, "fixed", 8, 1.5, "dpmpp_2m", "karras"], inner[1]["widgets_values"][:6])


class WebUITests(unittest.TestCase):
    def make_state(self, directory, model="flux2"):
        return webui.ImageUIState(
            1235, "http://127.0.0.1:1235", model, MODULE.MODELS[model],
            {name: Path(name) for name in ("diffusion", "text_encoder", "vae")},
            Path(directory) / "images", MODULE.api_workflow,
            Path(directory) / "history.json",
            MODULE.job_workflow,
        )

    def test_config_exposes_recommended_editable_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_state(directory).config()
        self.assertEqual("Recommended balance", config["presets"]["balanced"]["label"])
        self.assertIn("euler", config["samplers"])
        self.assertIn("steps", config["parameter_advice"])
        self.assertEqual("Diffusing", config["stage_nodes"]["11"])
        self.assertEqual("Recommended", config["option_guides"]["sampler"]["euler"]["label"])

    def test_qwen_validation_requires_square_image(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory, "qwen21")
            values = {**MODULE.MODELS["qwen21"]["presets"]["balanced"],
                      "prompt": "test", "width": 768, "height": 512, "seed": 1}
            with self.assertRaisesRegex(ValueError, "square"):
                state.validate(values)

    def test_submit_builds_and_queues_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.start_progress_monitor = mock.Mock()
            state.job_workflow_builder = mock.Mock(return_value={"id": "workflow-1", "nodes": []})
            state.backend_json = mock.Mock(side_effect=lambda _path, payload: {"prompt_id": payload["prompt_id"]})
            values = {**MODULE.MODELS["flux2"]["presets"]["balanced"],
                      "prompt": "test", "negative_prompt": "", "width": 512,
                      "height": 512, "seed": 1, "preset": "balanced"}
            job_id = state.submit(values)
            self.assertEqual(36, len(job_id))
            graph = state.backend_json.call_args.args[1]["prompt"]
            self.assertEqual(12, graph["9"]["inputs"]["steps"])
            self.assertEqual(job_id, state.backend_json.call_args.args[1]["prompt_id"])
            self.assertEqual("workflow-1", state.backend_json.call_args.args[1]
                             ["extra_data"]["extra_pnginfo"]["workflow"]["id"])
            state.start_progress_monitor.assert_called_once()

    def test_job_status_uses_comfy_job_api(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.jobs["job-1"] = {"id": "job-1", "status": "queued", "started": 0.0}
            state.backend_json = mock.Mock(return_value={
                "id": "job-1", "status": "in_progress", "workflow_id": "workflow-1",
            })
            job = state.job("job-1")
            self.assertEqual("running", job["status"])
            self.assertEqual("workflow-1", job["workflow_id"])
            self.assertEqual("http://127.0.0.1:1235/#workflow-1", job["workflow_url"])
            self.assertEqual("/api/jobs/job-1", state.backend_json.call_args.args[0])

    def test_history_deletion_removes_selected_image_only(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.output_directory.mkdir()
            first = state.output_directory / "first.png"
            second = state.output_directory / "second.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            state.history_path.write_text(json.dumps([
                {"id": "one", "image_path": str(first)},
                {"id": "two", "image_path": str(second)},
            ]))
            result = state.delete_history(["one"])
            self.assertEqual({"deleted": 1, "remaining": 1}, result)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual("two", state.history()[0]["id"])

    def test_delete_all_history_removes_all_images(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.output_directory.mkdir()
            image = state.output_directory / "only.png"
            image.write_bytes(b"image")
            state.history_path.write_text(json.dumps([{"id": "one", "image_path": str(image)}]))
            self.assertEqual(1, state.delete_history(delete_all=True)["deleted"])
            self.assertFalse(image.exists())
            self.assertEqual([], state.history())

    def test_ui_state_survives_refresh_and_is_removed_with_history(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            saved = state.save_ui_state({
                "params": {"prompt": "private draft", "steps": 8, "random_seed": True},
                "current_job": "job-1", "view_id": "image-1",
            })
            self.assertEqual(saved, self.make_state(directory).ui_state())
            state.delete_history(delete_all=True)
            self.assertEqual({}, state.ui_state())

    def test_missing_history_directory_is_recreated_on_save(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.history_path = Path(directory) / "missing" / "image-history.json"
            state.ui_state_path = state.history_path.with_name("image-ui-state.json")
            self.assertFalse(state.history_path.parent.exists())
            state.save_ui_state({"params": {"prompt": "draft"}})
            self.assertTrue(state.ui_state_path.is_file())

    def test_history_discovers_comfyui_png_and_recovers_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            state.output_directory.mkdir()
            graph = MODULE.api_workflow(
                "flux2", state.paths, "recovered prompt", 512, 512, 8, 42,
                cfg=4, negative_prompt="avoid this", sampler="euler")

            def chunk(kind, data):
                return len(data).to_bytes(4, "big") + kind + data + b"\0\0\0\0"

            image = state.output_directory / "outside-history.png"
            image.write_bytes(
                b"\x89PNG\r\n\x1a\n" +
                chunk(b"IHDR", (512).to_bytes(4, "big") * 2 + b"\x08\x02\0\0\0") +
                chunk(b"tEXt", b"prompt\0" + json.dumps(graph).encode()) +
                chunk(b"tEXt", b"workflow\0" + json.dumps({"id": "workflow-42"}).encode()) +
                chunk(b"IEND", b""))
            item = state.history()[0]
            self.assertTrue(item["id"].startswith("discovered-"))
            self.assertEqual("recovered prompt", item["prompt"])
            self.assertEqual("avoid this", item["negative_prompt"])
            self.assertEqual((8, 4.0, 42), (item["steps"], item["cfg"], item["seed"]))
            self.assertEqual("euler", item["sampler"])
            self.assertEqual("http://127.0.0.1:1235/#workflow-42", item["workflow_url"])

    def test_progress_monitor_records_steps_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.make_state(directory)
            job_id = "job-1"
            state.jobs[job_id] = {"id": job_id, "status": "queued"}
            server_socket, monitor_socket = socket.socketpair()
            progress = json.dumps({
                "type": "progress", "data": {"prompt_id": job_id, "value": 3, "max": 8},
            }).encode()
            image = b"jpeg-preview"
            frames = (
                b"\x81" + bytes([len(progress)]) + progress
                + b"\x82" + bytes([8 + len(image)]) + b"\0\0\0\1\0\0\0\1" + image
                + b"\x88\0"
            )
            server_socket.sendall(frames)
            server_socket.close()
            state.monitor_progress(job_id, monitor_socket)
            self.assertEqual(3, state.jobs[job_id]["progress"]["value"])
            self.assertEqual(8, state.jobs[job_id]["progress"]["max"])
            self.assertIn("elapsed_seconds", state.jobs[job_id]["progress"])
            self.assertEqual(1, state.jobs[job_id]["preview_version"])
            self.assertEqual((image, "image/jpeg"), state.preview(job_id))

    def test_html_contains_simple_and_advanced_controls(self):
        self.assertIn("Advanced parameters", webui.HTML)
        self.assertIn("Reset recommended parameters", webui.HTML)
        self.assertIn("Generation progress", webui.HTML)
        self.assertIn("Live diffusion preview", webui.HTML)
        self.assertIn("about ${duration(remaining)} left", webui.HTML)
        self.assertIn("Sampler and scheduler guide", webui.HTML)
        self.assertIn("Delete selected", webui.HTML)
        self.assertIn("Copy path", webui.HTML)
        self.assertIn("Back to current generation", webui.HTML)
        self.assertIn('maxlength="4000"', webui.HTML)
        self.assertNotIn("localStorage", webui.HTML)


if __name__ == "__main__":
    unittest.main()
