"""Lightweight regression tests for pipeline contracts and deterministic layout."""

from __future__ import annotations

import tempfile
import unittest
import os
import subprocess
import sys
import types
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(find_spec("omegaconf") and find_spec("torch"), "HairPort runtime dependencies not installed")
class RuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from hairport.config import load_config, reset_config, set_config

        self.reset_config = reset_config
        reset_config()
        set_config(load_config())

    def tearDown(self) -> None:
        self.reset_config()

    def test_official_landmark_mode_rejects_perturbations(self) -> None:
        from hairport.config import load_config

        with self.assertRaisesRegex(ValueError, "single frontal-view"):
            load_config(overrides=["landmark_3d.num_perturbations=1"])

    def test_versioned_and_conditioned_artifact_paths_do_not_collide(self) -> None:
        from hairport.data import DatasetManager

        dm = DatasetManager("/tmp/hairport")
        self.assertNotEqual(
            dm.rendered_view_file("target", "source", "w_seg"),
            dm.rendered_view_file("target", "source", "wo_seg"),
        )
        self.assertNotEqual(
            dm.hair_restored_file("target", "source", "w_seg", "3d_aware", "enhanced"),
            dm.hair_restored_file("target", "source", "w_seg", "3d_aware", "blended"),
        )

    def test_enhanced_phase_paths_are_explicit_and_validated(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.data import DatasetManager

        set_config(load_config(overrides=["enhance_view.conditioning_phase=2"]))
        dm = DatasetManager("/tmp/hairport")
        self.assertEqual(
            dm.enhanced_view_file("target", "source", "w_seg", phase=2).name,
            "target_image_phase_2.png",
        )
        self.assertEqual(
            dm.enhanced_view_mask_file("target", "source", "w_seg", phase=2).name,
            "target_image_phase_2_mask.png",
        )
        with self.assertRaisesRegex(ValueError, "phase must be 1 or 2"):
            dm.enhanced_view_file("target", "source", "w_seg", phase=3)

    def test_invalid_enhanced_conditioning_phase_is_rejected(self) -> None:
        from hairport.config import load_config

        with self.assertRaisesRegex(ValueError, "conditioning_phase must be 1 or 2"):
            load_config(overrides=["enhance_view.conditioning_phase=3"])

    def test_seed_derivation_is_stable_and_variant_specific(self) -> None:
        from hairport.runtime import derive_seed

        args = (42, "transfer_hair", "target_to_source", "w_seg", "enhanced", "generate")
        self.assertEqual(derive_seed(*args), derive_seed(*args))
        self.assertNotEqual(derive_seed(*args), derive_seed(42, "transfer_hair", "target_to_source", "w_seg", "blended", "generate"))

    def test_provenance_controls_cache_reuse(self) -> None:
        from hairport.runtime import build_provenance, can_reuse_artifact, write_provenance

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.txt"
            artifact = Path(tmp) / "output.txt"
            source.write_text("source")
            artifact.write_text("result")
            revisions = {"repository": "test", "dirty": False, "modules": {}}
            with patch("hairport.runtime._code_revisions", return_value=revisions):
                expected = build_provenance("test", seed=1, inputs=[source])
                self.assertFalse(can_reuse_artifact(artifact, expected))
                write_provenance(artifact, expected)
                self.assertTrue(can_reuse_artifact(artifact, expected))
                source.write_text("changed")
                changed = build_provenance("test", seed=1, inputs=[source])
                self.assertFalse(can_reuse_artifact(artifact, changed))

    def test_pipeline_reports_returned_failures_and_unloads(self) -> None:
        from hairport.pipeline import HairPortPipeline
        from hairport.runtime import StageSummary

        state = {"unloaded": False}

        class FailedStage:
            def __init__(self, **_kwargs):
                pass

            def run(self, **_kwargs):
                result = StageSummary(attempted=1)
                result.add_failure("sample", "failed")
                return result

            def unload(self):
                state["unloaded"] = True

        pipeline = HairPortPipeline(data_dir="/tmp/hairport", preflight=False)
        with patch("hairport.pipeline._get_stage_class", return_value=FailedStage):
            results = pipeline.run(only=["baldify"])
        self.assertFalse(results[0].success)
        self.assertTrue(state["unloaded"])

    def test_pair_decisions_do_not_mutate_input_csv(self) -> None:
        from hairport.view_aligner import Config, prepare_pairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "pairs.csv"
            csv_path.write_text("target_id,source_id\nreference,source\n")
            original = csv_path.read_text()
            manifest_path = root / "pair_decisions.json"
            with patch("hairport.view_aligner.compute_lift_3d", return_value=(True, 0.5)):
                pairs = prepare_pairs(
                    str(root), Config(), decision_manifest_path=manifest_path
                )
            self.assertEqual(pairs, [("reference", "source", True)])
            self.assertEqual(csv_path.read_text(), original)
            self.assertTrue(manifest_path.exists())

    def test_non_lift_pairs_skip_render_and_enhance_without_loading_models(self) -> None:
        from hairport.stages.enhance_view import EnhanceViewStage
        from hairport.stages.render_view import RenderViewStage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = (
                root
                / "view_aligned"
                / "shape_hi3dgen__texture_mvadapter"
                / "target_to_source"
                / "w_seg"
                / "alignment"
            )
            pair_dir.mkdir(parents=True)

            with patch.object(RenderViewStage, "_ensure_generator", side_effect=AssertionError("loaded generator")):
                render_summary = RenderViewStage(device="cpu").run(root, bald_version="w_seg")
            with patch.object(EnhanceViewStage, "_ensure_enhancer", side_effect=AssertionError("loaded enhancer")):
                enhance_summary = EnhanceViewStage(device="cpu").run(root, bald_version="w_seg")

            self.assertEqual(render_summary.attempted, 1)
            self.assertEqual(render_summary.skipped, 1)
            self.assertTrue(render_summary.success)
            self.assertEqual(enhance_summary.attempted, 1)
            self.assertEqual(enhance_summary.skipped, 1)
            self.assertTrue(enhance_summary.success)

    def test_shape_mesh_mvadapter_command_uses_absolute_paths(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.stages.shape_mesh import ShapeMeshStage

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_dir = workspace / "relative_outputs"
            mv_module = workspace / "modules" / "MV-Adapter"
            mv_module.mkdir(parents=True)
            (data_dir / "shape_mesh" / "target").mkdir(parents=True)
            (data_dir / "shape_mesh" / "target" / "shape_mesh.glb").write_text("mesh")
            (data_dir / "matted_image_centered").mkdir(parents=True)
            (data_dir / "matted_image_centered" / "target.png").write_text("image")
            (data_dir / "prompt").mkdir(parents=True)
            (data_dir / "prompt" / "target.json").write_text("{}")
            (data_dir / "pairs.csv").write_text("target_id,source_id\ntarget,source\n")

            cfg = load_config(
                overrides=[
                    f"paths.mv_adapter_module={mv_module}",
                    "pipeline.shape_provider=hi3dgen",
                    "pipeline.texture_provider=mvadapter",
                ]
            )
            set_config(cfg)
            old_cwd = Path.cwd()
            captured = {}

            def fake_run(cmd, cwd=None, capture_output=None, text=None):
                data_arg = Path(cmd[cmd.index("--data_dir") + 1])
                prompt_arg = Path(cmd[cmd.index("--prompt_dir") + 1])
                output_arg = Path(cmd[cmd.index("--output_dir") + 1])
                image_link = data_arg / "matted_image_centered" / "target.png"
                mesh_link = data_arg / "hi3dgen" / "target" / "shape_mesh.glb"
                captured.update(
                    data_arg=data_arg,
                    prompt_arg=prompt_arg,
                    output_arg=output_arg,
                    image_link=image_link,
                    mesh_link=mesh_link,
                    cwd=Path(cwd),
                )
                for link_path in (image_link, mesh_link):
                    self.assertTrue(link_path.exists())
                    if link_path.is_symlink():
                        self.assertTrue(Path(os.readlink(link_path)).is_absolute())
                output_mesh = output_arg / "hi3dgen" / "target" / "textured_mesh.glb"
                output_mesh.parent.mkdir(parents=True, exist_ok=True)
                output_mesh.write_text("textured")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            try:
                os.chdir(workspace)
                with patch("hairport.stages.shape_mesh.subprocess.run", side_effect=fake_run):
                    summary = ShapeMeshStage(device="cpu", seed=123).run("relative_outputs")
            finally:
                os.chdir(old_cwd)

            self.assertTrue(captured["data_arg"].is_absolute())
            self.assertTrue(captured["prompt_arg"].is_absolute())
            self.assertTrue(captured["output_arg"].is_absolute())
            self.assertEqual(captured["cwd"], mv_module)
            self.assertEqual(summary.completed, 1)

    def test_bald_konverter_uses_configured_flame_paths_by_default(self) -> None:
        from hairport.bald_konverter.pipeline import BaldKonverterPipeline
        from hairport.config import get_config

        pipeline = BaldKonverterPipeline(mode="wo_seg", device="cpu", use_flame=True)
        with patch("hairport.bald_konverter.preprocessing.flame.FLAMESegmenter") as constructor:
            pipeline._get_flame_segmenter()
        cfg = get_config()
        self.assertEqual(
            constructor.call_args.kwargs["flame_model_runtime"],
            cfg.paths.flame_model_runtime,
        )
        self.assertEqual(
            constructor.call_args.kwargs["flame_model_source"],
            cfg.paths.flame_model_source,
        )
        self.assertEqual(
            constructor.call_args.kwargs["flame_masks_path"],
            cfg.paths.flame_masks,
        )
        self.assertEqual(
            constructor.call_args.kwargs["flame_eyelids_path"],
            cfg.paths.flame_eyelids,
        )

    def test_blend_hair_stage_uses_configured_enhanced_phase_in_provenance(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.stages.blend_hair import BlendHairStage

        set_config(load_config(overrides=["device=cpu", "enhance_view.conditioning_phase=2"]))

        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

        fake_view_blender = types.ModuleType("hairport.view_blender")
        fake_view_blender.BlendingConfig = type(
            "BlendingConfig",
            (),
            {"__init__": lambda self: setattr(self, "SAM_CONFIDENCE_THRESHOLD", 0.4)},
        )
        fake_view_blender.process_view_aligned_folder = lambda **_kwargs: False
        fake_view_blender.flush = lambda: None

        fake_core = types.ModuleType("hairport.core")
        fake_core.BackgroundRemover = DummyModel
        fake_core.CodeFormerEnhancer = DummyModel
        fake_core.FacialLandmarkDetector = DummyModel
        fake_core.SAMMaskExtractor = DummyModel
        fake_core.FLAMEFitter = DummyModel

        captured_inputs = []

        def fake_build_provenance(_stage, *, seed, inputs, metadata):
            captured_inputs.extend(Path(path) for path in inputs)
            return {"signature": "test"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (
                root
                / "view_aligned"
                / "shape_hi3dgen__texture_mvadapter"
                / "target_to_source"
                / "w_seg"
                / "alignment"
            ).mkdir(parents=True)
            with (
                patch.dict(sys.modules, {"hairport.view_blender": fake_view_blender, "hairport.core": fake_core}),
                patch("hairport.stages.blend_hair.build_provenance", side_effect=fake_build_provenance),
                patch("hairport.stages.blend_hair.can_reuse_artifact", return_value=False),
            ):
                BlendHairStage().run(root, bald_version="w_seg", seed=123)

        self.assertTrue(any(path.name == "target_image_phase_2.png" for path in captured_inputs))

    def test_pipeline_defaults_data_dir_from_config(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.pipeline import HairPortPipeline

        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "configured_outputs"
            set_config(load_config(overrides=[f"paths.output_dir={expected}"]))
            pipeline = HairPortPipeline(preflight=False)
        self.assertEqual(pipeline.ctx.data_dir, expected)

    def test_config_defaults_use_new_flame_asset_layout(self) -> None:
        from hairport.config import get_config

        cfg = get_config()
        self.assertIn("assets/base_models/flame/parametric_models/generic_model.pkl", cfg.paths.flame_model_source)
        self.assertIn("assets/base_models/flame/parametric_models/generic_model.pt", cfg.paths.flame_model_runtime)
        self.assertIn("assets/base_models/flame/vertex_mappings/FLAME_masks.pkl", cfg.paths.flame_masks)
        self.assertIn("assets/landmarks/flame/eyelids.pt", cfg.paths.flame_eyelids)
        self.assertIn("assets/landmarks/flame/mediapipe_landmark_embedding.npz", cfg.paths.mediapipe_flame_embedding)

    def test_pipeline_stage_selection_rejects_unknown_only_and_skip(self) -> None:
        from hairport.pipeline import HairPortPipeline

        with self.assertRaisesRegex(ValueError, "Unknown stage\\(s\\) in only"):
            HairPortPipeline._resolve_stages(only=["not_a_stage"])
        with self.assertRaisesRegex(ValueError, "Unknown stage\\(s\\) in skip"):
            HairPortPipeline._resolve_stages(skip=["still_not_a_stage"])

    def test_pipeline_stage_selection_rejects_end_before_start(self) -> None:
        from hairport.pipeline import HairPortPipeline

        with self.assertRaisesRegex(ValueError, "occurs before start stage"):
            HairPortPipeline._resolve_stages(start="render_view", end="caption")

    def test_projection_helper_uses_points_device_and_nonsquare_dimensions(self) -> None:
        import torch
        from hairport.utility.estimate_camera import project_points_blender_ortho

        points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        cam_location = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        cam_rotvec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        ortho_scale = torch.tensor(2.0, dtype=torch.float32)
        projected = project_points_blender_ortho(
            points_world=points,
            cam_location=cam_location,
            cam_rotvec=cam_rotvec,
            ortho_scale=ortho_scale,
            image_size=(640, 320),
        )
        self.assertEqual(projected.device, points.device)
        self.assertAlmostEqual(float(projected[0, 0]), 320.0, places=4)
        self.assertAlmostEqual(float(projected[0, 1]), 160.0, places=4)

    def test_camera_single_run_forwards_nonsquare_image_size(self) -> None:
        import torch
        import hairport.utility.estimate_camera as estimate_camera

        captured = {}

        def fake_project(points_world, cam_location, cam_rotvec, ortho_scale, image_size):
            captured["image_size"] = image_size
            return cam_location[:2].view(1, 2).expand(points_world.shape[0], 2)

        xyz_ref = torch.zeros((1, 3), dtype=torch.float32)
        uv_input = torch.zeros((1, 2), dtype=torch.float32)
        with patch.object(estimate_camera, "project_points_blender_ortho", side_effect=fake_project):
            estimate_camera.estimate_camera_single_run(
                xyz_ref=xyz_ref,
                uv_input=uv_input,
                init_camera_location=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                init_camera_rotation=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                init_ortho_scale=1.0,
                num_iters=1,
                width=640,
                height=320,
                device="cpu",
                early_stopping=False,
                min_iterations=0,
                ortho_scale_freeze_iters=0,
            )
        self.assertEqual(captured["image_size"], (640, 320))

    def test_landmark_stage_head_orientation_uses_cache_by_default(self) -> None:
        from hairport.stages.landmark_3d import Landmark3DStage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image" / "person.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_text("placeholder")

            with patch(
                "hairport.core.flame_fitting.compute_head_orientation",
                return_value={"euler_angles_xyz_radians": [[0.1, 0.2, 0.3]]},
            ) as compute_orientation:
                angles = Landmark3DStage._load_head_orientation(root, "person")

        self.assertEqual(angles, [0.1, 0.2, 0.3])
        self.assertFalse(compute_orientation.call_args.kwargs["force"])

    def test_preflight_requires_importable_sheap_for_flame_stages(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.preflight import validate_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flame_root = root / "base_models" / "flame"
            flame_model_dir = flame_root / "parametric_models"
            flame_masks_dir = flame_root / "vertex_mappings"
            landmark_dir = root / "landmarks" / "flame"
            sheap_dir = root / "SHeaP"
            embedding = root / "embedding.npz"
            flame_model_dir.mkdir(parents=True)
            flame_masks_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)
            sheap_dir.mkdir(parents=True)
            (flame_model_dir / "generic_model.pt").write_text("model")
            (flame_model_dir / "generic_model.pkl").write_text("pkl")
            (flame_masks_dir / "FLAME_masks.pkl").write_text("mask")
            (landmark_dir / "eyelids.pt").write_text("eyelids")
            embedding.write_text("embed")
            set_config(
                load_config(
                    overrides=[
                        f"paths.flame_model_runtime={flame_model_dir / 'generic_model.pt'}",
                        f"paths.flame_model_source={flame_model_dir / 'generic_model.pkl'}",
                        f"paths.flame_masks={flame_masks_dir / 'FLAME_masks.pkl'}",
                        f"paths.flame_eyelids={landmark_dir / 'eyelids.pt'}",
                        f"paths.sheap_module={sheap_dir}",
                        f"paths.mediapipe_flame_embedding={embedding}",
                    ]
                )
            )
            with patch("hairport.preflight.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "importable Python package 'sheap'"):
                    validate_preflight(["align_view"])

    def test_preflight_converts_flame_model_from_pkl_when_pt_missing(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.preflight import validate_preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flame_model_dir = root / "base_models" / "flame" / "parametric_models"
            flame_masks_dir = root / "base_models" / "flame" / "vertex_mappings"
            landmark_dir = root / "landmarks" / "flame"
            sheap_dir = root / "SHeaP"
            flame_model_dir.mkdir(parents=True)
            flame_masks_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)
            sheap_dir.mkdir(parents=True)
            source_path = flame_model_dir / "generic_model.pkl"
            runtime_path = flame_model_dir / "generic_model.pt"
            source_path.write_text("placeholder")
            (flame_masks_dir / "FLAME_masks.pkl").write_text("mask")
            (landmark_dir / "eyelids.pt").write_text("eyelids")
            (landmark_dir / "mediapipe_landmark_embedding.npz").write_text("embed")
            set_config(
                load_config(
                    overrides=[
                        f"paths.flame_model_source={source_path}",
                        f"paths.flame_model_runtime={runtime_path}",
                        f"paths.flame_masks={flame_masks_dir / 'FLAME_masks.pkl'}",
                        f"paths.flame_eyelids={landmark_dir / 'eyelids.pt'}",
                        f"paths.sheap_module={sheap_dir}",
                        f"paths.mediapipe_flame_embedding={landmark_dir / 'mediapipe_landmark_embedding.npz'}",
                    ]
                )
            )
            with (
                patch("hairport.preflight.importlib.util.find_spec", return_value=object()),
                patch("hairport.preflight.ensure_flame_model_runtime", return_value=runtime_path) as ensure_runtime,
            ):
                validate_preflight(["align_view"])
            ensure_runtime.assert_called_once_with(source_path, runtime_path)

    def test_blend_hair_stage_counts_non_lift_outputs_as_completed(self) -> None:
        from hairport.config import load_config, set_config
        from hairport.stages.blend_hair import BlendHairStage

        set_config(load_config(overrides=["device=cpu"]))

        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

        fake_view_blender = types.ModuleType("hairport.view_blender")
        fake_view_blender.BlendingConfig = type(
            "BlendingConfig",
            (),
            {"__init__": lambda self: setattr(self, "SAM_CONFIDENCE_THRESHOLD", 0.4)},
        )
        fake_view_blender.flush = lambda: None

        def fake_process_view_aligned_folder(*, folder_path, bald_version, **_kwargs):
            output = Path(folder_path) / bald_version / "3d_unaware" / "blending" / "poisson_blended.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("ok")
            return True

        fake_view_blender.process_view_aligned_folder = fake_process_view_aligned_folder

        fake_core = types.ModuleType("hairport.core")
        fake_core.BackgroundRemover = DummyModel
        fake_core.CodeFormerEnhancer = DummyModel
        fake_core.FacialLandmarkDetector = DummyModel
        fake_core.SAMMaskExtractor = DummyModel
        fake_core.FLAMEFitter = DummyModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (
                root
                / "view_aligned"
                / "shape_hi3dgen__texture_mvadapter"
                / "target_to_source"
                / "w_seg"
                / "source_outpainted"
            ).mkdir(parents=True)
            with (
                patch.dict(sys.modules, {"hairport.view_blender": fake_view_blender, "hairport.core": fake_core}),
                patch("hairport.stages.blend_hair.can_reuse_artifact", return_value=False),
            ):
                summary = BlendHairStage().run(root, bald_version="w_seg", seed=123)

        self.assertEqual(summary.attempted, 1)
        self.assertEqual(summary.completed, 1)
        self.assertEqual(summary.skipped, 0)
        self.assertTrue(summary.success)
        self.assertTrue(any(path.endswith("3d_unaware/blending/poisson_blended.png") for path in summary.artifacts))

    def test_install_scripts_use_https_git_remotes(self) -> None:
        setup_script = Path(__file__).resolve().parents[1] / "scripts" / "setup_submodules.sh"
        install_script = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"
        setup_text = setup_script.read_text()
        install_text = install_script.read_text()

        self.assertNotIn("git@github.com:", setup_text)
        self.assertNotIn("git@github.com:", install_text)
        self.assertIn("https://github.com/deepmancer/CodeFormer.git", setup_text)
        self.assertIn("https://github.com/deepmancer/MV-Adapter.git", setup_text)
        self.assertIn("https://github.com/deepmancer/SHeaP.git", setup_text)
        self.assertIn("https://github.com/huggingface/diffusers.git", install_text)
        self.assertIn("https://github.com/huggingface/transformers.git", install_text)
        self.assertIn("https://github.com/deepmancer/easy_dwpose.git", install_text)

    def test_captioner_retries_without_flash_attention_when_unavailable(self) -> None:
        try:
            import hairport.core.captioner as captioner
        except Exception as exc:
            self.skipTest(f"captioner dependencies not importable: {exc}")

        class DummyTokenizer:
            padding_side = "right"

        class DummyProcessor:
            tokenizer = DummyTokenizer()

        with (
            patch.object(captioner.Qwen3VLForConditionalGeneration, "from_pretrained") as from_pretrained,
            patch.object(captioner.AutoProcessor, "from_pretrained", return_value=DummyProcessor()),
        ):
            from_pretrained.side_effect = [ImportError("flash_attn missing"), object()]
            pipeline = captioner.CaptionerPipeline(device_map="cpu", use_flash_attention=True)

        self.assertIsNotNone(pipeline.model)
        self.assertEqual(from_pretrained.call_count, 2)
        first_kwargs = from_pretrained.call_args_list[0].kwargs
        second_kwargs = from_pretrained.call_args_list[1].kwargs
        self.assertEqual(first_kwargs["attn_implementation"], "flash_attention_2")
        self.assertNotIn("attn_implementation", second_kwargs)

    def test_captioner_auto_flash_attention_does_not_require_flash_attn_package(self) -> None:
        try:
            import hairport.core.captioner as captioner
        except Exception as exc:
            self.skipTest(f"captioner dependencies not importable: {exc}")

        class DummyProcessor:
            tokenizer = type("DummyTokenizer", (), {"padding_side": "right"})()

        with (
            patch.object(captioner, "find_spec", return_value=None),
            patch.object(captioner.Qwen3VLForConditionalGeneration, "from_pretrained", return_value=object()) as from_pretrained,
            patch.object(captioner.AutoProcessor, "from_pretrained", return_value=DummyProcessor()),
        ):
            captioner.CaptionerPipeline(device_map="cpu")

        self.assertNotIn("attn_implementation", from_pretrained.call_args.kwargs)

    def test_dwpose_singleton_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        try:
            import hairport.postprocessing.restore_hair_klein as transfer
        except Exception as exc:
            self.skipTest(f"transfer dependencies not importable: {exc}")

        transfer.DWPoseSingleton._instance = None
        with (
            patch.object(transfer.torch.cuda, "is_available", return_value=False),
            patch.object(transfer, "DWposeDetector", return_value=object()) as constructor,
        ):
            transfer.DWPoseSingleton.get_instance(device="cuda")
        transfer.DWPoseSingleton._instance = None
        self.assertEqual(constructor.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
