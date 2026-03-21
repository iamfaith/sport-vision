#!/usr/bin/env python3
"""Re-export a MediaPipe pose TFLite model as a pure float32 TFLite model."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

FLATC_URL = (
    "https://github.com/google/flatbuffers/releases/download/"
    "v25.12.19-2026-02-06-03fffb2/Linux.flatc.binary.g%2B%2B-13.zip"
)
SCHEMA_URL = (
    "https://raw.githubusercontent.com/tensorflow/tensorflow/master/"
    "tensorflow/compiler/mlir/lite/schema/schema.fbs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export a MediaPipe pose TFLite model into a plain float32 TFLite model.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmark_full.tflite",
        help="Path to the source TFLite model.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/pose_landmark_full_float32",
        help="Directory where the re-exported model and intermediate files will be written.",
    )
    parser.add_argument(
        "--flatc-path",
        default="tools/flatbuffers/flatc",
        help="Path where the flatc binary should exist or be downloaded to.",
    )
    parser.add_argument(
        "--schema-path",
        default="tools/tflite_schema/schema.fbs",
        help="Path where schema.fbs should exist or be downloaded to.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    destination.write_bytes(data)


def ensure_flatc(flatc_path: Path) -> Path:
    if flatc_path.exists():
        make_executable(flatc_path)
        return flatc_path

    archive_path = flatc_path.with_suffix(".zip")
    download_file(FLATC_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(flatc_path.parent)
    archive_path.unlink(missing_ok=True)

    extracted_path = flatc_path.parent / "flatc"
    if not extracted_path.exists():
        raise RuntimeError(f"flatc was not found after extracting {archive_path}")
    make_executable(extracted_path)
    return extracted_path


def ensure_schema(schema_path: Path) -> Path:
    if schema_path.exists():
        return schema_path
    download_file(SCHEMA_URL, schema_path)
    return schema_path


def make_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_base_command(model_path: Path, output_dir: Path, flatc_path: Path, schema_path: Path) -> list[str]:
    return [
        "--model_path",
        model_path.name,
        "--flatc_path",
        str(flatc_path),
        "--schema_path",
        str(schema_path),
        "--model_output_path",
        str(output_dir),
    ]


@contextlib.contextmanager
def temporary_argv(argv: list[str]):
    previous = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


@contextlib.contextmanager
def temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def force_cpu_tensorflow() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def patch_interpreter_compatibility() -> None:
    try:
        from tensorflow.lite.python.interpreter import Interpreter as TfInterpreter

        original = TfInterpreter._get_tensor_details

        def compat_get_tensor_details(self, tensor_index, subgraph_index=0):
            return original(self, tensor_index, subgraph_index)

        TfInterpreter._get_tensor_details = compat_get_tensor_details
    except Exception:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter as RuntimeInterpreter  # type: ignore

        original_runtime = RuntimeInterpreter._get_tensor_details

        def compat_runtime_get_tensor_details(self, tensor_index, subgraph_index=0):
            return original_runtime(self, tensor_index, subgraph_index)

        RuntimeInterpreter._get_tensor_details = compat_runtime_get_tensor_details  # type: ignore[attr-defined]
    except Exception:
        pass


def patch_tflite2tensorflow_source() -> None:
    spec = importlib.util.find_spec("tflite2tensorflow.tflite2tensorflow")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("tflite2tensorflow.tflite2tensorflow")

    module_path = Path(spec.origin)
    source = module_path.read_text(encoding="utf-8")
    updated = source

    replacements = [
        (
            "    import tensorflow as tfv2\n    from tensorflow.keras.layers import Layer\n",
            "    import tensorflow as tfv2\n"
            "    try:\n"
            "        tfv2.config.set_visible_devices([], 'GPU')\n"
            "    except Exception:\n"
            "        pass\n"
            "    from tensorflow.keras.layers import Layer\n",
        ),
        (
            "    return op_name\n",
            "    op_name = re.sub(r'[^0-9A-Za-z_.-]+', '_', op_name)\n"
            "    return op_name\n",
        ),
        (
            "        import tensorflow.compat.v1 as tf\n        try:\n",
            "        import tensorflow.compat.v1 as tf\n"
            "        try:\n"
            "            import tensorflow as tfv2_runtime\n"
            "            tfv2_runtime.config.set_visible_devices([], 'GPU')\n"
            "        except Exception:\n"
            "            pass\n"
            "        try:\n",
        ),
        (
            "            config = tf.ConfigProto()\n            config.gpu_options.allow_growth = True\n",
            "            config = tf.ConfigProto()\n"
            "            config.device_count['GPU'] = 0\n"
            "            config.gpu_options.visible_device_list = ''\n"
            "            config.gpu_options.allow_growth = False\n",
        ),
        (
            "        import tensorflow as tf\n        import tensorflow_datasets as tfds\n",
            "        import tensorflow as tf\n"
            "        try:\n"
            "            tf.config.set_visible_devices([], 'GPU')\n"
            "        except Exception:\n"
            "            pass\n"
            "        import tensorflow_datasets as tfds\n",
        ),
    ]

    for old, new in replacements:
        if new in updated:
            continue
        if old not in updated:
            continue
        updated = updated.replace(old, new, 1)

    if updated != source:
        module_path.write_text(updated, encoding="utf-8")
        importlib.invalidate_caches()


def run_tflite2tensorflow(argv: list[str], cwd: Path) -> None:
    force_cpu_tensorflow()
    patch_tflite2tensorflow_source()
    patch_interpreter_compatibility()
    module = importlib.import_module("tflite2tensorflow.tflite2tensorflow")
    with temporary_cwd(cwd), temporary_argv(["tflite2tensorflow", *argv]):
        try:
            module.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code not in (0, None):
                raise RuntimeError(f"tflite2tensorflow failed with exit code {code}") from exc


def convert_saved_model_to_float32_tflite(saved_model_dir: Path, output_path: Path) -> None:
    conversion_script = """
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

saved_model_dir = sys.argv[1]
output_path = sys.argv[2]

loaded = tf.saved_model.load(saved_model_dir)
serving_function = loaded.signatures['serving_default']
converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_function], loaded)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

with open(output_path, 'wb') as file:
    file.write(converter.convert())
"""
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    subprocess.run(
        [sys.executable, "-c", conversion_script, str(saved_model_dir), str(output_path)],
        check=True,
        env=environment,
    )


def run_reexport(model_path: Path, output_dir: Path, flatc_path: Path, schema_path: Path) -> Path:
    shutil.rmtree(output_dir, ignore_errors=True)
    working_directory = model_path.parent

    saved_model_command = build_base_command(model_path, output_dir, flatc_path, schema_path)
    saved_model_command.append("--output_pb")
    run_tflite2tensorflow(saved_model_command, working_directory)

    exported_model = output_dir / "model_float32.tflite"
    convert_saved_model_to_float32_tflite(output_dir, exported_model)
    if not exported_model.exists():
        raise FileNotFoundError(f"Expected re-exported model not found: {exported_model}")
    return exported_model


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    flatc_path = Path(args.flatc_path).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Source TFLite model not found: {model_path}")

    flatc_path = ensure_flatc(flatc_path)
    schema_path = ensure_schema(schema_path)
    exported_model = run_reexport(model_path, output_dir, flatc_path, schema_path)

    print(f"flatc: {flatc_path}")
    print(f"schema: {schema_path}")
    print(f"float32 tflite: {exported_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())