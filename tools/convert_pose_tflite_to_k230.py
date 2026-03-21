#!/usr/bin/env python3
"""Compile the pose TFLite model into a K230 kmodel using nncase."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_SHAPE = [1, 256, 256, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TFLite pose model into a K230 .kmodel.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmark_full.tflite",
        help="Path to the source TFLite model.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Directory containing representative RGB calibration images.",
    )
    parser.add_argument(
        "--output",
        default="models/pose_landmark_full.kmodel",
        help="Path to the output .kmodel file.",
    )
    parser.add_argument(
        "--dump-dir",
        default="tmp/k230_pose_landmark_full_tflite",
        help="Directory for nncase dump artifacts.",
    )
    parser.add_argument(
        "--target",
        default="k230",
        help="nncase compilation target. Defaults to k230.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Maximum number of calibration images to use.",
    )
    parser.add_argument(
        "--keep-fp32-input",
        action="store_true",
        help="Compile with float32 input instead of uint8 preprocessing.",
    )
    parser.add_argument(
        "--input-shape",
        default="1,256,256,3",
        help="Model input shape as comma-separated integers. Defaults to 1,256,256,3.",
    )
    return parser.parse_args()


def parse_input_shape(raw_shape: str) -> list[int]:
    dims = [int(part.strip()) for part in raw_shape.split(",") if part.strip()]
    if len(dims) != 4:
        raise ValueError(f"Expected 4 input dimensions, got {dims}")
    return dims


def list_calibration_images(dataset_dir: Path) -> list[Path]:
    image_paths = sorted(
        path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {dataset_dir}")
    return image_paths


def resize_image(image: Image.Image, input_shape: list[int]) -> np.ndarray:
    if input_shape[-1] == 3:
        resized = image.resize((input_shape[2], input_shape[1]), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.uint8)
    elif input_shape[1] == 3:
        resized = image.resize((input_shape[3], input_shape[2]), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
    else:
        raise RuntimeError(f"Unsupported input shape for RGB preprocessing: {input_shape}")

    return array[np.newaxis, ...]


def generate_calibration_tensors(dataset_dir: Path, input_shape: list[int], sample_limit: int) -> list[list[np.ndarray]]:
    image_paths = list_calibration_images(dataset_dir)
    sample_paths = image_paths[: min(sample_limit, len(image_paths))]
    tensors: list[list[np.ndarray]] = []
    for image_path in sample_paths:
        with Image.open(image_path) as image:
            tensors.append([resize_image(image.convert("RGB"), input_shape)])
    return tensors


def ensure_nncase() -> tuple[object, object]:
    try:
        import nncase  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "nncase is not installed. Run tools/setup_k230_env.sh and source tools/activate_k230_env.sh first."
        ) from exc

    try:
        import nncase_kpu  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "nncase-kpu is not installed. Install the same version as nncase before targeting k230."
        ) from exc

    return nncase, nncase_kpu  # type: ignore[name-defined]


def configure_compile_options(nncase_module: object, input_shape: list[int], use_fp32_input: bool, dump_dir: Path, target: str):
    compile_options = nncase_module.CompileOptions()
    compile_options.target = target
    compile_options.preprocess = True
    compile_options.input_shape = input_shape
    compile_options.swapRB = False
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = str(dump_dir)
    compile_options.input_layout = "NHWC" if input_shape[-1] == 3 else "NCHW"

    if use_fp32_input:
        compile_options.input_type = "float32"
        compile_options.mean = [0.0, 0.0, 0.0]
        compile_options.std = [1.0, 1.0, 1.0]
    else:
        compile_options.input_type = "uint8"
        compile_options.input_range = [0, 255]
        compile_options.mean = [0.0, 0.0, 0.0]
        compile_options.std = [255.0, 255.0, 255.0]

    return compile_options


def compile_kmodel(
    nncase_module: object,
    model_path: Path,
    output_path: Path,
    dump_dir: Path,
    input_shape: list[int],
    calibration_tensors: list[list[np.ndarray]],
    use_fp32_input: bool,
    target: str,
) -> None:
    compile_options = configure_compile_options(nncase_module, input_shape, use_fp32_input, dump_dir, target)
    compiler = nncase_module.Compiler(compile_options)

    import_options = nncase_module.ImportOptions()
    compiler.import_tflite(model_path.read_bytes(), import_options)

    ptq_options = nncase_module.PTQTensorOptions()
    ptq_options.samples_count = len(calibration_tensors)
    ptq_options.set_tensor_data(calibration_tensors)
    ptq_options.quant_type = "uint8"
    ptq_options.w_quant_type = "uint8"
    ptq_options.finetune_weights_method = "NoFineTuneWeights"
    ptq_options.export_quant_scheme = True
    ptq_options.export_weight_range_by_channel = True
    compiler.use_ptq(ptq_options)

    compiler.compile()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(compiler.gencode_tobytes())


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dump_dir = Path(args.dump_dir).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Calibration dataset directory not found: {dataset_dir}")

    input_shape = parse_input_shape(args.input_shape)
    dump_dir.mkdir(parents=True, exist_ok=True)

    nncase_module, _ = ensure_nncase()
    calibration_tensors = generate_calibration_tensors(dataset_dir, input_shape, args.samples)

    compile_kmodel(
        nncase_module=nncase_module,
        model_path=model_path,
        output_path=output_path,
        dump_dir=dump_dir,
        input_shape=input_shape,
        calibration_tensors=calibration_tensors,
        use_fp32_input=args.keep_fp32_input,
        target=args.target,
    )

    print(f"Generated K230 model: {output_path}")
    print(f"Calibration samples used: {len(calibration_tensors)}")
    print(f"Model input shape: {input_shape}")
    print(f"Dump directory: {dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())