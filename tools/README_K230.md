# K230 Model Conversion

This repository does not include the K230 compiler toolchain. Prefer converting [models/pose_landmark_full.tflite](/home/faith/sport-vision/models/pose_landmark_full.tflite) into a K230 `.kmodel` with [tools/convert_pose_tflite_to_k230.py](/home/faith/sport-vision/tools/convert_pose_tflite_to_k230.py), but MediaPipe pose TFLite models in this repo must first be re-exported as plain float32 TFLite because they contain many `DEQUANTIZE` weight nodes.

## Verified model facts

- Input tensor: `float32`, shape `1 x 256 x 256 x 3`
- Output tensors: `1x195`, `1x1`, `1x256x256x1`, `1x64x64x39`, `1x117`
- Current preprocessing in [backend/pose_analyzer_onnx.py](/home/faith/sport-vision/backend/pose_analyzer_onnx.py) is RGB resize to `256x256`, then divide by `255.0`

## K230 toolchain requirements

- Python `3.8` to `3.10`
- Local or system `.NET 7` runtime for `hostfxr`
- Matching `nncase` and `nncase-kpu` package versions

K230 SDK to `nncase` examples from the official correspondence table:

- SDK `1.4.0` -> `nncase 2.8.0`
- SDK `1.5.0` -> `nncase 2.8.1`
- SDK `1.6.0` -> `nncase 2.8.3`

## Suggested environment setup

This conversion should run in a dedicated virtual environment instead of the project's main `.venv`.

The setup script first installs a local `.NET 7` runtime into `.dotnet`, then creates `.venv-k230`. If your system is missing `python3.8-venv`, it automatically falls back to `virtualenv`.

```bash
bash tools/setup_k230_env.sh
source tools/activate_k230_env.sh
source .venv-k230/bin/activate
```

Replace `2.8.3` with the version that matches your K230 SDK.

If you want to install a different `nncase` version during setup:

```bash
NNCASE_VERSION=2.8.1 bash tools/setup_k230_env.sh
```

If `python3.8` is not the correct interpreter on your machine:

```bash
PYTHON_BIN=/usr/bin/python3.9 bash tools/setup_k230_env.sh
```

## Calibration dataset

K230 requires PTQ for `k230` target compilation. Prepare a directory with representative RGB images from your real input domain. A practical minimum is `20` to `50` images.

You can generate these images directly from the demo video with [tools/extract_calibration_frames.py](/home/faith/sport-vision/tools/extract_calibration_frames.py):

```bash
.venv/bin/python tools/extract_calibration_frames.py \
  --video demo_videos/c52abfbb.mp4 \
  --output-dir calibration \
  --count 32
```

```bash
/home/faith/sport-vision/.venv/bin/python tools/extract_calibration_frames.py \
  --video demo_videos/c52abfbb.mp4 \
  --output-dir calibration \
  --count 48 \
  --start-seconds 2 \
  --end-seconds 18
```


Example layout:

```text
calibration/
  frame_001.jpg
  frame_002.jpg
  frame_003.jpg
```

## Convert the pose model

### Step 1: Re-export TFLite to pure float32

Run this with the main project environment because `tflite2tensorflow` is installed there:

```bash
.venv/bin/python tools/reexport_pose_tflite_float32.py \
  --model models/pose_landmark_full.tflite \
  --output-dir tmp/pose_landmark_full_float32
```

That produces:

```text
tmp/pose_landmark_full_float32/model_float32.tflite
```

### Step 2: Convert the clean float32 TFLite to K230

```bash
source tools/activate_k230_env.sh
.venv-k230/bin/python tools/convert_pose_tflite_to_k230.py \
  --model tmp/pose_landmark_full_float32/model_float32.tflite \
  --dataset calibration \
  --output models/pose_landmark_full.kmodel \
  --dump-dir tmp/k230_pose_landmark_full_tflite
```

### Step 3: Test the generated KModel

For board-side testing, the repository now includes [backend/pose_analyzer_kmodel.py](/home/faith/sport-vision/backend/pose_analyzer_kmodel.py) and [infer_image_pose_kmodel.py](/home/faith/sport-vision/infer_image_pose_kmodel.py). They keep the same pose post-processing as the ONNX version. On board they use `nncase_runtime`; on a Linux host with the nncase toolchain environment activated they can also fall back to `nncase.Simulator`.

Run a simple full-frame test:

```bash
python infer_image_pose_kmodel.py demo.jpg \
  --model models/pose_landmark_full.kmodel \
  --output demo_pose_kmodel.jpg
```

To force the host-side simulator on Linux:

```bash
source tools/activate_k230_env.sh
.venv-k230/bin/python infer_image_pose_kmodel.py demo.jpg \
  --model models/pose_landmark_full.kmodel \
  --simulator
```

If you already know the person ROI, you can pass it directly:

```bash
python infer_image_pose_kmodel.py demo.jpg \
  --model models/pose_landmark_full.kmodel \
  --roi 120,40,520,720
```

If the kmodel was compiled with `--keep-fp32-input`, add:

```bash
--keep-fp32-input
```

Notes:

- The KModel analyzer prefers `nncase_runtime` on board, and can fall back to `nncase.Simulator` on host after `source tools/activate_k230_env.sh`.
- By default it feeds `uint8` NHWC input, matching the current compile path in this repository.
- The KModel test script does not depend on the YOLO detector and can run directly on a full image or a provided ROI.
- Host simulation is useful for quick validation, but some K230 kmodels may still fail with `Operation not supported`; in that case, test on the real K230 runtime.

If you still want to try the ONNX path:

```bash
source tools/activate_k230_env.sh
.venv-k230/bin/python tools/convert_pose_onnx_to_k230.py \
  --model models/pose_landmark_full.onnx \
  --dataset calibration \
  --output models/pose_landmark_full.kmodel \
  --dump-dir tmp/k230_pose_landmark_full
```

The script compiles the model with these preprocessing assumptions:

- Input layout: `NHWC`
- Input type: `uint8`
- `input_range=[0,255]`
- `mean=[0,0,0]`
- `std=[255,255,255]`

That matches the repository's current ONNX preprocessing path, which converts RGB pixels to `[0,1]` float values.

## Common failures

- `Failed to initialize hostfxr`: run `bash tools/setup_k230_env.sh` and `source tools/activate_k230_env.sh`
- `The given key 'K230' was not present in the dictionary`: install `nncase-kpu` with the same version as `nncase`
- `Invalid kmodel` on board: the compiled `nncase` version does not match the board SDK runtime version
- ONNX importer aborts inside `CompilerImportOnnxModule`: switch to the TFLite conversion path above
- TFLite importer fails with `Quantize Parameter not found in tflite DeQuantize importer`: first run [tools/reexport_pose_tflite_float32.py](/home/faith/sport-vision/tools/reexport_pose_tflite_float32.py)