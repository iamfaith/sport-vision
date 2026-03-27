/home/faith/sport-vision/.venv/bin/python /home/faith/sport-vision/infer_video_fall_onnx.py aa.mp4 --multi-person --max-persons 5 --track-iou-threshold 0.22 --track-center-threshold 0.18 --track-max-missed 24 --output-json aa_fall_tuned_v2.json -o aa_fall_tuned_v2.mp4


--fall-preset recall  balance

/home/faith/sport-vision/.venv/bin/python infer_video_fall_onnx.py aa.mp4 --fall-preset recall --multi-person --max-persons 5 --output-json aa_fall_recall.json -o aa_fall_recall.mp4