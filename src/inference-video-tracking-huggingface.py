#%% Suppress Warnings (Must precede torch/transformers imports)

import os
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.disable(logging.WARNING)
for logger_name in ["transformers", "torch", "timm"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

#%% Imports and constants

import argparse
import json
import math
import sys
from datetime import datetime
import torch
import supervision as sv
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

DEFAULT_IMAGE_SIZE = 1280
DEFAULT_TILE_OVERLAP = 100
DEFAULT_TILE_NMS_IOU = 0.5


def _parse_tile_overlap(s):
    try:
        overlap = float(s) if '.' in s else int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid overlap: {s}")
    if (isinstance(overlap, float) and not (0 < overlap < 1)) or overlap < 0:
        raise argparse.ArgumentTypeError(f"Invalid overlap constraint: {s}")
    return overlap


def _tile_overlap_pixels(tile_overlap, tile_size):
    if isinstance(tile_overlap, float):
        tile_overlap = round(tile_overlap * tile_size)
    return tile_overlap


def convert_detections_to_md_format(detections, image_width, image_height, frame_number):
    md_detections = []
    if len(detections) == 0:
        return md_detections

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        x_min_norm = max(0.0, min(1.0, float(x1) / image_width))
        y_min_norm = max(0.0, min(1.0, float(y1) / image_height))
        width_norm = max(0.0, min(1.0 - x_min_norm, float(x2 - x1) / image_width))
        height_norm = max(0.0, min(1.0 - y_min_norm, float(y2 - y1) / image_height))

        conf = round(float(detections.confidence[i]), 3)
        bbox = [round(v, 4) for v in [x_min_norm, y_min_norm, width_norm, height_norm]]

        det = {
            'category': str(int(detections.class_id[i])),
            'conf': conf,
            'bbox': bbox,
            'frame_number': frame_number
        }

        if detections.tracker_id is not None and len(detections.tracker_id) > i:
            det['tracker_id'] = str(detections.tracker_id[i])
        md_detections.append(det)
    return md_detections


def run_tracking_inference(
    input_video_path,
    output_video_path="output_tracked.mp4",
    output_json_path="output_tracked.json",
    fps=None,
    threshold=0.3,
    image_size=None,
    tiled=False,
    tile_overlap=None,
    model_repo="ConservationDrones/DroneMegaDetector"
):
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Video file not found: {input_video_path}")

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Hugging Face model from '{model_repo}' on {device.upper()}...")
    processor = AutoImageProcessor.from_pretrained(model_repo)
    model = AutoModelForObjectDetection.from_pretrained(model_repo).to(device).eval()

    if image_size is not None:
        processor.size = {"height": image_size, "width": image_size}
        resolved_size = image_size
    else:
        resolved_size = processor.size.get("height", DEFAULT_IMAGE_SIZE) if isinstance(processor.size, dict) else DEFAULT_IMAGE_SIZE
    print(f"Configured inference resolution: {resolved_size}")

    if device in ["cuda", "mps"]:
        print(f"Optimizing model for {device.upper()} inference (float16)")
        model = model.half()

    if tile_overlap is not None and not tiled:
        print("A tile overlap was specified, enabling tiled inference")
        tiled = True
    tile_size = resolved_size
    tile_overlap = _tile_overlap_pixels(tile_overlap or DEFAULT_TILE_OVERLAP, tile_size)
    if tiled:
        print(f"Using tiled inference with {tile_size}x{tile_size} tiles, overlapping by {tile_overlap} pixels")

    def predict_single(bgr_or_pil_img):
        pil_img = Image.fromarray(bgr_or_pil_img[..., ::-1]) if not isinstance(bgr_or_pil_img, Image.Image) else bgr_or_pil_img
        w, h = pil_img.size
        encoded = processor(images=pil_img, return_tensors="pt").to(device)
        if device in ["cuda", "mps"]:
            encoded["pixel_values"] = encoded["pixel_values"].half()
        with torch.inference_mode():
            outputs = model(**encoded)
        res = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[(h, w)])[0]
        return sv.Detections(
            xyxy=res["boxes"].cpu().numpy(),
            confidence=res["scores"].cpu().numpy(),
            class_id=res["labels"].cpu().numpy()
        )

    slicer = sv.InferenceSlicer(
        callback=predict_single,
        slice_wh=tile_size,
        overlap_wh=tile_overlap,
        iou_threshold=DEFAULT_TILE_NMS_IOU
    ) if tiled else None

    video_info = sv.VideoInfo.from_video_path(video_path=input_video_path)
    native_fps = video_info.fps

    if fps is not None and fps > 0:
        stride = max(1, round(native_fps / fps))
        actual_fps = native_fps / stride
        print(f"FPS override provided: target={fps:.2f}. Sampling every {stride} frame(s) (~{actual_fps:.2f} FPS).")
    else:
        stride = 1
        actual_fps = native_fps
        print(f"Using video's native frame rate: {native_fps:.2f} FPS.")

    total_frames_to_process = math.ceil(video_info.total_frames / stride) if video_info.total_frames else None

    tracker = sv.ByteTrack(frame_rate=max(1, round(actual_fps)))
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    detection_categories = {str(k): v for k, v in model.config.id2label.items()}
    video_result = {'file': os.path.basename(input_video_path), 'frame_rate': actual_fps, 'frames_processed': [], 'detections': []}

    output_video_info = sv.VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=actual_fps,
        total_frames=total_frames_to_process or 0
    )
    frames_generator = sv.get_video_frames_generator(source_path=input_video_path, stride=stride)

    with sv.VideoSink(target_path=output_video_path, video_info=output_video_info) as sink:
        for i, frame in enumerate(tqdm(frames_generator, total=total_frames_to_process, desc="Processing frames", unit="frame")):
            frame_number = i * stride

            detections = slicer(frame) if tiled else predict_single(frame)
            detections = tracker.update_with_detections(detections)

            video_result['frames_processed'].append(frame_number)
            video_result['detections'].extend(convert_detections_to_md_format(detections, video_info.width, video_info.height, frame_number))

            labels = [
                f"#{t_id} {model.config.id2label[c_id]} {conf:.2f}"
                for c_id, conf, t_id in zip(detections.class_id, detections.confidence, detections.tracker_id)
            ]
            annotated = label_annotator.annotate(
                scene=box_annotator.annotate(scene=frame.copy(), detections=detections),
                detections=detections,
                labels=labels
            )
            sink.write_frame(frame=annotated)

    detector_metadata = {
        'image_size': resolved_size,
        'tiled': tiled,
        'native_fps': native_fps,
        'fps_requested': fps,
        'fps_processed': actual_fps,
        'confidence_threshold': threshold,
        'accelerator': device,
        'optimized': device in ["cuda", "mps"]
    }
    if tiled:
        detector_metadata['tile_overlap'] = tile_overlap

    output_json = {
        'info': {
            'format_version': '1.5',
            'detector': model_repo,
            'detection_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detector_metadata': detector_metadata
        },
        'detection_categories': detection_categories,
        'images': [video_result]
    }

    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Completed.\nVideo: {output_video_path}\nJSON: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="DroneMegaDetector HF with ByteTrack.")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS (defaults to native video FPS)")
    parser.add_argument("--image_size", "--resolution", dest="image_size", type=int, default=None,
                        help=f"Inference resolution override (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference")
    parser.add_argument("--tile_overlap", type=_parse_tile_overlap, default=None,
                        help=f"Tile overlap in pixels or fraction 0-1 (default: {DEFAULT_TILE_OVERLAP}); implies --tiled")
    parser.add_argument("--output_video", type=str, default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--output_json", type=str, default="output_tracked.json", help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    args = parser.parse_args()

    run_tracking_inference(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        output_json_path=args.output_json,
        fps=args.fps,
        threshold=args.threshold,
        image_size=args.image_size,
        tiled=args.tiled,
        tile_overlap=args.tile_overlap
    )

if __name__ == "__main__":
    main()