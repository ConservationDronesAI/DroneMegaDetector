#%% Suppress Warnings (Must precede torch/rfdetr imports)

import os
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.disable(logging.WARNING)
for logger_name in ["rf-detr", "rfdetr", "timm", "torch"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

#%% Imports and constants

import argparse
import json
import math
import sys
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
import supervision as sv
import rfdetr
from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS

DEFAULT_CONFIDENCE_THRESHOLD = 0.005
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
    if (detections is None) or (len(detections) == 0):
        return md_detections

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        x_min_norm = max(0.0, min(1.0, float(x1) / image_width))
        y_min_norm = max(0.0, min(1.0, float(y1) / image_height))
        width_norm = max(0.0, min(1.0 - x_min_norm, float(x2 - x1) / image_width))
        height_norm = max(0.0, min(1.0 - y_min_norm, float(y2 - y1) / image_height))

        conf = round_float(float(detections.confidence[i]), precision=CONF_DIGITS)
        bbox = round_float_array([x_min_norm, y_min_norm, width_norm, height_norm], precision=COORD_DIGITS)

        det = {'category': str(int(detections.class_id[i])), 'conf': conf, 'bbox': bbox, 'frame_number': frame_number}
        if detections.tracker_id is not None and len(detections.tracker_id) > i:
            det['tracker_id'] = str(detections.tracker_id[i])
        md_detections.append(det)
    return md_detections


def load_model(detector_file, image_size=None, batch_size=1):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    checkpoint = torch.load(detector_file, weights_only=False, map_location='cpu')
    from_checkpoint_kwargs = {'resolution': image_size} if image_size else {} if 'model_config' in checkpoint else {'resolution': DEFAULT_IMAGE_SIZE}
    del checkpoint

    print(f'Loading model from {detector_file} on {device.upper()}...')
    model = rfdetr.from_checkpoint(detector_file, device=device, **from_checkpoint_kwargs)
    resolved_size = model.model_config.resolution
    print(f'Loaded {type(model).__name__} at resolution {resolved_size}')

    optimized = False
    if device in ["cuda", "mps"]:
        compile_model = (batch_size == 1) and (device == "cuda")
        print(f'Optimizing model for {device.upper()} inference (float16, {"compiled" if compile_model else "not compiled"})')
        model.inference(compile=compile_model, batch_size=batch_size, dtype=torch.float16)
        optimized = True

    return {
        'model': model,
        'model_type': type(model).__name__,
        'image_size': resolved_size,
        'detection_categories': {str(i): name for i, name in enumerate(model.class_names)},
        'optimized_for_inference': optimized,
        'device': device
    }


def _detect(model, images, threshold, tiled=False, tile_size=None, tile_overlap=None, batch_size=1):
    if not tiled:
        res = model.predict(images[0] if len(images) == 1 else images, threshold=threshold, include_source_image=False)
        return [sv.Detections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id) for d in (res if len(images) > 1 else [res])]

    def tile_cb(tiles):
        res = model.predict(tiles, threshold=threshold, include_source_image=False)
        return [sv.Detections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id) for d in res] if isinstance(tiles, list) else sv.Detections(xyxy=res.xyxy, confidence=res.confidence, class_id=res.class_id)

    slicer = sv.InferenceSlicer(callback=tile_cb, slice_wh=tile_size, overlap_wh=tile_overlap, iou_threshold=DEFAULT_TILE_NMS_IOU, batch_size=batch_size)
    return [slicer(img) for img in images]


def run_tracking(detector_file, video_file, output_video, output_json, fps=None, threshold=DEFAULT_CONFIDENCE_THRESHOLD, image_size=None, tiled=False, tile_overlap=None, batch_size=1):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(video_path=video_file)
    native_fps = video_info.fps

    stride = max(1, round(native_fps / fps)) if fps and fps > 0 else 1
    actual_fps = native_fps / stride
    total_frames_to_process = math.ceil(video_info.total_frames / stride) if video_info.total_frames else None
    print(f"Sampling {video_file} at ~{actual_fps:.2f} FPS (Stride: {stride}).")

    model_info = load_model(detector_file, image_size=image_size, batch_size=batch_size)
    model, tile_size = model_info['model'], model_info['image_size']
    if tile_overlap is not None and not tiled:
        print("A tile overlap was specified, enabling tiled inference")
        tiled = True
    tile_overlap = _tile_overlap_pixels(tile_overlap or DEFAULT_TILE_OVERLAP, tile_size)
    if tiled:
        print(f"Using tiled inference with {tile_size}x{tile_size} tiles, overlapping by {tile_overlap} pixels")

    tracker = sv.ByteTrack(frame_rate=max(1, round(actual_fps)))
    box_ann = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator()

    video_result = {'file': os.path.basename(video_file), 'frame_rate': actual_fps, 'frames_processed': [], 'detections': []}
    out_info = sv.VideoInfo(width=video_info.width, height=video_info.height, fps=actual_fps, total_frames=total_frames_to_process or 0)
    frames_generator = sv.get_video_frames_generator(source_path=video_file, stride=stride)

    start_time = time.time()
    with sv.VideoSink(target_path=output_video, video_info=out_info) as sink:
        for i, frame in enumerate(tqdm(frames_generator, total=total_frames_to_process, desc="Processing frames", unit="frame")):
            frame_num = i * stride
            detections = _detect(model, [Image.fromarray(frame[..., ::-1])], threshold, tiled, tile_size, tile_overlap, batch_size)[0]
            detections = tracker.update_with_detections(detections)

            video_result['frames_processed'].append(frame_num)
            video_result['detections'].extend(convert_detections_to_md_format(detections, video_info.width, video_info.height, frame_num))

            labels = [f"#{t_id or ''} {model_info['detection_categories'].get(str(c_id))} {conf:.2f}" for c_id, conf, t_id in zip(detections.class_id, detections.confidence, detections.tracker_id)]
            sink.write_frame(label_ann.annotate(scene=box_ann.annotate(scene=frame.copy(), detections=detections), detections=detections, labels=labels))

    print(f"Tracking complete in {time.time() - start_time:.1f}s.")

    output = {
        'info': {
            'format_version': '1.5',
            'detector': os.path.basename(detector_file),
            'detection_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detector_metadata': {
                'model_type': model_info['model_type'],
                'image_size': tile_size,
                'tiled': tiled,
                'optimized_for_inference': model_info['optimized_for_inference'],
                'accelerator': model_info['device'],
                'confidence_threshold': threshold,
                'native_fps': native_fps,
                'fps_requested': fps,
                'fps_processed': actual_fps
            }
        },
        'detection_categories': model_info['detection_categories'],
        'images': [video_result]
    }
    if tiled:
        output['info']['detector_metadata']['tile_overlap'] = tile_overlap
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Completed.\nVideo: {output_video}\nJSON: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="DroneMegaDetector (.pth) Tracking.")
    parser.add_argument("detector_file", type=str, help="Path to RF-DETR checkpoint (.pth)")
    parser.add_argument("video_file", type=str, help="Path to input video file")
    parser.add_argument("--output_video", type=str, default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--output_json", type=str, default="output_tracked.json", help="Output JSON path")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS (defaults to native video FPS)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--image_size", "--resolution", dest="image_size", type=int, default=None,
                        help=f"Inference resolution override (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference")
    parser.add_argument("--tile_overlap", type=_parse_tile_overlap, default=None,
                        help=f"Tile overlap in pixels or fraction 0-1 (default: {DEFAULT_TILE_OVERLAP}); implies --tiled")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for tiled inference")
    args = parser.parse_args()

    run_tracking(
        args.detector_file,
        args.video_file,
        args.output_video,
        args.output_json,
        args.fps,
        args.threshold,
        args.image_size,
        args.tiled,
        args.tile_overlap,
        args.batch_size
    )

if __name__ == "__main__":
    main()