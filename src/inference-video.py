#%% Header

"""
inference-video.py

Run DroneMegaDetector on a folder of videos (or a single video file),
producing output in the MegaDetector batch output format.

https://lila.science/megadetector-output-format
"""

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
import sys
import time
from datetime import datetime

import torch
import numpy as np
import supervision as sv
from PIL import Image

import rfdetr

from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS
from megadetector.detection.video_utils import find_videos, is_video_file
from megadetector.detection.video_utils import run_callback_on_frames_for_folder
from megadetector.detection.video_utils import _filename_to_frame_number

DEFAULT_CONFIDENCE_THRESHOLD = 0.005
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_TILE_OVERLAP = 100
DEFAULT_TILE_NMS_IOU = 0.5
DEFAULT_FPS = 1.0


#%% Support functions

def _parse_tile_overlap(s):
    error = argparse.ArgumentTypeError(
        f'Tile overlap must be an integer number of pixels, or a fraction greater than 0 '
        f'and less than 1, got {s}')
    try:
        overlap = int(s)
    except ValueError:
        try:
            overlap = float(s)
        except ValueError:
            raise error
        if not (0 < overlap < 1):
            raise error
    if overlap < 0:
        raise error
    return overlap


def _tile_overlap_pixels(tile_overlap, tile_size):
    if isinstance(tile_overlap, float):
        if not (0 < tile_overlap < 1):
            raise ValueError(f'A fractional tile overlap must be greater than 0 and less '
                             f'than 1, got {tile_overlap}')
        tile_overlap = round(tile_overlap * tile_size)

    if not (0 <= tile_overlap < tile_size):
        raise ValueError(f'Tile overlap must be at least 0 and less than the tile size '
                         f'({tile_size}), got {tile_overlap}')

    return tile_overlap


def convert_detections_to_md_format(detections, image_width, image_height):
    md_detections = []
    if (detections is None) or (len(detections) == 0):
        return md_detections

    for i_detection in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i_detection]

        x_min_norm = float(x1) / image_width
        y_min_norm = float(y1) / image_height
        width_norm = float(x2 - x1) / image_width
        height_norm = float(y2 - y1) / image_height

        x_min_norm = max(0.0, min(1.0, x_min_norm))
        y_min_norm = max(0.0, min(1.0, y_min_norm))
        width_norm = max(0.0, min(1.0 - x_min_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_min_norm, height_norm))

        conf = float(detections.confidence[i_detection])
        class_id = int(detections.class_id[i_detection])
        category = str(class_id)

        bbox = round_float_array([x_min_norm, y_min_norm, width_norm, height_norm],
                                 precision=COORD_DIGITS)
        conf = round_float(conf, precision=CONF_DIGITS)

        md_detections.append({
            'category': category,
            'conf': conf,
            'bbox': bbox
        })

    return md_detections


#%% Model loading

def load_model(detector_file, image_size=None, batch_size=1):
    """
    Load an RF-DETR model and automatically apply MPS/CUDA hardware optimizations[cite: 1].
    """
    if detector_file.lower().endswith('.ckpt'):
        raise ValueError(f"Cannot run inference directly from a .ckpt file: {detector_file}. Use a .pth file.")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    checkpoint = torch.load(detector_file, weights_only=False, map_location='cpu')
    has_model_config = ('model_config' in checkpoint)
    del checkpoint

    from_checkpoint_kwargs = {}
    if image_size is not None:
        from_checkpoint_kwargs['resolution'] = image_size
    elif not has_model_config:
        from_checkpoint_kwargs['resolution'] = DEFAULT_IMAGE_SIZE

    print(f'Loading model from {detector_file} on {device.upper()}...')
    model = rfdetr.from_checkpoint(detector_file, device=device, **from_checkpoint_kwargs)

    model_type = type(model).__name__
    resolved_size = model.model_config.resolution
    print(f'Loaded {model_type} at resolution {resolved_size}')

    optimized = False
    if device in ["cuda", "mps"]:
        compile_model = (batch_size == 1) and (device == "cuda")
        print(f'Optimizing model for {device.upper()} inference (float16, {"compiled" if compile_model else "not compiled"})')
        model.inference(compile=compile_model, batch_size=batch_size, dtype=torch.float16)
        optimized = True

    detection_categories = {str(i_class): class_name for i_class, class_name in enumerate(model.class_names)}

    return {
        'model': model,
        'model_type': model_type,
        'image_size': resolved_size,
        'detection_categories': detection_categories,
        'optimized_for_inference': optimized,
        'device': device
    }


#%% Inference on individual frames (tiled or non-tiled)

def _predict(model, images, threshold):
    if len(images) == 1:
        detections_list = [model.predict(images[0], threshold=threshold, include_source_image=False)]
    else:
        detections_list = model.predict(images, threshold=threshold, include_source_image=False)

    return [sv.Detections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id)
            for d in detections_list]


def _predict_tiled(model, image, threshold, tile_size, tile_overlap, batch_size=1):
    def tile_callback(tiles):
        if isinstance(tiles, list):
            return _predict(model, tiles, threshold)
        return _predict(model, [tiles], threshold)[0]

    slicer = sv.InferenceSlicer(
        callback=tile_callback,
        slice_wh=tile_size,
        overlap_wh=tile_overlap,
        iou_threshold=DEFAULT_TILE_NMS_IOU,
        batch_size=batch_size
    )
    return slicer(image)


def _detect(model, images, threshold, tiled=False, tile_size=None, tile_overlap=None, batch_size=1):
    if not tiled:
        return _predict(model, images, threshold)

    return [_predict_tiled(model, image, threshold, tile_size, tile_overlap, batch_size=batch_size)
            for image in images]


#%% Video inference

def _run_detector_on_videos(model, video_folder, video_files_relative, threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                            fps=DEFAULT_FPS, tiled=False, tile_size=None, tile_overlap=None, batch_size=1, verbose=False):
    time_sample = 1.0 / fps
    every_n_frames = -1 * time_sample
    start_time = time.time()

    def frame_callback(image_np, frame_id):
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        frame_image = Image.fromarray(image_np)
        img_width, img_height = frame_image.size

        try:
            detections = _detect(model, [frame_image], threshold, tiled=tiled,
                                 tile_size=tile_size, tile_overlap=tile_overlap,
                                 batch_size=batch_size)[0]
        except Exception as e:
            print(f'Error during inference on frame {frame_id}: {e}')
            return {'file': frame_id, 'detections': []}

        md_detections = convert_detections_to_md_format(detections, img_width, img_height)
        return {'file': frame_id, 'detections': md_detections}

    md_results = run_callback_on_frames_for_folder(
        input_video_folder=video_folder,
        frame_callback=frame_callback,
        every_n_frames=every_n_frames,
        verbose=verbose,
        files_to_process_relative=video_files_relative
    )

    video_results = md_results['results']
    video_filenames = md_results['video_filenames']
    video_frame_rates = md_results['frame_rates']
    results = []

    for i_video, results_this_video in enumerate(video_results):
        video_fn = video_filenames[i_video]
        im = {'file': video_fn, 'frame_rate': video_frame_rates[i_video], 'frames_processed': []}

        if isinstance(results_this_video, dict):
            im['failure'] = results_this_video['failure']
            im['detections'] = None
        else:
            im['detections'] = []
            for results_one_frame in results_this_video:
                frame_number = _filename_to_frame_number(results_one_frame['file'])
                im['frames_processed'].append(frame_number)
                for det in results_one_frame['detections']:
                    det['frame_number'] = frame_number
                im['detections'].extend(results_one_frame['detections'])

        im['frames_processed'] = sorted(im['frames_processed'])
        results.append(im)

    elapsed = time.time() - start_time
    print(f'Processed {len(video_results)} videos in {elapsed:.1f}s')

    return results


#%% Batch inference function

def run_detector_batch(
    detector_file,
    video_folder,
    output_file,
    image_size=None,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size=1,
    fps=DEFAULT_FPS,
    tiled=False,
    tile_overlap=None,
    verbose=False
):
    assert os.path.isfile(detector_file), f'Detector file not found: {detector_file}'
    assert os.path.exists(video_folder), f'Input file/folder not found: {video_folder}'
    assert output_file.endswith('.json'), 'Output file must have .json extension'

    if fps is None or fps <= 0:
        fps = DEFAULT_FPS

    if (tile_overlap is not None) and (not tiled):
        print('A tile overlap was specified, enabling tiled inference')
        tiled = True
    if tile_overlap is None:
        tile_overlap = DEFAULT_TILE_OVERLAP

    if os.path.isfile(video_folder):
        input_base_folder = os.path.dirname(video_folder) or "."
        if is_video_file(video_folder):
            video_files = [video_folder]
        else:
            print('Input file is not recognized as a video, exiting.')
            return None
    else:
        input_base_folder = video_folder
        print(f'Searching for videos in {video_folder}...')
        video_files = find_videos(video_folder, recursive=True, return_relative_paths=False)
        print(f'Found {len(video_files)} videos')

    if len(video_files) == 0:
        print('No videos found, exiting')
        return None

    model_info = load_model(
        detector_file,
        image_size=image_size,
        batch_size=batch_size
    )

    model = model_info['model']
    model_type = model_info['model_type']
    resolved_image_size = model_info['image_size']
    detection_categories = model_info['detection_categories']
    optimized_for_inference = model_info['optimized_for_inference']
    device = model_info['device']

    tile_size = resolved_image_size
    tile_overlap = _tile_overlap_pixels(tile_overlap, tile_size)
    if tiled:
        print(f'Using tiled inference with {tile_size}x{tile_size} tiles, overlapping by {tile_overlap} pixels')

    results = []
    video_files_relative = [os.path.relpath(fn, input_base_folder).replace('\\', '/') for fn in video_files]

    results.extend(_run_detector_on_videos(
        model=model, video_folder=input_base_folder, video_files_relative=video_files_relative,
        threshold=threshold, fps=fps, tiled=tiled, tile_size=tile_size,
        tile_overlap=tile_overlap, batch_size=batch_size, verbose=verbose))

    results = sort_list_of_dicts_by_key(results, 'file')

    detector_metadata = {
        'model_type': model_type,
        'image_size': resolved_image_size,
        'tiled': tiled,
        'optimized_for_inference': optimized_for_inference,
        'accelerator': device,
        'confidence_threshold': threshold,
        'fps': fps
    }
    if tiled:
        detector_metadata['tile_overlap'] = tile_overlap

    output = {
        'info': {
            'format_version': '1.5',
            'detector': os.path.basename(detector_file),
            'detection_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detector_metadata': detector_metadata
        },
        'detection_categories': detection_categories,
        'images': results
    }

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Results written to {output_file}')
    return output


#%% Command-line driver

def main():
    parser = argparse.ArgumentParser(description='Run DroneMegaDetector on videos[cite: 1].')
    parser.add_argument('detector_file', type=str, help='Path to RF-DETR checkpoint (.pth)[cite: 1]')
    parser.add_argument('folder', type=str, help='Path to folder of videos or single video file[cite: 1]')
    parser.add_argument('output_file', type=str, help='Path to output JSON file[cite: 1]')
    parser.add_argument('--image_size', '--resolution', dest='image_size', type=int, default=None,
                        help=f'Inference resolution override (default: {DEFAULT_IMAGE_SIZE})[cite: 1]')
    parser.add_argument('--threshold', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f'Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})[cite: 1]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for tiled inference (default: 1)[cite: 1]')
    parser.add_argument('--tiled', action='store_true', help='Enable tiled inference[cite: 1]')
    parser.add_argument('--tile_overlap', type=_parse_tile_overlap, default=None,
                        help=f'Tile overlap in pixels or fraction 0-1 (default: {DEFAULT_TILE_OVERLAP}); implies --tiled[cite: 1]')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help=f'Frames per second to sample (default: {DEFAULT_FPS})')
    parser.add_argument('--verbose', action='store_true', help='Enable debug output[cite: 1]')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    run_detector_batch(
        detector_file=args.detector_file,
        video_folder=args.folder,
        output_file=args.output_file,
        image_size=args.image_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        fps=args.fps,
        tiled=args.tiled,
        tile_overlap=args.tile_overlap,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()