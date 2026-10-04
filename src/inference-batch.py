#%% Header

"""
inference-batch.py

Run DroneMegaDetector on a folder of images and/or videos (or a single image
or video file), producing output in the MegaDetector batch output format.

https://lila.science/megadetector-output-format

Category IDs in the output are the class IDs produced by the model
(0 = person, 1 = animal, 2 = vehicle); the 'detection_categories' field in the
output maps IDs to names.

Optionally runs tiled inference: each image is divided into overlapping square
tiles at the model's inference resolution, the detector runs on each tile, and
detections are merged across tiles with non-maximum suppression.

Videos are handled by sampling frames (by default one frame per second) and
running the detector on each sampled frame; per-frame detections are collected
into a single per-video entry in the output file, following the video
conventions described in the format spec.
"""

#%% Imports and constants

import argparse
import json
import os
import sys
import time
import torch

import numpy as np
import supervision as sv

from datetime import datetime
from PIL import Image, ImageOps
from tqdm import tqdm
import queue
import math

from multiprocessing import Process
from threading import Thread

# RF-DETR: models are loaded via rfdetr.from_checkpoint(), which resolves the
# variant architecture and training resolution from metadata in the checkpoint.
import rfdetr

from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS
from megadetector.utils.path_utils import find_images

# Video support, all leveraged from the MegaDetector package
from megadetector.detection.video_utils import find_videos, is_video_file
from megadetector.detection.video_utils import run_callback_on_frames_for_folder
from megadetector.detection.video_utils import _filename_to_frame_number

# By default, exclude detections below this confidence level
DEFAULT_CONFIDENCE_THRESHOLD = 0.005

# Inference resolution to use when the model file doesn't record its training
# resolution (DroneMegaDetector model files released before 2026.10.04)
DEFAULT_IMAGE_SIZE = 1280

# For tiled inference, the default overlap between adjacent tiles (in pixels), and
# the IoU threshold for non-maximum suppression when merging detections across tiles
DEFAULT_TILE_OVERLAP = 100
DEFAULT_TILE_NMS_IOU = 0.5

# Default maximum number of images to buffer ahead of inference
DEFAULT_MAX_QUEUE_SIZE = 20

# Default number of parallel image loading threads
DEFAULT_LOADER_WORKERS = 4

# When sampling frames from video and neither a frame interval nor a time
# interval is specified, sample frames at this rate (in seconds).  For a typical
# 30 fps video, this samples roughly every 30th frame.
DEFAULT_SECONDS_PER_VIDEO_FRAME = 1.0


#%% Support functions

def load_image(image_path):
    """
    Load an image from disk, applying any EXIF orientation.

    Args:
        image_path (str): path to image file

    Returns:
        PIL.Image or None: loaded image, or None if loading failed
    """

    try:
        img = Image.open(image_path)
        # Rotate according to the EXIF orientation tag, if present; a corrupt EXIF
        # block shouldn't prevent us from processing the image.
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            print(f'Warning: could not apply EXIF orientation for {image_path}: {e}')
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        return None

# ...def load_image(...)


def _producer_func(q, image_paths, image_folder):
    """
    Producer function for the image loading queue.

    Loads images from disk and puts (relative_path, PIL.Image) tuples onto a
    bounded queue.  Sends None when all images have been loaded.

    Args:
        q (queue.Queue): bounded queue shared with the consumer
        image_paths (list): absolute image file paths to load
        image_folder (str): root folder used to compute relative paths
    """

    for path in image_paths:
        rel_path = os.path.relpath(path, image_folder).replace('\\', '/')
        img = load_image(path)
        q.put((rel_path, img))

    # Signal that this producer is finished
    q.put(None)


def _parse_tile_overlap(s):
    """
    Parse a tile overlap from the command line.

    Args:
        s (str): an integer number of pixels, or a float greater than 0 and less
            than 1, representing a fraction of the tile size

    Returns:
        int or float: the parsed overlap
    """

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

# ...def _parse_tile_overlap(...)


def _tile_overlap_pixels(tile_overlap, tile_size):
    """
    Convert a tile overlap to pixels.

    Args:
        tile_overlap (int or float): an integer number of pixels, or a float greater
            than 0 and less than 1, representing a fraction of [tile_size]
        tile_size (int): tile width and height in pixels

    Returns:
        int: overlap in pixels
    """

    if isinstance(tile_overlap, float):
        if not (0 < tile_overlap < 1):
            raise ValueError(f'A fractional tile overlap must be greater than 0 and less '
                             f'than 1, got {tile_overlap}')
        tile_overlap = round(tile_overlap * tile_size)

    if not (0 <= tile_overlap < tile_size):
        raise ValueError(f'Tile overlap must be at least 0 and less than the tile size '
                         f'({tile_size}), got {tile_overlap}')

    return tile_overlap

# ...def _tile_overlap_pixels(...)


def convert_detections_to_md_format(detections, image_width, image_height):
    """
    Convert RF-DETR/Supervision detections to MegaDetector format.

    Args:
        detections: supervision Detections object with xyxy, confidence, class_id
        image_width (int): image width in pixels
        image_height (int): image height in pixels

    Returns:
        list: list of detection dicts in MegaDetector format
    """

    md_detections = []

    if (detections is None) or (len(detections) == 0):
        return md_detections

    for i_detection in range(len(detections)):

        # Extract xyxy coordinates (absolute pixels)
        x1, y1, x2, y2 = detections.xyxy[i_detection]

        # Convert to normalized xywh format
        x_min_norm = float(x1) / image_width
        y_min_norm = float(y1) / image_height
        width_norm = float(x2 - x1) / image_width
        height_norm = float(y2 - y1) / image_height

        # Clamp values to [0, 1] range
        x_min_norm = max(0.0, min(1.0, x_min_norm))
        y_min_norm = max(0.0, min(1.0, y_min_norm))
        width_norm = max(0.0, min(1.0 - x_min_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_min_norm, height_norm))

        # Get confidence and class_id
        conf = float(detections.confidence[i_detection])

        # RF-DETR class_ids are 0-indexed when returned from the API
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

    # ...for each detection

    return md_detections

# ...def convert_detections_to_md_format(...)


#%% Model loading

def load_model(detector_file,
               image_size=None,
               optimize_for_inference=False,
               batch_size=1):
    """
    Load an RF-DETR model from an inference-ready .pth checkpoint via
    rfdetr.from_checkpoint(), which resolves the variant architecture, training
    resolution, and class names from metadata stored in the checkpoint.

    Args:
        detector_file (str): path to .pth checkpoint file.  If the checkpoint doesn't
            contain a top-level 'model_config' (which records the training resolution),
            the model runs at DEFAULT_IMAGE_SIZE.
        image_size (int, optional): image resolution for inference.  None uses the
            training resolution recorded in the checkpoint (or DEFAULT_IMAGE_SIZE); a
            value overrides it.
        optimize_for_inference (bool, optional): whether to optimize the model for
            inference, which should be a free lunch, but as of 9/2025 there is some
            risk of accuracy regression.
        batch_size (int, optional): batch size to pass to model.inference()

    Returns:
        dict: dictionary with keys:
            - 'model': the loaded RF-DETR model
            - 'model_type' (str): resolved variant class name (e.g. 'RFDETRMedium')
            - 'image_size' (int): resolved inference resolution
            - 'detection_categories' (dict): mapping from string category IDs to class names
    """

    if detector_file.lower().endswith('.ckpt'):
        raise ValueError(
            f"Cannot run inference directly from a .ckpt file: {detector_file}\n"
            f"Use an inference-ready .pth file."
        )

    # rfdetr.from_checkpoint() reads the training resolution from the 'model_config'
    # field, which was not present in DroneMegaDetector model files released before
    # 2026.10.04.  Without it, from_checkpoint() would use the default resolution for
    # the model architecture, which is not what DroneMegaDetector was trained at.
    print(f'Reading checkpoint metadata from: {detector_file}')
    checkpoint = torch.load(detector_file, weights_only=False, map_location='cpu')
    has_model_config = ('model_config' in checkpoint)
    del checkpoint

    # Load the model, letting from_checkpoint() resolve the model type and resolution.
    #
    # A caller-supplied image_size overrides the loaded resolution.
    from_checkpoint_kwargs = {}
    if image_size is not None:
        from_checkpoint_kwargs['resolution'] = image_size
    elif not has_model_config:
        print(f'Model file has no model_config metadata, using resolution {DEFAULT_IMAGE_SIZE}')
        from_checkpoint_kwargs['resolution'] = DEFAULT_IMAGE_SIZE
    print(f'Loading model from {detector_file}...')
    model = rfdetr.from_checkpoint(detector_file, **from_checkpoint_kwargs)

    model_type = type(model).__name__
    image_size = model.model_config.resolution
    print(f'Loaded {model_type} at resolution {image_size}')

    if optimize_for_inference:
        model.inference(batch_size=batch_size)

    # Get class names from model
    #
    # model.class_names is a list of strings.  Note to self: in older rfdetr versions, it was
    # a dict mapping 1-indexed class IDs to names.
    class_names = model.class_names
    print(f'Class names: {class_names}')

    # Build detection_categories dict
    detection_categories = {}
    for i_class,class_name in enumerate(class_names):
        detection_categories[str(i_class)] = class_name

    return \
    {
        'model': model,
        'model_type': model_type,
        'image_size': image_size,
        'detection_categories': detection_categories
    }

# ...def load_model(...)


#%% Inference on individual images (tiled or non-tiled)

def _predict(model, images, threshold, pad_to=None):
    """
    Run [model] on a list of images, without tiling.

    Args:
        model: a loaded RF-DETR model (from load_model)
        images (list): PIL images (or tiles)
        threshold (float): confidence threshold for detections
        pad_to (int, optional): if [images] has fewer than this many images, pad it
            with copies of the last image (and discard the results for the copies).
            A model compiled by model.inference() only accepts batches of the size it
            was compiled for.

    Returns:
        list: one supervision Detections object per image, containing only
        boxes, confidence values, and class IDs
    """

    n_images = len(images)
    if (pad_to is not None) and (n_images < pad_to):
        images = images + [images[-1]] * (pad_to - n_images)

    if len(images) == 1:
        detections_list = [model.predict(images[0], threshold=threshold,
                                         include_source_image=False)]
    else:
        detections_list = model.predict(images, threshold=threshold,
                                        include_source_image=False)

    detections_list = detections_list[:n_images]

    # Keep just the core arrays, so detections from different tiles can be merged
    return [sv.Detections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id)
            for d in detections_list]

# ...def _predict(...)


def _predict_tiled(model, image, threshold, tile_size, tile_overlap, batch_size=1,
                   pad_batches=False):
    """
    Run [model] on [image] in overlapping square tiles, merging detections across
    tiles with per-class non-maximum suppression.

    Tiles are [tile_size] pixels on a side, overlapping by [tile_overlap] pixels; the
    last row and column of tiles are aligned to the image edges, so every tile is
    full-size unless the image is smaller than a tile.

    Args:
        model: a loaded RF-DETR model (from load_model)
        image (PIL.Image): image to process
        threshold (float): confidence threshold for detections
        tile_size (int): tile width and height in pixels
        tile_overlap (int): overlap between adjacent tiles in pixels
        batch_size (int, optional): number of tiles to run through the model at once
        pad_batches (bool, optional): pad every batch of tiles to [batch_size] (see
            _predict)

    Returns:
        supervision Detections: detections in full-image pixel coordinates
    """

    pad_to = batch_size if pad_batches else None

    def tile_callback(tiles):
        # InferenceSlicer passes a single tile when batch_size is 1, otherwise a list
        if isinstance(tiles, list):
            return _predict(model, tiles, threshold, pad_to=pad_to)
        return _predict(model, [tiles], threshold, pad_to=pad_to)[0]

    slicer = sv.InferenceSlicer(
        callback=tile_callback,
        slice_wh=tile_size,
        overlap_wh=tile_overlap,
        iou_threshold=DEFAULT_TILE_NMS_IOU,
        batch_size=batch_size
    )

    return slicer(image)

# ...def _predict_tiled(...)


def _detect(model, images, threshold, tiled=False, tile_size=None, tile_overlap=None,
            batch_size=1, pad_batches=False):
    """
    Run [model] on a list of images, with or without tiling.

    Args:
        model: a loaded RF-DETR model (from load_model)
        images (list): PIL images
        threshold (float): confidence threshold for detections
        tiled (bool, optional): whether to use tiled inference
        tile_size (int, optional): tile size in pixels for tiled inference
        tile_overlap (int, optional): tile overlap in pixels for tiled inference
        batch_size (int, optional): the batch size the caller is using; for tiled
            inference, the number of tiles to run through the model at once
        pad_batches (bool, optional): pad every batch of images (or tiles) to
            [batch_size] (see _predict)

    Returns:
        list: one supervision Detections object per image
    """

    if not tiled:
        return _predict(model, images, threshold,
                        pad_to=batch_size if pad_batches else None)

    return [_predict_tiled(model, image, threshold, tile_size, tile_overlap,
                           batch_size=batch_size, pad_batches=pad_batches)
            for image in images]

# ...def _detect(...)


#%% Image inference

def _run_detector_on_images(model,
                            image_files,
                            image_folder,
                            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                            batch_size=1,
                            loader_workers=DEFAULT_LOADER_WORKERS,
                            worker_type='thread',
                            include_image_size=False,
                            tiled=False,
                            tile_size=None,
                            tile_overlap=None,
                            pad_batches=False):
    """
    Run [model] on a list of image files, returning per-image results in
    MegaDetector format.

    Images are processed using a producer/consumer pattern: loader threads (or
    processes) populate a bounded queue, and the calling thread pulls from that
    queue for inference.

    Args:
        model: a loaded RF-DETR model (from load_model)
        image_files (list): absolute paths to the images to process
        image_folder (str): base folder used to compute relative output paths
        threshold (float, optional): confidence threshold for detections
        batch_size (int, optional): batch size for inference (images per batch, or
            tiles per batch for tiled inference)
        loader_workers (int, optional): number of parallel image loaders
        worker_type (str, optional): 'thread' or 'process' for image loading workers
        include_image_size (bool, optional): whether to include image dimensions in output
        tiled (bool, optional): whether to use tiled inference
        tile_size (int, optional): tile size in pixels for tiled inference
        tile_overlap (int, optional): tile overlap in pixels for tiled inference
        pad_batches (bool, optional): pad every batch of images (or tiles) to
            [batch_size], required when the model has been compiled for a fixed batch size

    Returns:
        list: per-image result dicts in MegaDetector format
    """

    results = []
    start_time = time.time()

    max_queue_size = max(DEFAULT_MAX_QUEUE_SIZE, 4 * batch_size)

    # Split image list across loader workers and start producer workers
    if worker_type == 'thread':
        image_queue = queue.Queue(maxsize=max_queue_size)
    else:
        import multiprocessing
        image_queue = multiprocessing.Queue(maxsize=max_queue_size)

    chunks = []
    chunk_size = math.ceil(len(image_files) / loader_workers)
    for i in range(loader_workers):
        chunk = image_files[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)

    worker_class = Thread if worker_type == 'thread' else Process

    producers = []
    for chunk in chunks:
        t = worker_class(target=_producer_func, args=(image_queue, chunk, image_folder))
        t.daemon = True
        t.start()
        producers.append(t)

    # Consumer: pull loaded images from the queue and run inference in batches
    n_producers_finished = 0
    n_total_producers = len(producers)

    pbar = tqdm(total=len(image_files), desc='Processing images')

    while (n_producers_finished < n_total_producers):

        # Collect a batch of images from the queue
        valid_items = []  # list of (rel_path, img) for images that loaded successfully
        n_collected = 0

        while (n_collected < batch_size) and (n_producers_finished < n_total_producers):

            item = image_queue.get()

            # None is the sentinel indicating a producer thread has finished
            if item is None:
                n_producers_finished += 1
                continue

            rel_path, img = item
            n_collected += 1

            if img is None:
                results.append({
                    'file': rel_path,
                    'failure': 'Image could not be loaded'
                })
            else:
                valid_items.append((rel_path, img))

        # ...while collecting a batch

        pbar.update(n_collected)

        if len(valid_items) == 0:
            continue

        # Run inference
        images_for_inference = [item[1] for item in valid_items]

        try:
            detections_list = _detect(model, images_for_inference, threshold,
                                      tiled=tiled, tile_size=tile_size,
                                      tile_overlap=tile_overlap,
                                      batch_size=batch_size,
                                      pad_batches=pad_batches)
        except Exception as e:
            # If batch inference fails, mark all images in batch as failed
            print(f'Error during inference: {e}')
            for rel_path, _ in valid_items:
                results.append({
                    'file': rel_path,
                    'failure': f'Inference error: {str(e)}'
                })
            continue

        # Convert detections to MegaDetector format
        for (rel_path, img), detections in zip(valid_items, detections_list):
            img_width, img_height = img.size

            md_detections = convert_detections_to_md_format(
                detections, img_width, img_height
            )

            result = {
                'file': rel_path,
                'detections': md_detections
            }

            if include_image_size:
                result['width'] = img_width
                result['height'] = img_height

            results.append(result)

        # ...for each image in the batch

    # ...while producers are still running

    pbar.close()

    for t in producers:
        t.join()

    elapsed = time.time() - start_time
    images_per_second = len(image_files) / elapsed if elapsed > 0 else 0
    print(f'Processed {len(image_files)} images in {elapsed:.1f}s ({images_per_second:.2f} images/sec)')

    return results

# ...def _run_detector_on_images(...)


#%% Video inference

def _run_detector_on_videos(model,
                            video_folder,
                            video_files_relative,
                            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                            frame_sample=None,
                            time_sample=None,
                            tiled=False,
                            tile_size=None,
                            tile_overlap=None,
                            batch_size=1,
                            pad_batches=False,
                            verbose=False):
    """
    Run [model] on a list of videos, returning one per-video result dict in
    MegaDetector format for each video.

    Frame extraction and sampling are handled by the MegaDetector package; we
    supply a per-frame callback that runs the detector on each sampled frame.
    Per-frame detections are collected into a single per-video entry, with a
    'frame_number' added to each detection and a sorted 'frames_processed' list,
    following the video conventions in the MegaDetector output format.

    Args:
        model: a loaded RF-DETR model (from load_model)
        video_folder (str): base folder used to compute relative output paths and to
            resolve [video_files_relative]
        video_files_relative (list): video paths relative to [video_folder]
        threshold (float, optional): confidence threshold for detections
        frame_sample (int, optional): process every Nth frame; mutually exclusive
            with time_sample
        time_sample (float, optional): process frames every N seconds; mutually
            exclusive with frame_sample
        tiled (bool, optional): whether to use tiled inference on each frame
        tile_size (int, optional): tile size in pixels for tiled inference
        tile_overlap (int, optional): tile overlap in pixels for tiled inference
        batch_size (int, optional): for tiled inference, number of tiles to run
            through the model at once (frames are always processed one at a time)
        pad_batches (bool, optional): pad every batch of frames (or tiles) to
            [batch_size], required when the model has been compiled for a fixed batch size
        verbose (bool, optional): enable additional debug output

    Returns:
        list: per-video result dicts in MegaDetector format
    """

    assert not ((frame_sample is not None) and (time_sample is not None)), \
        'frame_sample and time_sample are mutually exclusive'

    # The MegaDetector frame helpers use a single "every_n_frames" parameter, where
    # a negative value is interpreted as a sampling interval in seconds.
    if time_sample is not None:
        every_n_frames = -1 * time_sample
    else:
        every_n_frames = frame_sample

    start_time = time.time()

    def frame_callback(image_np, frame_id):
        """
        Run the detector on a single video frame.

        Args:
            image_np (numpy.ndarray): frame data in PIL orientation/channel order (RGB)
            frame_id (str): synthetic frame filename, e.g. "frame000030.jpg"

        Returns:
            dict: {'file': frame_id, 'detections': [...]} in MegaDetector format
        """

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        frame_image = Image.fromarray(image_np)
        img_width, img_height = frame_image.size

        try:
            detections = _detect(model, [frame_image], threshold, tiled=tiled,
                                 tile_size=tile_size, tile_overlap=tile_overlap,
                                 batch_size=batch_size, pad_batches=pad_batches)[0]
        except Exception as e:
            print(f'Error during inference on frame {frame_id}: {e}')
            return {'file': frame_id, 'detections': []}

        md_detections = convert_detections_to_md_format(
            detections, img_width, img_height)

        return {'file': frame_id, 'detections': md_detections}

    # ...def frame_callback(...)

    # [md_results] is a dict with keys 'video_filenames' (list of relative str),
    # 'frame_rates' (list of float), and 'results' (list, one element per video, of
    # lists of per-frame callback return values).  For failed videos, the frame rate
    # is -1 and 'results' is a dict with at least the key 'failure'.
    md_results = run_callback_on_frames_for_folder(
        input_video_folder=video_folder,
        frame_callback=frame_callback,
        every_n_frames=every_n_frames,
        verbose=verbose,
        files_to_process_relative=video_files_relative,
        error_on_empty_video=False)

    video_results = md_results['results']
    video_filenames = md_results['video_filenames']
    video_frame_rates = md_results['frame_rates']

    assert len(video_results) == len(video_filenames)
    assert len(video_results) == len(video_frame_rates)

    results = []

    # i_video = 0; results_this_video = video_results[i_video]
    for i_video, results_this_video in enumerate(video_results):

        video_fn = video_filenames[i_video]

        im = {}
        im['file'] = video_fn
        im['frame_rate'] = video_frame_rates[i_video]
        im['frames_processed'] = []

        if isinstance(results_this_video, dict):

            # This was a failed video
            assert 'failure' in results_this_video
            im['failure'] = results_this_video['failure']
            im['detections'] = None

        else:

            im['detections'] = []

            # results_one_frame = results_this_video[0]
            for results_one_frame in results_this_video:

                assert results_one_frame['file'].startswith(video_fn)

                frame_number = _filename_to_frame_number(results_one_frame['file'])

                assert frame_number not in im['frames_processed'], \
                    'Received the same frame twice for video {}'.format(im['file'])

                im['frames_processed'].append(frame_number)

                for det in results_one_frame['detections']:
                    det['frame_number'] = frame_number

                # This is a no-op if there were no above-threshold detections
                # in this frame
                im['detections'].extend(results_one_frame['detections'])

            # ...for each frame

        # ...was this a failed video?

        im['frames_processed'] = sorted(im['frames_processed'])

        results.append(im)

    # ...for each video

    elapsed = time.time() - start_time
    print(f'Processed {len(video_results)} videos in {elapsed:.1f}s')

    return results

# ...def _run_detector_on_videos(...)


#%% Batch inference function

def run_detector_batch(
    detector_file,
    image_folder,
    output_file,
    image_size=None,
    loader_workers=DEFAULT_LOADER_WORKERS,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size=1,
    include_image_size=False,
    optimize_for_inference=False,
    worker_type='thread',
    skip_images=False,
    skip_video=False,
    frame_sample=None,
    time_sample=None,
    tiled=False,
    tile_overlap=None,
    verbose=False
):
    """
    Run DroneMegaDetector on the images and/or videos in a folder, or on a single
    image or video file.

    Args:
        detector_file (str): path to .pth checkpoint file
        image_folder (str): path to a folder (searched recursively for images and/or
            videos) or to a single image or video file.  Despite the name, this may
            contain videos and/or be a single file.
        output_file (str): path to output .json file
        image_size (int, optional): image resolution for inference, None to use the
            training resolution recorded in the model file (or DEFAULT_IMAGE_SIZE if the
            model file doesn't record it)
        loader_workers (int, optional): number of parallel image loaders
        threshold (float, optional): confidence threshold for detections
        batch_size (int, optional): batch size for inference (images per batch, or
            tiles per batch for tiled inference; video frames are processed one at a time)
        include_image_size (bool, optional): whether to include image dimensions in output
            (images only)
        optimize_for_inference (bool, optional): whether to optimize the model for inference,
            which should be a free lunch, but as of 9/2025 there is some risk of accuracy
            regression.  The optimized model is compiled for [batch_size], so smaller
            batches (including individual video frames) are padded to [batch_size].
        worker_type (str, optional): 'thread' or 'process' for image loading workers
            (default: 'thread')
        skip_images (bool, optional): ignore images, only process videos
        skip_video (bool, optional): ignore videos, only process images
        frame_sample (int, optional): sample every Nth frame from videos; mutually
            exclusive with time_sample
        time_sample (float, optional): sample frames every N seconds from videos;
            mutually exclusive with frame_sample.  If neither frame_sample nor
            time_sample is specified, defaults to DEFAULT_SECONDS_PER_VIDEO_FRAME.
        tiled (bool, optional): run tiled inference, using tiles the size of the
            inference resolution
        tile_overlap (int or float, optional): overlap between adjacent tiles, as an
            integer number of pixels, or as a float greater than 0 and less than 1,
            representing a fraction of the tile size.  None uses DEFAULT_TILE_OVERLAP.
            Specifying a tile overlap enables tiled inference.
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: Results dictionary in MegaDetector format
    """

    # Validate and normalize inputs
    assert os.path.isfile(detector_file), f'Detector file not found: {detector_file}'
    assert os.path.exists(image_folder), f'Input file/folder not found: {image_folder}'
    assert output_file.endswith('.json'), 'Output file must have .json extension'

    if loader_workers is None:
        loader_workers = DEFAULT_LOADER_WORKERS
    if threshold is None:
        threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if batch_size is None:
        batch_size = 1
    if include_image_size is None:
        include_image_size = False
    if optimize_for_inference is None:
        optimize_for_inference = False
    if worker_type is None:
        worker_type = 'thread'
    if tiled is None:
        tiled = False

    if (tile_overlap is not None) and (not tiled):
        print('A tile overlap was specified, enabling tiled inference')
        tiled = True
    if tile_overlap is None:
        tile_overlap = DEFAULT_TILE_OVERLAP

    if skip_images and skip_video:
        raise ValueError('Cannot skip both images and videos')

    if (frame_sample is not None) and (time_sample is not None):
        raise ValueError('frame_sample and time_sample are mutually exclusive')

    # Default the video sampling rate if the caller didn't specify one
    if (frame_sample is None) and (time_sample is None):
        time_sample = DEFAULT_SECONDS_PER_VIDEO_FRAME

    # Determine the set of images and videos to process, and the base folder used
    # to compute relative output paths.
    if os.path.isfile(image_folder):

        input_base_folder = os.path.dirname(image_folder)
        if is_video_file(image_folder):
            image_files = []
            video_files = [] if skip_video else [image_folder]
        else:
            image_files = [] if skip_images else [image_folder]
            video_files = []

    else:

        input_base_folder = image_folder

        if skip_images:
            image_files = []
        else:
            print(f'Searching for images in {image_folder}...')
            image_files = find_images(image_folder, recursive=True,
                                      return_relative_paths=False)
            print(f'Found {len(image_files)} images')

        if skip_video:
            video_files = []
        else:
            print(f'Searching for videos in {image_folder}...')
            video_files = find_videos(image_folder, recursive=True,
                                      return_relative_paths=False)
            print(f'Found {len(video_files)} videos')

    # ...whether the input is a file or a folder

    if (len(image_files) == 0) and (len(video_files) == 0):
        print('No images or videos found, exiting')
        return None

    # Load the model once; we'll use it for both images and videos
    model_info = load_model(detector_file,
                            image_size=image_size,
                            optimize_for_inference=optimize_for_inference,
                            batch_size=batch_size)
    model = model_info['model']
    model_type = model_info['model_type']
    image_size = model_info['image_size']
    detection_categories = model_info['detection_categories']

    # Tiles match the resolution the model runs at
    tile_size = image_size
    tile_overlap = _tile_overlap_pixels(tile_overlap, tile_size)
    if tiled:
        print(f'Using tiled inference with {tile_size}x{tile_size} tiles, '
              f'overlapping by {tile_overlap} pixels')

    # A model optimized for inference only accepts batches of the size it was compiled for
    pad_batches = optimize_for_inference

    results = []

    # Process images
    if len(image_files) > 0:
        results.extend(_run_detector_on_images(
            model=model,
            image_files=image_files,
            image_folder=input_base_folder,
            threshold=threshold,
            batch_size=batch_size,
            loader_workers=loader_workers,
            worker_type=worker_type,
            include_image_size=include_image_size,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            pad_batches=pad_batches))

    # Process videos
    if len(video_files) > 0:
        video_files_relative = \
            [os.path.relpath(fn, input_base_folder).replace('\\', '/')
             for fn in video_files]
        results.extend(_run_detector_on_videos(
            model=model,
            video_folder=input_base_folder,
            video_files_relative=video_files_relative,
            threshold=threshold,
            frame_sample=frame_sample,
            time_sample=time_sample,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            pad_batches=pad_batches,
            verbose=verbose))

    results = sort_list_of_dicts_by_key(results,'file')

    # Build output structure
    detector_metadata = {
        'model_type': model_type,
        'image_size': image_size,
        'tiled': tiled,
        'confidence_threshold': threshold
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

    # Write output
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Results written to {output_file}')

    return output

# ...def run_detector_batch(...)


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(
        description='Run DroneMegaDetector on a folder of images and/or videos (or a '
                    'single image or video file), producing MegaDetector-format output'
    )

    parser.add_argument(
        'detector_file',
        type=str,
        help='Path to RF-DETR checkpoint file (.pth)'
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to a folder containing images and/or videos (searched recursively), '
             'or to a single image or video file'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output JSON file'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help='Image resolution for inference (default: the training resolution recorded '
             'in the model file, or {} if the model file does not record it)'.format(
             DEFAULT_IMAGE_SIZE)
    )

    parser.add_argument(
        '--loader_workers',
        type=int,
        default=4,
        help='Number of parallel image loader workers (default: 4)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help='Confidence threshold for detections (default: {})'.format(
            DEFAULT_CONFIDENCE_THRESHOLD)
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (images per batch, or tiles per batch with '
             '--tiled) (default: 1)'
    )

    parser.add_argument(
        '--tiled',
        action='store_true',
        help='Run tiled inference, using overlapping tiles the size of the inference '
             'resolution'
    )

    parser.add_argument(
        '--tile_overlap',
        type=_parse_tile_overlap,
        default=None,
        help='Overlap between adjacent tiles, as an integer number of pixels, or as a '
             'fraction of the tile size between 0 and 1 (default: {} pixels); '
             'implies --tiled'.format(DEFAULT_TILE_OVERLAP)
    )

    parser.add_argument(
        '--include_image_size',
        action='store_true',
        help='Include image dimensions (width, height) in output'
    )

    parser.add_argument(
        '--optimize_for_inference',
        action='store_true',
        help='Optimize the model for inference after loading'
    )

    parser.add_argument(
        '--worker_type',
        type=str,
        default='thread',
        choices=['thread', 'process'],
        help='Use threads or processes for image loading workers (default: thread)'
    )

    parser.add_argument(
        '--skip_images',
        action='store_true',
        help='Ignore images, only process videos'
    )

    parser.add_argument(
        '--skip_video',
        action='store_true',
        help='Ignore videos, only process images'
    )

    parser.add_argument(
        '--frame_sample',
        type=int,
        default=None,
        help='Sample every Nth frame from videos (mutually exclusive with --time_sample)'
    )

    parser.add_argument(
        '--time_sample',
        type=float,
        default=None,
        help='Sample frames every N seconds from videos (default: {}); mutually '
             'exclusive with --frame_sample'.format(DEFAULT_SECONDS_PER_VIDEO_FRAME)
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable additional debug output'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    run_detector_batch(
        detector_file=args.detector_file,
        image_folder=args.folder,
        output_file=args.output_file,
        image_size=args.image_size,
        loader_workers=args.loader_workers,
        threshold=args.threshold,
        batch_size=args.batch_size,
        include_image_size=args.include_image_size,
        optimize_for_inference=args.optimize_for_inference,
        worker_type=args.worker_type,
        skip_images=args.skip_images,
        skip_video=args.skip_video,
        frame_sample=args.frame_sample,
        time_sample=args.time_sample,
        tiled=args.tiled,
        tile_overlap=args.tile_overlap,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
