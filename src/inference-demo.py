# ==========================================
# IMPORTS AND CONSTANTS
# ==========================================

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from functools import partial
import supervision as sv
from rfdetr.variants import RFDETRMedium

DEFAULT_INFERENCE_SIZE = 1280


# ==========================================
# DEVICE CONFIGURATION
# ==========================================

# Properly checks for Apple Silicon first, then CUDA, then defaults to CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_rf_resolution(checkpoint_path, default=DEFAULT_INFERENCE_SIZE):
    """
    Attempts to read the optimal resolution directly from the model weights.
    Falls back to a default value if missing.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'args' in ckpt and hasattr(ckpt['args'], 'resolution'):
            res = ckpt['args'].resolution
            print(f"[*] Detected native resolution {res} from {os.path.basename(checkpoint_path)}")
            return res
    except Exception as e:
        print(f"[!] Could not read resolution from checkpoint (using default {default}): {e}")
    return default

def rf_detr_callback(image_slice: np.ndarray, model, threshold: float = 0.01) -> sv.Detections:
    """
    Callback function required by supervision's InferenceSlicer.
    It takes an image slice (BGR numpy array), converts it to PIL RGB, and passes it to the model.
    """
    slice_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
    raw_detections = model.predict(slice_pil, threshold=threshold)

    # FIX: Supervision's InferenceSlicer crashes if custom metadata (like 'source_image')
    # differs across tiles. We extract just the core arrays to allow seamless merging.
    clean_detections = sv.Detections(
        xyxy=raw_detections.xyxy,
        confidence=raw_detections.confidence,
        class_id=raw_detections.class_id
    )
    return clean_detections


# ==========================================
# MAIN INFERENCE LOGIC
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Single Image Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the RF-DETR model checkpoint (.pth)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (e.g. 0.5)")
    parser.add_argument("--tiled", action="store_true",
                        help=f"Flag to enable tiled inference at {DEFAULT_INFERENCE_SIZE}x{DEFAULT_INFERENCE_SIZE}")
    parser.add_argument("--output", type=str, default=None, help="Path to save the annotated output image")

    args = parser.parse_args()

    # Load model
    print("\n[*] Loading RF-DETR Model...")
    # Dynamically extract resolution or use default
    res = get_rf_resolution(args.weights, default=DEFAULT_INFERENCE_SIZE)

    model = RFDETRMedium(pretrain_weights=args.weights, resolution=res)
    model.inference() # Speeds up inference pass, formerly optimize_for_inference()

    print(f"[*] Model loaded successfully. Device selected: {DEVICE.upper()}")

    # Read input image
    print(f"[*] Loading image from {args.image}...")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Could not read image. Please check the path: {args.image}")

    # Run inference
    if args.tiled:
        print(f"[*] Running tiled inference ({DEFAULT_INFERENCE_SIZE}x{DEFAULT_INFERENCE_SIZE} tiles)...")
        # Pre-configure the callback with the loaded model and confidence threshold
        callback_with_model = partial(rf_detr_callback, model=model, threshold=args.conf)

        # Initialize Supervision's slicer logic (removed overlap_ratio_wh to match original script)
        rf_slicer = sv.InferenceSlicer(
            callback=callback_with_model,
            slice_wh=(DEFAULT_INFERENCE_SIZE, DEFAULT_INFERENCE_SIZE),
            iou_threshold=0.5
        )
        detections = rf_slicer(image_bgr)
    else:
        print("[*] Running non-tiled inference...")
        # Normal inference expects a PIL Image in RGB format
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        detections = model.predict(image_pil, threshold=args.conf)

    # Process and draw annotations
    print(f"[*] Generating overlays for {len(detections)} detection(s)...")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=10)

    annotated_image = image_bgr.copy()

    # Draw bounding boxes
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # Create descriptive labels (class ID + confidence Score)
    labels = []
    if hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            labels.append(f"Cls {int(class_id)}: {confidence:.2f}")

        # Draw Labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

    # Write image
    if args.output is not None:
        output_filename = args.output
    else:
        bn,ext = os.path.splitext(args.image)
        output_filename = bn + '.annotated' + ext
    cv2.imwrite(output_filename, annotated_image)
    print(f"[*] Success, annotated image saved to: {output_filename}")

    # Show the image window
    print("[*] Displaying image. Press any key on the image window to exit.")

    height, width = annotated_image.shape[:2]
    if max(height, width) > 1000:
        scale = 1000 / max(height, width)
        display_img = cv2.resize(annotated_image, (int(width * scale), int(height * scale)))
    else:
        display_img = annotated_image

    cv2.imshow("RF-DETR Inference", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
