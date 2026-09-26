# DroneMegaDetector

## Overview

This repository provides a trained **RF-DETR Medium object-detection model for large-scale aerial drone imagery**.

The model is designed as a general-purpose detection layer for aerial monitoring applications. It produces explicit bounding boxes around detected objects rather than only point locations or image-level counts. This makes the model suitable for wildlife detection and counting, while also allowing the detections to be used in downstream computer-vision workflows such as classification, tracking, behavioural analysis and individual re-identification.

The model was developed using heterogeneous aerial imagery and is intended for application to drone imagery acquired under a range of survey conditions.

The repository provides:

* Trained RF-DETR Medium model weights
* A standalone inference script
* Support for single-image inference
* Optional tiled inference for large images and smaller targets
* Automatic hardware selection
* Configurable confidence thresholds
* Automatic detection of the model's native inference resolution
* Annotated image output with bounding boxes and confidence scores

The model weights and inference code are provided together in this repository so that the detector can be directly deployed for local inference.

---

## Model

The detector is based on **RF-DETR Medium** and is trained as a three-class object detector:

| Class ID | Class   |
| -------: | ------- |
|        0 | Animal  |
|        1 | Person  |
|        2 | Vehicle |

The model is intended primarily for aerial drone imagery and can detect multiple objects within the same image.

The principal output is a bounding box for each detected object, together with its predicted class and confidence score.

This means that the model can be used for applications including:

* Wildlife detection
* Animal counting
* Aerial population surveys
* Human detection
* Vehicle detection
* Wildlife monitoring
* Object localisation
* Detection-based tracking
* Detection as an input to species classification
* Detection as an input to behavioural or individual-level analysis

The model is a **general-purpose aerial detector**, rather than a species-specific classifier. The `Animal` class therefore represents an object category and does not identify the animal to species level.

---

## Intended use

The model is designed for imagery collected using drones or other aerial platforms.

It is particularly suited to workflows in which a large number of aerial images need to be processed automatically and where explicit object localisation is useful.

A typical workflow is:

```text
Drone imagery
     |
     v
RF-DETR detection
     |
     +------------------+
     |                  |
     v                  v
  Counting        Bounding boxes
                       |
             +---------+---------+
             |         |         |
             v         v         v
        Tracking  Classification  Behaviour
```

The model can therefore function as a first-stage detector within a larger aerial computer-vision pipeline.

---

## Flight altitude and image scale

The training imagery predominantly represents drone flights conducted at approximately **50--120 m above ground level**.

This range should be considered the principal operating range represented by the training data rather than a strict altitude requirement.

Actual detection performance depends on several factors, including:

* Flight altitude
* Camera sensor
* Focal length
* Image resolution
* Ground sampling distance
* Animal body size
* Image quality
* Lighting conditions
* Viewing angle
* Target density

As altitude increases, animals occupy fewer pixels in the image. Detection performance can therefore decline when targets become substantially smaller than those represented in the training imagery.

Users intending to operate substantially outside the approximately 50--120 m range should validate the model using imagery representative of their intended survey conditions.

---

## Model weights

The trained model weights are provided with this repository.

The checkpoint is supplied as a PyTorch `.pth` file and can be loaded directly using the RF-DETR Medium architecture used by the inference script.

The inference script automatically attempts to read the native model resolution stored in the checkpoint. If the resolution cannot be recovered from the checkpoint, the script falls back to a resolution of 1280 pixels.

The model weights and inference code should be kept compatible with the RF-DETR implementation specified by the project.

---

## Requirements

The inference script requires Python and the following main packages:

* PyTorch
* RF-DETR
* NumPy
* OpenCV
* Pillow
* Supervision

Install the required packages according to the environment used for the project.

A CUDA-enabled NVIDIA GPU can be used where available. Apple Silicon systems can use Apple's MPS backend, while CPU inference is also supported.

---

## Hardware selection

The inference script automatically selects the available device.

The selection order is:

1. Apple Silicon MPS
2. NVIDIA CUDA
3. CPU

No manual device argument is required.

When running on an Apple Silicon Mac, the script will use the MPS backend when it is available.

When running on a system with a CUDA-compatible NVIDIA GPU, CUDA will be selected automatically.

If neither is available, inference falls back to the CPU.

---

## Basic inference

The supplied inference script is designed to run the model on a single image.

The basic command is:

```bash
python inference.py \
    --image path/to/image.jpg \
    --weights path/to/model.pth \
    --conf 0.5 \
    --output output.jpg
```

### Arguments

| Argument    | Required | Description                           |
| ----------- | -------- | ------------------------------------- |
| `--image`   | Yes      | Path to the input image               |
| `--weights` | Yes      | Path to the RF-DETR `.pth` checkpoint |
| `--conf`    | No       | Detection confidence threshold        |
| `--tiled`   | No       | Enables tiled inference               |
| `--output`  | No       | Path for the annotated output image   |

The default confidence threshold is `0.5`.

The default output filename is:

```text
output.jpg
```

---

## Example

If the model weights are stored in:

```text
weights/RFDETR-Medium.pth
```

and the input image is:

```text
images/drone_image.jpg
```

run:

```bash
python inference.py \
    --image images/drone_image.jpg \
    --weights weights/RFDETR-Medium.pth \
    --conf 0.5 \
    --output detections.jpg
```

The script will:

1. Load the RF-DETR Medium model.
2. Read the model resolution from the checkpoint where available.
3. Select the available inference device.
4. Load the input image.
5. Run object detection.
6. Draw bounding boxes around detected objects.
7. Add class IDs and confidence scores.
8. Save the annotated image.

---

## Confidence threshold

The confidence threshold controls which detections are returned.

For example:

```bash
--conf 0.5
```

will return detections meeting the specified confidence threshold.

A lower threshold can be useful when higher recall is required:

```bash
python inference.py \
    --image image.jpg \
    --weights model.pth \
    --conf 0.25 \
    --output detections.jpg
```

A higher threshold can be used when only higher-confidence detections are required:

```bash
python inference.py \
    --image image.jpg \
    --weights model.pth \
    --conf 0.75 \
    --output detections.jpg
```

There is no single confidence threshold that is necessarily optimal for every aerial survey. Threshold selection should therefore reflect the intended application and imagery.

For counting applications in particular, users should calibrate the confidence threshold using representative imagery from the intended survey.

---

## Tiled inference

The inference script supports optional tiled inference for large images.

Tiled inference divides the input image into **1280 × 1280 pixel tiles** and performs detection independently on each tile before combining the detections.

Enable tiled inference using:

```bash
python inference.py \
    --image path/to/image.jpg \
    --weights path/to/model.pth \
    --conf 0.5 \
    --tiled \
    --output output_tiled.jpg
```

Tiled inference can be useful when:

* The original drone image is substantially larger than the model's inference resolution.
* Animals occupy a small proportion of the full image.
* Higher effective target resolution is required.
* A large orthomosaic or aerial image is being processed.

Tiled inference increases the amount of image area processed by the model and will generally require more computation than single-pass inference.

---

## Standard versus tiled inference

### Standard inference

Standard inference processes the complete image in a single model pass.

```bash
python inference.py \
    --image image.jpg \
    --weights model.pth
```

This is appropriate when targets are sufficiently large at the model's input resolution.

### Tiled inference

Tiled inference processes the image using multiple 1280 × 1280 regions.

```bash
python inference.py \
    --image image.jpg \
    --weights model.pth \
    --tiled
```

This can improve the effective scale of small targets within large aerial images, although it requires additional computation.

The two approaches should therefore be selected according to image resolution and target size.

---

## Output

The inference script produces an annotated image containing the model's detections.

Each detection includes:

* Bounding box
* Class ID
* Confidence score

The output is saved using the path specified by:

```bash
--output
```

For example:

```bash
--output detected_animals.jpg
```

The original image is not modified.

The underlying RF-DETR detections remain bounding boxes. The script does not convert them to point locations.

---

## Using the detections in other workflows

The bounding-box representation can be used as the input to additional computer-vision or ecological workflows.

For example:

### Counting

Each detected animal can be counted directly from the bounding-box detections.

### Classification

The detected bounding boxes can be cropped and passed to a separate species or taxonomic classifier.

```text
Drone image
     |
     v
RF-DETR
     |
     v
Animal bounding boxes
     |
     v
Species classifier
     |
     v
Species identification
```

### Tracking

Bounding boxes can be passed to a multi-object tracking system to associate detections between video frames.

```text
Video
  |
  v
RF-DETR
  |
  v
Bounding boxes
  |
  v
Object tracker
  |
  v
Animal trajectories
```

### Behaviour and individual analysis

Bounding boxes can similarly provide the spatial input for downstream behavioural analysis, re-identification or other individual-level computer-vision systems.

---

## Batch processing

WIP

```bash
WIP
```
---


### Taxonomic variation

The model is not a species-specific detector. Performance can vary between taxa and between animal appearances that differ from those represented during training.

### Image conditions

Changes in illumination, vegetation, background, camera system, viewing angle and image quality can affect detection performance.

### Flight altitude

The principal training distribution represents approximately 50--120 m above ground level. Imagery acquired substantially outside this range should be validated before operational deployment.

### Confidence threshold

Detection behaviour depends on the selected confidence threshold. users should select and validate thresholds according to their application.

---

## Recommended workflow

For a new drone survey, the recommended workflow is:

1. Acquire representative aerial imagery.
2. Run the model using standard inference.
3. Inspect the resulting detections.
4. Adjust the confidence threshold if necessary.
5. Use tiled inference if targets are too small in the full-resolution image.
6. Validate detection performance on representative survey imagery.
7. Integrate the detections into the intended downstream workflow.

For large-scale surveys, threshold selection should be performed using a representative subset of the imagery before processing the complete dataset.

---

## Citation

WIP

```bibtex
% WIP
```

