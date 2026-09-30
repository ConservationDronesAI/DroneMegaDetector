# DroneMegaDetector

## Overview

This repository provides an object-detection model for drone-based wildlife surveys. The model is intended primarily for drone imagery, but the training data includes some imagery from crewed aircraft.


## Model

### Model categories

The detector is based on [RF-DETR Medium](https://rfdetr.roboflow.com/reference/medium/) and is trained as a three-class object detector:

| Class ID | Class   |
| -------: | ------- |
|        0 | Animal  |
|        1 | Person  |
|        2 | Vehicle |

### Flight altitude and image scale

The model's training data predominantly represents drone flights conducted at approximately 50--120 m above ground level.  This range should be considered the principal operating range represented by the training data, rather than a strict altitude requirement.

As altitude increases, animals occupy fewer pixels in the image. Detection performance can therefore decline when targets become substantially smaller than those represented in the training imagery.


## Running the model

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


### Tiled inference

The inference script supports optional tiled inference for large images.  Tiled inference divides the input image into 1280 × 1280 pixel tiles (matching the resolution at which the model was trained) and performs detection independently on each tile before combining the detections.  Enable tiled inference using the `--tiled` argument.

Tiled inference can be useful when each image is substantially larger than the model's training resolution (1280 × 1280), particularly when animals occupy a small proportion of the full image.
