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

The model's training data predominantly represents drone flights conducted at approximately 50-120 m above ground level.  This range should be considered the principal operating range represented by the training data, rather than a strict altitude requirement.

As altitude increases, animals occupy fewer pixels in the image. Detection performance can therefore decline when targets become substantially smaller than those represented in the training imagery.


## Running the model

We provide two inference scripts:

* `inference-demo.py` is a simple script that runs on a single image, intended for quick testing to make sure the model runs properly in your environment.  This script creates and displays an annotated image, but doesn't write results to a file.
* `inference-batch.py` is intended for practical workflows; it processes a folder recursively and writes results to a .json file.

### Using inference-demo.py

`inference-demo.py` is designed to run the model on a single image, to quickly test whether the model is running correctly in your environment.

Usage:

```bash
python inference.py \
    --image path/to/image.jpg \
    --weights path/to/model.pth \
    --conf 0.5 \
    --output output.jpg
```

Arguments:

| Argument    | Required | Description                           |
| ----------- | -------- | ------------------------------------- |
| `--image`   | Yes      | Path to the input image               |
| `--weights` | Yes      | Path to the RF-DETR `.pth` checkpoint |
| `--conf`    | No       | Detection confidence threshold        |
| `--tiled`   | No       | Enables tiled inference               |
| `--output`  | No       | Path for the annotated output image   |

The default confidence threshold is `0.5`.

If the input filename is "image.jpg", the default output filename is "image.annotated.jpg".

### Using inference-batch.py

TODO


### Tiled inference

Both inference scripts support tiled inference for large images.  Tiled inference divides the input image into 1280 × 1280 pixel tiles (matching the resolution at which the model was trained) and performs detection independently on each tile before combining the detections.  Enable tiled inference using the `--tiled` argument.  

Tiled inference can be useful when each image is substantially larger than the model's training resolution (1280 × 1280), particularly when animals occupy a small proportion of the full image.

### HuggingFace

The model also has a huhhingface compatiability which can be found at:

https://huggingface.co/ConservationDrones/DroneMegaDetector
