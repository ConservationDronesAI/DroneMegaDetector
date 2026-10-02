import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection


# Hugging Face repository
repo_id = "ConservationDrones/DroneMegaDetector"

# Load the demo image, or an image supplied on the command line.
image_path = sys.argv[1] if len(sys.argv) > 1 else "example.png"
image = Image.open(image_path).convert("RGB")


# Load the processor and model from Hugging Face.
processor = AutoImageProcessor.from_pretrained(
    repo_id,
)

model = AutoModelForObjectDetection.from_pretrained(
    repo_id,
).eval()


# Use 4 CPU threads, as in the original local demo.
torch.set_num_threads(4)


# Preprocess the image and run inference.
encoded = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**encoded)


# Keep scores above 0.3 and map boxes back to the original image dimensions.
results = processor.post_process_object_detection(
    outputs,
    threshold=0.3,
    target_sizes=[(image.height, image.width)],
)[0]


# Print the class, confidence, and [xmin, ymin, xmax, ymax] in source pixels.
print(
    f"{image_path}: {len(results['scores'])} detections (score >= 0.3)"
)

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"],
):
    print(
        model.config.id2label[label.item()],
        f"{score.item():.3f}",
        [round(v, 1) for v in box.tolist()],
    )


# Draw the detections.
draw = ImageDraw.Draw(image)
font = ImageFont.load_default(size=18)

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"],
):
    bounds = box.tolist()

    text = (
        f"{model.config.id2label[label.item()]} "
        f"{score.item():.2f}"
    )

    draw.rectangle(
        bounds,
        outline="red",
        width=3,
    )

    text_x = min(
        max(2, bounds[0]),
        image.width - draw.textlength(text, font=font) - 2,
    )

    draw.text(
        (text_x, max(0, bounds[1] - 20)),
        text,
        font=font,
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )


# Save the annotated image.
output_path = "example_annotated.jpg"
image.save(output_path, quality=95)

print(f"Saved {output_path}")

