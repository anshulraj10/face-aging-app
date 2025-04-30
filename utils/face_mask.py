import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_face_mask(image: Image.Image) -> Image.Image:
    image_rgb = np.array(image.convert("RGB"))
    results = face_detection.process(image_rgb)

    mask = Image.new("L", image.size, 0)
    if results.detections:
        draw = ImageDraw.Draw(mask)
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            w, h = image.size
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)
            expand = 20
            draw.rectangle(
                [
                    (max(xmin - expand, 0), max(ymin - expand, 0)),
                    (min(xmin + box_width + expand, w), min(ymin + box_height + expand, h))
                ],
                fill=255
            )
    return mask