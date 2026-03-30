import cv2
from src.anpr import ALPR
from src.config import PlateDetectionModel
from core.base import BoundingBox


def recognize_license_plate(image_path: str) -> list[dict]:
    """
    Perform license plate recognition
    Args:
        imgage_path: the local path for image file
    Returns:
        results: the recognised license plates
    """
    anpr = ALPR(
        detector_model=PlateDetectionModel,
        ocr_device="cpu",
    )
    license_plates = anpr.predict(image_path)
    result = {}
    confidence = 0
    for lp in license_plates:

        detection_confidence = lp.detection.confidence
        if detection_confidence > confidence:
            box: BoundingBox = lp.detection.bounding_box
            coordiates = {
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2
            }
            width = box.width
            height = box.height
            x, y = box.center
            license_plate_text = lp.ocr.text
            result = {
                "license_plate": license_plate_text,
                "confidence": detection_confidence,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "coordinate": coordiates,
                "status": ""
            }
            
    return result


def plate_crop(result: dict, image_path: str, cropped_image_path: str):
    coordinate = result["coordinate"]
    x1 = coordinate["x1"]
    y1 = coordinate["y1"]
    x2 = coordinate["x2"]
    y2 = coordinate["y2"]
    frame = cv2.imread(image_path)
    cropped_plate = frame[y1:y2, x1:x2]
    cv2.imwrite(cropped_image_path, cropped_plate)
