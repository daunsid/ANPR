from typing import Any, cast, Protocol
import logging
import os

import pathlib
from collections.abc import Sequence
import numpy as np
from torch import Tensor
import cv2
from ultralytics import YOLO
from core.base import BoundingBox, DetectionResult, BaseDetector

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Yolov11ObjectDetector(BaseDetector):
    """
    YoloV11-specific Yolo inference class for performing object detection using the Yolo v11 pytorch model.
    """
    def __init__(
        self,
        model_path: str | os.PathLike[str],
    ):
        """
        Initializes the YoloV9ObjectDetector with the specified detection model and inference device.

        Args:
            model_path: Path to the ONNX model file to use.
        """
        # check if model path exists
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at '{model_path}'")
        self.model_name = model_path.stem
        self.model = YOLO(
            model_path
        )
        

    def predict(self, images:list[str | os.PathLike | np.ndarray]) :
        """
        Perform object detection on one or multiple images.

        Args:
            images: A single image as a numpy array, a single image path as a string, a list of images as numpy arrays,
                    or a list of image file paths.

        Returns:
            A list of DetectionResult for a single image input,
            or a list of lists of DetectionResult for multiple images.
        """
        # check the type of input and process accordingly
        if all(isinstance(img, str | os.PathLike) for img in images):
            # List of image paths
            loaded_images = [cv2.imread(str(img)) for img in images]
            for idx, img in enumerate(loaded_images):
                if img is None:
                    raise ValueError(f"Failed to load at path: {images[idx]}")
                return [self._predict(img) for img in loaded_images]
            
        if all(isinstance(img, np.ndarray) for img in images):
            # List of image arrays
            images = cast(list[np.ndarray], images)
            return [self._predict(img) for img in images]
        #raise TypeError("List must contain either all numpy arrays or all image file paths.")
        raise TypeError("Input must be a numpy array, a list of numpy arrays, or a list of image file paths.")

    def _predict(self, image: np.ndarray) -> list:
        """
        Perform object detection on a single image frame.

        This function takes an image in BGR format, runs it through the object detection model,
        and returns a list of detected objects, including their class labels and bounding boxes.

        Args:
            image: Input image frame in BGR format.

        Returns:
            A list of DetectionResult containing detected objects information.
        """
        # Preprocess the image using Yolov11 specifc preprocessing function
        
        # Run inference
        #image = preprocess(image, img_size=self.img_size)
        try:
            predictions = self.model(image)
            # pylint: ignore 
        except Exception as e:
            # Log a generic warning message with the exception details
            LOGGER.warning("An error occurred during model inference: %s", e)
            return []
        results = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                confidence = box.conf.item()
                xyxy = box.xyxy.numpy().tolist()[0]
                x1, y1, x2, y2 = xyxy
                bbox = BoundingBox(
                    x1=int(x1),
                    x2=int(x2),
                    y1=int(y1),
                    y2=int(y2)
                )
                detection_result = DetectionResult(
                    label="",
                    confidence=confidence,
                    bounding_box=bbox
                )
                results.append(detection_result)
            
        return results
    
    def display_prediction(self, image):
        results = self.model(image)
        for result in results:
            # boxes = result.boxes  # Boxes object for bounding box outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk