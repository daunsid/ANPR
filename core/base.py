from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Protocol, Any

@dataclass(frozen=True)
class BoundingBox:
    """
    Represents a bounding box with top-left and bottom-right coordinates.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Returns the width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Returns the height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Returns the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """
        Returns the (x, y) coordinates of the center of the bounding box.
        """
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0

        return cx, cy


@dataclass(frozen=True)
class DetectionResult:
    """
    Represents the result of an object detection.
    """

    label: str
    """Detected object label"""
    confidence: float
    """Confidence score of the detection"""
    bounding_box: BoundingBox
    """Bounding box of the detected object"""


class BaseDetector(Protocol):
    def predict(self, images: Any) -> list[DetectionResult] | list[list[DetectionResult]]:
        """
        Perform object detection on one or multiple images.

        Args:
            images: A single image as a numpy array, a single image path as a string, a list of images as numpy arrays,
                    or a list of image file paths.

        Returns:
            A list of DetectionResult for a single image input,
            or a list of lists of DetectionResult for multiple images.
        """

@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float


class BaseOCR(ABC):
    @abstractmethod
    def predict(self, cropped_plate: np.ndarray) -> OcrResult:
        """Perform OCR on the cropped plate image and return the recognized text and character
        probabilities."""
