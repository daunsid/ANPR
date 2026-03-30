import pathlib

BASE_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent

"""Available PYTORCH models for doing inference."""
print(BASE_DIR)
PlateDetectionModel = BASE_DIR / "weights/detector/best-yolo.pt"
ClassLabel = "Plate"
DEVICE = "cpu"