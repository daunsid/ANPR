from src.pipline import recognize_license_plate, plate_crop


# Example:
IMAGE_PATH = "<image_path>"
CROPPED_IMAGE_PATH = "<image_path>"
results = recognize_license_plate(IMAGE_PATH)
plate_crop(results, IMAGE_PATH, CROPPED_IMAGE_PATH)
print(results)