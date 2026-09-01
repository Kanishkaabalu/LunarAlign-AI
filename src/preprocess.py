import cv2
import os

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast for lunar surface features
    enhanced = cv2.equalizeHist(gray)

    return enhanced


if __name__ == "__main__":
    print("Lunar image preprocessing module is ready!")