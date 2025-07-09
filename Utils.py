import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def mask_to_bboxes(mask_path, min_area=100):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")

    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

        # area = w * h
        # if area >= min_area:
        #     bboxes.append((x, y, w, h))

    return bboxes

def draw_bboxes_on_image(image_path, bboxes, output_path=None):
    """
    Draws rectangles on the image and saves/displays the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    for (x, y, w, h) in bboxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow("BBoxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print(os.getcwd())
    data_dir = Path(os.getcwd()).parent / "Data"
    test_dir = data_dir / "test"
    mask_dir = test_dir / "mask"

    for folder_name in ["f1", "f2", "f3", "f4", "f5"]:
        subfolder = mask_dir / folder_name
        print(f"\nProcessing folder: {subfolder.name}")

        for mask_path in subfolder.iterdir():
            if mask_path.is_file():
                print(mask_path)
                bboxes = mask_to_bboxes(mask_path)
                print(bboxes)
                # draw_bboxes_on_image(mask_path, bboxes)
                break

        break