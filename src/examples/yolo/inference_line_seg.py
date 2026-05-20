from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob

model = YOLO("D:/code/pat-cv-lifecycle/src/examples/yolo/best (1).pt")
input_dir = "D:/datasets/warp_all_line/images"
output_dir = "D:/code/pat-cv-lifecycle/src/examples/yolo/results"

os.makedirs(output_dir, exist_ok=True)

png_files = glob(os.path.join(input_dir, "*.png"))

for img_path in png_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]

    results = model(img_path)
    result = results[0]

    if result.masks is not None:
        masks = result.masks.data
        combined_mask = None
        for i, mask in enumerate(masks):
            mask_img = (mask.cpu().numpy() * 255).astype("uint8")
            if combined_mask is None:
                combined_mask = mask_img
            else:
                combined_mask = np.maximum(combined_mask, mask_img)

        if combined_mask is not None:
            h, w = img.shape[:2]
            combined_mask = cv2.resize(combined_mask, (w, h))
            result_img = np.hstack(
                (img, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR))
            )
            output_path = os.path.join(output_dir, f"{name_without_ext}_result.png")
            cv2.imwrite(output_path, result_img)
            print(f"Saved: {output_path}")
    else:
        result_img = np.hstack((img, img))
        output_path = os.path.join(output_dir, f"{name_without_ext}_result.png")
        cv2.imwrite(output_path, result_img)
        print(f"Saved (no mask): {output_path}")

print(f"Done! Processed {len(png_files)} images.")
