import cv2
import numpy as np
from scipy.optimize import curve_fit
import os


def extractRedLine2HeightVector(img_path, save_path=None):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    redness = r - (g + b) / 2

    redness_norm = cv2.normalize(redness, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    edges = cv2.Canny(redness_norm, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    h, w = redness.shape

    height_vector = []
    for col_index in range(w):
        column = closed[:, col_index]
        white_positions = np.where(column == 255)[0]

        if len(white_positions) > 0:
            midpoint = int(np.mean(white_positions))
            height_vector.append(midpoint)
        else:
            height_vector.append(
                height_vector[len(height_vector) - 1] if height_vector else 0
            )

    return np.array(height_vector)


def bezier_x(t, px0, px1, px2, px3):
    u = 1 - t
    return u**3 * px0 + 3 * u**2 * t * px1 + 3 * u * t**2 * px2 + t**3 * px3


def bezier_y(t, py0, py1, py2, py3):
    u = 1 - t
    return u**3 * py0 + 3 * u**2 * t * py1 + 3 * u * t**2 * py2 + t**3 * py3


folder_path = "./warp_samples"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # image_path = "warp_samples/image_2035.png"
    height_vector = extractRedLine2HeightVector(file_path)

    print(f"Height vector length: {len(height_vector)}")

    x_data = np.arange(len(height_vector))
    y_data = height_vector.astype(float)

    x_start, x_end = 300, 630
    x_data = x_data[x_start:x_end]
    y_data = y_data[x_start:x_end]

    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()

    x_norm = (x_data - x_min) / (x_max - x_min + 1e-8)
    y_norm = (y_data - y_min) / (y_max - y_min + 1e-8)

    t_data = np.linspace(0, 1, len(x_norm))

    popt_x, _ = curve_fit(
        bezier_x, t_data, x_norm, p0=[0.0, 0.3, 0.7, 1.0], maxfev=5000
    )
    popt_y, _ = curve_fit(
        bezier_y, t_data, y_norm, p0=[0.0, 0.3, 0.7, 1.0], maxfev=5000
    )

    print(f"Fitted control points X: {popt_x}")
    print(f"Fitted control points Y: {popt_y}")

    t_fit = np.linspace(0, 1, 200)
    x_fit = bezier_x(t_fit, *popt_x)
    y_fit = bezier_y(t_fit, *popt_y)

    x_fit_orig = x_fit * (x_max - x_min) + x_min
    y_fit_orig = y_fit * (y_max - y_min) + y_min

    import matplotlib.pyplot as plt

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_with_rect = img_rgb.copy()
    y_start = int(y_min)
    y_end = int(y_max)
    cv2.rectangle(img_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    for i in range(len(x_fit_orig)):
        cv2.circle(
            img_with_rect, (int(x_fit_orig[i]), int(y_fit_orig[i])), 2, (255, 0, 0), -1
        )

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_with_rect)
    plt.title(f"Image with Fit Area - {filename}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(x_data, y_data, "b.", markersize=2, label="Extracted Points")
    plt.plot(x_fit_orig, y_fit_orig, "r-", linewidth=2, label="Fitted Bezier Curve")
    plt.gca().invert_yaxis()
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("Height Vector Curve Fitting")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bezier_fit_from_height_vector.png", dpi=150)
    plt.show()

    print("Result saved to bezier_fit_from_height_vector.png")
