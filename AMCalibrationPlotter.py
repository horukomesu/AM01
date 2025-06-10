# AMCalibrationPlotter.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_calibration_results(
    calibrator,
    locator_names=None,
    image_paths=None
):
    """
    Визуализирует результат калибровки в 3D через matplotlib.

    calibrator      — объект CameraCalibrator после калибровки.
    locator_names   — список имён локаторов (если есть, иначе loc1, loc2, ...)
    image_paths     — список путей к изображениям (для подписей камер).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Локаторы (3D точки)
    points = calibrator.get_points_3d()
    if locator_names and len(locator_names) == len(points):
        for name, pt in zip(locator_names, points):
            if pt is not None:
                ax.scatter(*pt, c='g', marker='o')
                ax.text(*pt, name, color='green', fontsize=10)
    else:
        for i, pt in enumerate(points):
            if pt is not None:
                ax.scatter(*pt, c='g', marker='o')
                ax.text(*pt, f'loc{i+1}', color='green', fontsize=10)

    # Камеры
    cams = calibrator.get_camera_parameters()
    for idx, cam in enumerate(cams):
        if cam.center is None or cam.rotation is None:
            continue
        pos = np.array(cam.center).reshape(3)
        # Ось Z камеры в мировой системе (третья строка rotation)
        view_dir = np.array(cam.rotation)[2, :] * 50  # масштаб стрелки
        ax.scatter(*pos, c='b', marker='^')
        if image_paths and idx < len(image_paths):
            label = Path(image_paths[idx]).stem
        else:
            label = f'cam{idx+1}'
        ax.text(*pos, label, color='blue', fontsize=10)
        # Стрелка направления взгляда камеры
        ax.quiver(
            pos[0], pos[1], pos[2],
            view_dir[0], view_dir[1], view_dir[2],
            length=1.0, normalize=True, color='blue', arrow_length_ratio=0.15
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Calibration Result (3D)')
    ax.view_init(elev=20, azim=40)
    plt.tight_layout()
    plt.show()
