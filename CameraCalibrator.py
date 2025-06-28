from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pycolmap

@dataclass
class CameraParameters:
    intrinsics: List[List[float]]  # 3x3 матрица K
    rotation: List[List[float]]    # 3x3 матрица R
    translation: List[float]       # 3-элементный вектор t

class CameraCalibrator:
    def __init__(self) -> None:
        self.image_paths: List[str] = []
        self.image_shapes: List[Tuple[int, int]] = []
        self.point_data: Dict[int, Dict[int, Tuple[float, float]]] = {}
        self.image_names: List[str] = []
        self.calibration_results: Optional[Dict[str, Any]] = None
        self.keypoint_maps: Dict[int, Dict[int, int]] = {}
        self._temp_dir: Optional[str] = None

    def load_images(self, image_paths: List[str]) -> None:
        self.image_paths = list(image_paths)
        self.image_names = [Path(p).name for p in self.image_paths]
        self.image_shapes = [(Image.open(p).width, Image.open(p).height) for p in self.image_paths]

    def load_point_data(self, point_data: Dict[int, Dict[int, Tuple[float, float]]]) -> None:
        self.point_data = point_data

    def _populate_colmap_database(self, db_path: str) -> bool:
        """
        Creates and populates the COLMAP database using PyCOLMAP bindings,
        including manually writing matches and two-view geometries based
        on point_data. Requires image_shapes to be pre-loaded.

        Args:
            db_path: Absolute path to the COLMAP database file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not pycolmap:
                print("Error: PyCOLMAP is not available.")
                return False

            print(f"Creating COLMAP database: {db_path}")

            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)

            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    print(f"Removed existing database: {db_path}")
                except OSError as e:
                    print(f"Error removing existing database file:\n{db_path}\n{e}")
                    return False

            db = pycolmap.Database(db_path)
            print(f"Database object created at {db_path}")

            # Prepare keypoints per image
            keypoints_per_image = {}
            self.keypoint_maps = {idx: {} for idx in range(len(self.image_paths))}
            sorted_set_ids = sorted(self.point_data.keys())

            for img_idx in range(len(self.image_paths)):
                keypoints_per_image[img_idx] = []
                current_keypoint_idx = 0
                for set_id in sorted_set_ids:
                    if img_idx in self.point_data[set_id]:
                        x, y = self.point_data[set_id][img_idx]
                        keypoints_per_image[img_idx].append((x, y, 1.0, 0.0))
                        self.keypoint_maps[img_idx][set_id] = current_keypoint_idx
                        current_keypoint_idx += 1

            print("Prepared keypoint data and keypoint_maps.")

            # Use a transaction
            with pycolmap.DatabaseTransaction(db):
                SIMPLE_PINHOLE = pycolmap.CameraModelId.SIMPLE_PINHOLE
                camera_ids = {}
                image_ids = {}

                for img_idx, img_path in enumerate(self.image_paths):
                    basename = os.path.basename(img_path)
                    width, height = self.image_shapes[img_idx]
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid shape for image {img_path}")

                    f = 1.2 * max(width, height)
                    cx, cy = width / 2.0, height / 2.0
                    params = np.array([f, cx, cy], dtype=np.float64)

                    cam = pycolmap.Camera(
                        model=SIMPLE_PINHOLE, width=width, height=height, params=params
                    )
                    cam_id = db.write_camera(cam, use_camera_id=False)
                    camera_ids[img_idx] = cam_id

                    img = pycolmap.Image(name=basename, camera_id=cam_id)
                    img_id = db.write_image(img, use_image_id=False)
                    image_ids[img_idx] = img_id

                    keypoints = keypoints_per_image[img_idx]
                    if keypoints:
                        kp_array = np.array(keypoints, dtype=np.float32)
                        desc_array = np.zeros((len(keypoints), 128), dtype=np.uint8)
                        db.write_keypoints(img_id, kp_array)
                        db.write_descriptors(img_id, desc_array)

                print("Wrote all cameras, images, keypoints, and descriptors.")

                # Write matches and two-view geometries
                pair_count = 0
                for i in range(len(self.image_paths)):
                    for j in range(i + 1, len(self.image_paths)):
                        common_sets = set(self.keypoint_maps[i]) & set(self.keypoint_maps[j])
                        if not common_sets:
                            continue

                        matches = []
                        for set_id in common_sets:
                            kp1 = self.keypoint_maps[i][set_id]
                            kp2 = self.keypoint_maps[j][set_id]
                            matches.append((kp1, kp2))

                        if matches:
                            pair_count += 1
                            match_arr = np.array(matches, dtype=np.uint32)
                            db.write_matches(image_ids[i], image_ids[j], match_arr)

                            geom = pycolmap.TwoViewGeometry()
                            geom.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
                            geom.inlier_matches = match_arr
                            db.write_two_view_geometry(image_ids[i], image_ids[j], geom)

                print(f"Wrote matches for {pair_count} image pairs.")
            return True

        except Exception as e:
            print(f"Exception during COLMAP DB population: {e}")
            import traceback
            traceback.print_exc()
            return False


    def calibrate(self) -> bool:
        import traceback

        if len(self.image_paths) < 2:
            print("Calibration failed: Need at least 2 images.")
            return False

        if len(self.point_data) < 3:
            print(f"Calibration failed: Need at least 3 point sets (found {len(self.point_data)}).")
            return False

        if not self.point_data:
            print("Calibration failed: No point data.")
            return False

        self.calibration_results = None

        colmap_base_dir = tempfile.gettempdir()
        if hasattr(self, "current_save_path") and self.current_save_path:
            project_dir = os.path.dirname(os.path.abspath(self.current_save_path))
            if os.path.isdir(project_dir):
                colmap_base_dir = project_dir
            else:
                print(f"Warning: Project directory '{project_dir}' not found. Using system temp.")
        else:
            print("Warning: No project path set. Using system temp directory for COLMAP.")

        colmap_work_dir = os.path.join(colmap_base_dir, "colmap_py_work")
        if os.path.exists(colmap_work_dir):
            try:
                shutil.rmtree(colmap_work_dir)
            except OSError as e:
                print(f"Could not remove old directory: {e}")
                return False

        try:
            os.makedirs(colmap_work_dir)
        except OSError as e:
            print(f"Could not create directory: {e}")
            return False

        database_path_abs = os.path.join(colmap_work_dir, "database.db")
        image_copy_dir_abs = os.path.join(colmap_work_dir, "images")
        sparse_output_path_abs = os.path.join(colmap_work_dir, "sparse")

        try:
            os.makedirs(image_copy_dir_abs, exist_ok=True)
            for path in self.image_paths:
                shutil.copy2(path, os.path.join(image_copy_dir_abs, os.path.basename(path)))
        except Exception as e:
            print(f"Error copying images: {e}")
            shutil.rmtree(colmap_work_dir, ignore_errors=True)
            return False

        try:
            if not self._populate_colmap_database(database_path_abs):
                print("Error populating COLMAP database.")
                shutil.rmtree(colmap_work_dir, ignore_errors=True)
                return False
        except Exception as e:
            print(f"Error populating COLMAP DB: {e}")
            shutil.rmtree(colmap_work_dir, ignore_errors=True)
            return False

        os.makedirs(sparse_output_path_abs, exist_ok=True)

        options = pycolmap.IncrementalPipelineOptions()
        options.min_num_matches = 3
        options.mapper.init_min_num_inliers = 3
        options.mapper.init_min_tri_angle = 1.0
        options.mapper.abs_pose_min_num_inliers = 3
        options.mapper.abs_pose_max_error = 24.0
        options.mapper.filter_min_tri_angle = 0.0

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path_abs,
                image_path=image_copy_dir_abs,
                output_path=sparse_output_path_abs,
                options=options,
            )
        except Exception as e:
            print(f"PyCOLMAP mapping error:\n{e}\n{traceback.format_exc()}")
            return False

        if not reconstructions or not isinstance(reconstructions, dict):
            print("Calibration finished: No reconstruction dictionary.")
            return False

        largest_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images(), default=None)
        if not largest_rec or largest_rec.num_reg_images() < 2:
            print("Calibration finished: Insufficient registered images.")
            return False

        self.calibration_results = {
            "intrinsics": {},
            "poses": {},
            "points_3d": [],
            "point_ids": [],
            "registered_indices": [],
        }

        try:
            db = pycolmap.Database(database_path_abs)
            img_name_to_id = {img.name: img.image_id for img in db.read_all_images()}
            id_to_index = {}
            for idx, path in enumerate(self.image_paths):
                basename = os.path.basename(path)
                if basename in img_name_to_id:
                    id_to_index[img_name_to_id[basename]] = idx
        except Exception as e:
            print(f"Error reading image DB: {e}")
            return False

        for image_id, image in largest_rec.images.items():
            idx = id_to_index.get(image_id)
            if idx is None:
                continue
            cam = largest_rec.cameras[image.camera_id]
            if cam.model == pycolmap.CameraModelId.SIMPLE_PINHOLE and len(cam.params) == 3:
                f, cx, cy = cam.params
                self.calibration_results["intrinsics"][idx] = {
                    "K": [[f, 0, cx], [0, f, cy], [0, 0, 1]]
                }

            pose = image.cam_from_world
            R_w2c = pose.rotation.matrix()
            t_w2c = pose.translation
            R_c2w = R_w2c.T
            t_c2w = (-R_c2w @ t_w2c).tolist()

            self.calibration_results["poses"][idx] = {
                "R": R_c2w.tolist(),
                "t": t_c2w
            }
            self.calibration_results["registered_indices"].append(idx)

        self.calibration_results["points_3d"] = [pt.xyz.tolist() for pt in largest_rec.points3D.values()]
        self.calibration_results["point_ids"] = [None] * len(largest_rec.points3D)

        return True

    def get_camera_parameters(self) -> List[CameraParameters]:
        """
        Возвращает список параметров камеры (K, R, t) для всех зарегистрированных изображений.
        """
        if not self.calibration_results:
            return []

        result = []
        for img_idx in sorted(self.calibration_results["registered_indices"]):
            intr = self.calibration_results["intrinsics"].get(img_idx)
            pose = self.calibration_results["poses"].get(img_idx)
            if intr and pose:
                param = CameraParameters(
                    intrinsics=intr["K"],
                    rotation=pose["R"],
                    translation=pose["t"]
                )
                result.append(param)
        return result


    def get_points_3d(self):
        if not self.calibration_results:
            return []
        return [np.array(pt) for pt in self.calibration_results["points_3d"]]

    def get_reprojection_error(self) -> Optional[Dict[str, float]]:
        """Return reprojection error for each locator (set_id)."""
        if not self.calibration_results or not self.point_data:
            return None

        errors: Dict[str, float] = {}

        # Build projection matrices for registered images
        proj_mats: Dict[int, np.ndarray] = {}
        for img_idx in self.calibration_results["registered_indices"]:
            intr = self.calibration_results["intrinsics"].get(img_idx)
            pose = self.calibration_results["poses"].get(img_idx)
            if not intr or not pose:
                continue
            K = np.array(intr["K"], dtype=np.float64)
            R_c2w = np.array(pose["R"], dtype=np.float64)
            t_c2w = np.array(pose["t"], dtype=np.float64).reshape(3, 1)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            P = K @ np.hstack((R_w2c, t_w2c))
            proj_mats[img_idx] = P

        def triangulate(obs):
            A = []
            for img_idx, (x, y) in obs.items():
                if img_idx not in proj_mats:
                    continue
                P = proj_mats[img_idx]
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            if len(A) < 4:
                return None
            A = np.stack(A, axis=0)
            _, _, vt = np.linalg.svd(A)
            X = vt[-1]
            X /= X[3]
            return X[:3]

        for set_id, obs in self.point_data.items():
            pt3d = triangulate(obs)
            if pt3d is None:
                errors[str(set_id)] = float("inf")
                continue
            total_err = 0.0
            count = 0
            pt_h = np.hstack((pt3d, 1.0))
            for img_idx, (x, y) in obs.items():
                P = proj_mats.get(img_idx)
                if P is None:
                    continue
                proj = P @ pt_h
                proj /= proj[2]
                err = np.linalg.norm(np.array([x, y]) - proj[:2])
                total_err += err
                count += 1
            errors[str(set_id)] = total_err / count if count else float("inf")

        return errors
