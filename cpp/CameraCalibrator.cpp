#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <colmap/base/database.h>
#include <colmap/controllers/incremental_mapper.h>
#include <colmap/sfm/incremental_triangulator.h>
#include <colmap/util/option_manager.h>
#include <colmap/util/temp_directory.h>
#include <Eigen/Core>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>

namespace py = pybind11;

struct CameraParameters {
    std::array<std::array<double,3>,3> intrinsics;
    std::array<std::array<double,3>,3> rotation;
    std::array<double,3> translation;
};

class CameraCalibrator {
public:
    CameraCalibrator() = default;
    ~CameraCalibrator() = default;

    void load_images(const std::vector<std::string>& paths) {
        image_paths_ = paths;
        image_names_.clear();
        image_sizes_.clear();
        for (const auto& p : paths) {
            image_names_.push_back(std::filesystem::path(p).filename().string());
            // In a complete implementation we would read the image size here.
            // Placeholder values
            image_sizes_.emplace_back(0,0);
        }
    }

    void load_point_data(const std::unordered_map<int, std::unordered_map<int, std::pair<double,double>>>& data) {
        point_data_ = data;
    }

    bool calibrate() {
        if (image_paths_.size() < 2) return false;
        if (point_data_.size() < 3) return false;

        try {
            colmap::OptionManager options;
            options.AddDefaultOptions();
            options.Parse();

            colmap::TempDirectory temp_dir;
            std::string db_path = (temp_dir.path() / "database.db").string();
            std::string image_dir = (temp_dir.path() / "images").string();
            std::filesystem::create_directory(image_dir);

            // Populate database
            colmap::Database db(db_path);
            std::unordered_map<int, colmap::camera_t> cam_ids;
            std::unordered_map<int, colmap::image_t> img_ids;

            for (size_t i=0;i<image_paths_.size();++i) {
                const auto& name = image_names_[i];
                const auto& size = image_sizes_[i];
                // Placeholder intrinsics
                const double f = 1.0;
                const double cx = size.first / 2.0;
                const double cy = size.second / 2.0;
                std::vector<double> params = {f,cx,cy};
                colmap::Camera camera(colmap::CameraModel::SIMPLE_PINHOLE, size.first, size.second, params);
                cam_ids[i] = db.WriteCamera(camera);
                colmap::Image image;
                image.SetName(name);
                image.SetCameraId(cam_ids[i]);
                img_ids[i] = db.WriteImage(image);
            }

            // Write keypoints and matches
            std::unordered_map<int, std::vector<Eigen::Vector4d>> keypoints_per_image;
            for (const auto& [set_id, obs] : point_data_) {
                for (const auto& [img_idx, xy] : obs) {
                    keypoints_per_image[img_idx].push_back(Eigen::Vector4d(xy.first, xy.second, 1.0, 0.0));
                }
            }
            for (const auto& [idx, kps] : keypoints_per_image) {
                if (kps.empty()) continue;
                Eigen::MatrixXf kp_mat(kps.size(),4);
                for (size_t i=0;i<kps.size();++i) kp_mat.row(i) = kps[i].cast<float>();
                Eigen::Matrix<uint8_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> desc(kps.size(),128);
                desc.setZero();
                db.WriteKeypoints(img_ids[idx], kp_mat);
                db.WriteDescriptors(img_ids[idx], desc);
            }

            // Matches
            for (size_t i=0;i<image_paths_.size();++i) {
                for (size_t j=i+1;j<image_paths_.size();++j) {
                    std::vector<Eigen::Vector2i> matches;
                    for (const auto& [set_id, obs] : point_data_) {
                        auto it1 = obs.find(i);
                        auto it2 = obs.find(j);
                        if (it1!=obs.end() && it2!=obs.end()) {
                            matches.emplace_back(matches.size(), matches.size());
                        }
                    }
                    if (matches.empty()) continue;
                    Eigen::Matrix<uint32_t,Eigen::Dynamic,2,Eigen::RowMajor> match_mat(matches.size(),2);
                    for (size_t k=0;k<matches.size();++k) match_mat.row(k) = matches[k].cast<uint32_t>();
                    db.WriteMatches(img_ids[i], img_ids[j], match_mat);
                    colmap::TwoViewGeometry geom;
                    geom.config = colmap::TwoViewGeometry::CALIBRATED;
                    geom.inlier_matches = match_mat;
                    db.WriteTwoViewGeometry(img_ids[i], img_ids[j], geom);
                }
            }

            // Run incremental mapper
            options.project_path = temp_dir.path().string();
            options.database_path = db_path;
            options.image_path = image_dir;
            options.output_path = (temp_dir.path() / "sparse").string();
            colmap::IncrementalMapperController mapper(&options);
            mapper.Start();
            mapper.Wait();
            const auto& recs = mapper.Reconstructions();
            if (recs.empty()) return false;
            const colmap::Reconstruction* largest = nullptr;
            size_t num_reg = 0;
            for (const auto& rec : recs) {
                if (rec.second->NumRegImages() > num_reg) {
                    num_reg = rec.second->NumRegImages();
                    largest = rec.second.get();
                }
            }
            if (!largest || num_reg < 2) return false;

            // Extract results
            calibration_results_.clear();
            for (const auto& image_id : largest->RegImageIds()) {
                const auto& image = largest->Image(image_id);
                const auto& camera = largest->Camera(image.CameraId());
                CameraParameters params;
                const auto& p = camera.Params();
                params.intrinsics = {{{p[0],0,p[1]},{0,p[0],p[2]},{0,0,1}}};
                Eigen::Matrix3d R_cw = image.Qvec().ToRotationMatrix();
                Eigen::Vector3d t_cw = image.Tvec();
                Eigen::Matrix3d R_wc = R_cw.transpose();
                Eigen::Vector3d t_wc = -R_wc*t_cw;
                for (int r=0;r<3;++r)
                    for (int c=0;c<3;++c)
                        params.rotation[r][c] = R_wc(r,c);
                for (int k=0;k<3;++k) params.translation[k]=t_wc(k);
                calibration_results_[image.ImageId()] = params;
            }
            for (const auto& p3d_id : largest->Point3DIds()) {
                const auto& pt = largest->Point3D(p3d_id);
                points3d_.push_back({pt.X(), pt.Y(), pt.Z()});
            }

            return true;
        } catch (const std::exception& e) {
            py::print("colmap exception:", e.what());
            return false;
        }
    }

    std::vector<CameraParameters> get_camera_parameters() const {
        std::vector<CameraParameters> out;
        for (const auto& [id, params] : calibration_results_) {
            out.push_back(params);
        }
        return out;
    }

    std::vector<std::array<double,3>> get_points_3d() const { return points3d_; }

private:
    std::vector<std::string> image_paths_;
    std::vector<std::string> image_names_;
    std::vector<std::pair<int,int>> image_sizes_;
    std::unordered_map<int, std::unordered_map<int,std::pair<double,double>>> point_data_;
    std::unordered_map<int, CameraParameters> calibration_results_;
    std::vector<std::array<double,3>> points3d_;
};

PYBIND11_MODULE(CameraCalibrator_cpp, m) {
    py::class_<CameraParameters>(m, "CameraParameters")
        .def_readwrite("intrinsics", &CameraParameters::intrinsics)
        .def_readwrite("rotation", &CameraParameters::rotation)
        .def_readwrite("translation", &CameraParameters::translation);

    py::class_<CameraCalibrator>(m, "CameraCalibrator")
        .def(py::init<>())
        .def("load_images", &CameraCalibrator::load_images)
        .def("load_point_data", &CameraCalibrator::load_point_data)
        .def("calibrate", &CameraCalibrator::calibrate)
        .def("get_camera_parameters", &CameraCalibrator::get_camera_parameters)
        .def("get_points_3d", &CameraCalibrator::get_points_3d);
}
