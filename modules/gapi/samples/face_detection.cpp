#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/highgui.hpp>

const std::string about =
"This is an OpenCV-based version of OMZ MTCNN Face Detection example";
const std::string keys =
"{ h help     |                           | Print this help message }"
"{ input      |                           | Path to the input video file }"
"{ mtcnnpm    | mtcnn-p.xml               | Path to OpenVINO MTCNN P (Proposal) detection model (.xml)}"
"{ mtcnnpd    | CPU                       | Target device for the MTCNN P (e.g. CPU, GPU, VPU, ...) }"
"{ mtcnnrm    | mtcnn-r.xml               | Path to OpenVINO MTCNN R (Refinement) detection model (.xml)}"
"{ mtcnnrd    | CPU                       | Target device for the MTCNN R (e.g. CPU, GPU, VPU, ...) }"
"{ mtcnnom    | mtcnn-o.xml               | Path to OpenVINO MTCNN O (Output) detection model (.xml)}"
"{ mtcnnod    | CPU                       | Target device for the MTCNN O (e.g. CPU, GPU, VPU, ...) }"
"{ thrp       | 0.6                       | MTCNN P confidence threshold}"
"{ thrr       | 0.7                       | MTCNN R confidence threshold}"
"{ thro       | 0.7                       | MTCNN O confidence threshold}"
"{ half_scale | false                     | MTCNN P use half scale pyramid}"
;

namespace {
std::string weights_path(const std::string& model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    const auto ext = model_path.substr(sz - EXT_LEN);
    CV_Assert(cv::toLowerCase(ext) == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
//////////////////////////////////////////////////////////////////////
} // anonymous namespace

namespace custom {
namespace {

// Define custom structures and operations
#define NUM_REGRESSIONS 4
#define NUM_PTS 5

struct BBox {
    double x1;
    double y1;
    double x2;
    double y2;

    cv::Rect getRect() const { return cv::Rect(static_cast<int>(x1),
                                               static_cast<int>(y1),
                                               static_cast<int>(x2 - x1),
                                               static_cast<int>(y2 - y1)); }

    BBox getSquare() const {
        BBox bbox;
        double bboxWidth = x2 - x1;
        double bboxHeight = y2 - y1;
        double side = std::max(bboxWidth, bboxHeight);
        bbox.x1 = static_cast<double>(x1) + (bboxWidth - side) * 0.5;
        bbox.y1 = static_cast<double>(y1) + (bboxHeight - side) * 0.5;
        bbox.x2 = bbox.x1 + side;
        bbox.y2 = bbox.y1 + side;
        return bbox;
    }
};

struct Face {
    BBox bbox;
    double score;
    std::array<double, NUM_REGRESSIONS> regression;
    double ptsCoords[2 * NUM_PTS];

    static void applyRegression(std::vector<Face>& faces, bool addOne = false) {
        for (auto& face : faces) {
            double bboxWidth =
                face.bbox.x2 - face.bbox.x1 + static_cast<double>(addOne);
            double bboxHeight =
                face.bbox.y2 - face.bbox.y1 + static_cast<double>(addOne);
            face.bbox.x1 = face.bbox.x1 + static_cast<double>(face.regression[1]) * bboxWidth;
            face.bbox.y1 = face.bbox.y1 + static_cast<double>(face.regression[0]) * bboxHeight;
            face.bbox.x2 = face.bbox.x2 + static_cast<double>(face.regression[3]) * bboxWidth;
            face.bbox.y2 = face.bbox.y2 + static_cast<double>(face.regression[2]) * bboxHeight;
        }
    }

    static void bboxes2Squares(std::vector<Face>& faces) {
        for (auto& face : faces) {
            face.bbox = face.bbox.getSquare();
        }
    }

    static std::vector<Face> runNMS(std::vector<Face>& faces, const double threshold,
                                    const bool useMin = false) {
        std::cout << "runNMS threshold " << threshold << " useMin " << useMin << std::endl;
        std::vector<Face> facesNMS;
        if (faces.empty()) {
            return facesNMS;
        }

        std::cout << "runNMS faces size before sort " << faces.size() << std::endl;
        for (size_t i = 0; i < faces.size(); ++i) {
             std::cout << "x1 "  << faces[i].bbox.x1 << " y1 " << faces[i].bbox.y1 << " x2 " << faces[i].bbox.x2 << " y2 " << faces[i].bbox.y2 << " score " << faces[i].score << std::endl;
        }

        std::sort(faces.begin(), faces.end(), [](const Face& f1, const Face& f2) {
            return f1.score > f2.score;
        });

        //std::cout << "runNMS faces size after sort " << faces.size() << std::endl;
        //for (size_t i = 0; i < faces.size(); ++i) {
        //     std::cout << "x1 "  << faces[i].bbox.x1 << " x2 " << faces[i].bbox.x2 << " y1 " << faces[i].bbox.y1 << " y2 " << faces[i].bbox.y2 << " score " << faces[i].score << std::endl;
        //}

        std::vector<int> indices(faces.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::cout << "runNMS indices size before while " << indices.size() << std::endl;
        while (indices.size() > 0) {
            const int idx = indices[0];
            facesNMS.push_back(faces[idx]);
            //std::cout << "runNMS indices size inside while " << indices.size() << " idx " << idx << std::endl;
            std::vector<int> tmpIndices = indices;
            indices.clear();
            const double area1 = (faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) *
                (faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
            for (size_t i = 1; i < tmpIndices.size(); ++i) {
                int tmpIdx = tmpIndices[i];
                //std::cout << "runNMS tmpIdx " << tmpIdx << std::endl;
                const double interX1 = std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1);
                const double interY1 = std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1);
                const double interX2 = std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2);
                const double interY2 = std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2);

                const double bboxWidth = std::max(0.0, (interX2 - interX1 + 1));
                const double bboxHeight = std::max(0.0, (interY2 - interY1 + 1));

                const double interArea = bboxWidth * bboxHeight;
                const double area2 = (faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) *
                    (faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
                double overlap = 0.0;
                if (useMin) {
                    overlap = interArea / std::min(area1, area2);
                } else {
                    overlap = interArea / (area1 + area2 - interArea);
                }
                //std::cout << "runNMS area1 " << area1 << " area2 " << area2 << " overlap " << overlap << " interArea " << interArea << std::endl;
                if (overlap <= threshold) {
                    //std::cout << "runNMS tmpIdx " << tmpIdx << std::endl;
                    indices.push_back(tmpIdx);
                }
            }
        }
        return facesNMS;
    }
};

const double P_NET_WINDOW_SIZE = 12.0;
const double P_NET_STRIDE = 2.0;

std::vector<Face> buildFaces(const cv::Mat& scores,
                             const cv::Mat& regressions,
                             const double scaleFactor,
                             const double threshold) {

    auto w = scores.size[3];
    auto h = scores.size[2];
    auto size = w * h;
    std::cout << "scores_data w " << w << " scores_data h " << h << std::endl;
    const float* scores_data = scores.ptr<float>();
    for (int i = 0; i < 200; i++)
    {
        std::cout << scores_data[i] << " ";
    }
    std::cout << std::endl;
    for (int i = size - 200; i < size; i++)
    {
        std::cout << scores_data[i] << " ";
    }
    std::cout << std::endl;


    scores_data += size;

    const float* reg_data = regressions.ptr<float>();

    auto wr = regressions.size[3];
    auto hr = regressions.size[2];
    std::cout << "regressions_data w "  << wr <<  " regressions_data h "  << hr << std::endl;
    //for(int i = 0; i < 20; i++)
    //{
    //    std::cout << reg_data[i] << " ";
    //}
    //std::cout << std::endl;

    //Python example
    //////////
    auto out_side = std::max(h, w);
    auto in_side = 2 * out_side + 11;
    double stride = 0.0;
    if (out_side != 1)
    {
        stride = static_cast<double>(in_side - 12) / static_cast<double>(out_side - 1);
    }
    std::cout << "stride " << stride << std::endl;
    std::cout << "threshold " << threshold << std::endl;
    //////////

    std::vector<Face> boxes;
    std::cout << "SCORES" << std::endl;
    for (int i = 0; i < size; i++) {
        if (scores_data[i] >= (float)(threshold)) {
            //std::cout  << scores_data[i]  << std::endl;
            int y = i / w;
            int x = i - w * y;

            Face faceInfo;
            BBox& faceBox = faceInfo.bbox;

            //faceBox.x1 = (static_cast<double>(x) * P_NET_STRIDE) / scaleFactor;
            //faceBox.y1 = (static_cast<double>(y) * P_NET_STRIDE) / scaleFactor;
            //faceBox.x2 = (static_cast<double>(x) * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.0) / scaleFactor;
            //faceBox.y2 = (static_cast<double>(y) * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.0) / scaleFactor;
            faceBox.x1 = (static_cast<double>(x) * stride) / scaleFactor;
            faceBox.y1 = (static_cast<double>(y) * stride) / scaleFactor;
            faceBox.x2 = (static_cast<double>(x) * stride + P_NET_WINDOW_SIZE - 1.0) / scaleFactor;
            faceBox.y2 = (static_cast<double>(y) * stride + P_NET_WINDOW_SIZE - 1.0) / scaleFactor;
            faceInfo.regression[0] = reg_data[i];
            faceInfo.regression[1] = reg_data[i + size];
            faceInfo.regression[2] = reg_data[i + 2 * size];
            faceInfo.regression[3] = reg_data[i + 3 * size];
            faceInfo.score = scores_data[i];
            //boxes.push_back(faceInfo);
            //Python example
            //////////
            if ((faceBox.x2 > faceBox.x1) && (faceBox.y2 > faceBox.y1)) {
                boxes.push_back(faceInfo);
            }
            else {
                std::cout << "faceBox skipped: " << "x2 " << faceBox.x2 << " x1 " << faceBox.x2 << " y2 " << faceBox.y2 << " y1 " << faceBox.y1 << std::endl;

            }
            //////////

        }
    }

    return boxes;
}

// Define networks for this sample
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
using GMats = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;

G_API_NET(MTCNNRefinement,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_refinement");

G_API_NET(MTCNNOutput,
          <GMat3(cv::GMat)>,
          "sample.custom.mtcnn_output");

using GFaces = cv::GArray<Face>;
G_API_OP(BuildFaces,
         <GFaces(cv::GMat, cv::GMat, double, double)>,
         "sample.custom.mtcnn.build_faces") {
         static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                       const cv::GMatDesc&,
                                       const double,
                                       const double) {
              return cv::empty_array_desc();
    }
};

G_API_OP(RunNMS,
         <GFaces(GFaces, double, bool)>,
         "sample.custom.mtcnn.run_nms") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const double, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(AccumulatePyramidOutputs,
         <GFaces(GFaces, GFaces)>,
         "sample.custom.mtcnn.accumulate_pyramid_outputs") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ApplyRegression,
         <GFaces(GFaces, bool)>,
         "sample.custom.mtcnn.apply_regression") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(BBoxesToSquares,
         <GFaces(GFaces)>,
         "sample.custom.mtcnn.bboxes_to_squares") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(R_O_NetPreProcGetROIs,
         <GRects(GFaces, GSize)>,
         "sample.custom.mtcnn.bboxes_r_o_net_preproc_get_rois") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GOpaqueDesc&) {
              return cv::empty_array_desc();
    }
};


G_API_OP(RNetPostProc,
         <GFaces(GFaces, GMats, GMats, double)>,
         "sample.custom.mtcnn.rnet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const double) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ONetPostProc,
         <GFaces(GFaces, GMats, GMats, GMats, double)>,
         "sample.custom.mtcnn.onet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const double) {
             return cv::empty_array_desc();
    }
};

G_API_OP(SwapFaces,
         <GFaces(GFaces)>,
         "sample.custom.mtcnn.swap_faces") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(Transpose,
         <cv::GMat(cv::GMat)>,
         "sample.custom.mtcnn.transpose") {
          static cv::GMatDesc outMeta(const cv::GMatDesc in) {
               return in.withSize(cv::Size(in.size.height, in.size.width));
    }
};

//Custom kernels implementation
GAPI_OCV_KERNEL(OCVBuildFaces, BuildFaces) {
    static void run(const cv::Mat & in_scores,
                    const cv::Mat & in_regresssions,
                    const double scaleFactor,
                    const double threshold,
                    std::vector<Face> &out_faces) {
        out_faces = buildFaces(in_scores, in_regresssions, scaleFactor, threshold);
        std::cout << "OCVBuildFaces!!! faces number " << out_faces.size() <<
            " scaleFactor " << scaleFactor <<
            " threshold " << threshold << std::endl;
    }
};// GAPI_OCV_KERNEL(BuildFaces)

GAPI_OCV_KERNEL(OCVRunNMS, RunNMS) {
    static void run(const std::vector<Face> &in_faces,
                    const double threshold,
                    const bool useMin,
                    std::vector<Face> &out_faces) {
                    std::vector<Face> in_faces_copy = in_faces;
        out_faces = Face::runNMS(in_faces_copy, threshold, useMin);
        std::cout << "OCVRunNMS!!! in_faces size " << in_faces.size() <<
            " out_faces size " << out_faces.size() <<
            " for threshold " << threshold <<
            " and useMin " << useMin << std::endl;
    }
};// GAPI_OCV_KERNEL(RunNMS)

GAPI_OCV_KERNEL(OCVAccumulatePyramidOutputs, AccumulatePyramidOutputs) {
    static void run(const std::vector<Face> &total_faces,
                    const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
                    out_faces = total_faces;
        out_faces.insert(out_faces.end(), in_faces.begin(), in_faces.end());
        std::cout << "OCVAccumulatePyramidOutputs!!! output faces number " << out_faces.size() << std::endl;
    }
};// GAPI_OCV_KERNEL(AccumulatePyramidOutputs)

GAPI_OCV_KERNEL(OCVApplyRegression, ApplyRegression) {
    static void run(const std::vector<Face> &in_faces,
                    const bool addOne,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::applyRegression(in_faces_copy, addOne);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
        std::cout << "OCVApplyRegression!!! in_faces size " << in_faces.size() <<
            " out_faces size " << out_faces.size() << " and addOne " << addOne << std::endl;
    }
};// GAPI_OCV_KERNEL(ApplyRegression)

GAPI_OCV_KERNEL(OCVBBoxesToSquares, BBoxesToSquares) {
    static void run(const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::bboxes2Squares(in_faces_copy);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
        std::cout << "OCVBBoxesToSquares!!! input faces number " << in_faces.size() <<
            " output faces number " << out_faces.size() << std::endl;
    }
};// GAPI_OCV_KERNEL(BBoxesToSquares)

GAPI_OCV_KERNEL(OCVR_O_NetPreProcGetROIs, R_O_NetPreProcGetROIs) {
    static void run(const std::vector<Face> &in_faces,
                    const cv::Size & in_image_size,
                    std::vector<cv::Rect> &outs) {
        outs.clear();
        for (const auto& face : in_faces) {
            cv::Rect tmp_rect = face.bbox.getRect();
            //Compare to transposed sizes width<->height
            tmp_rect &= cv::Rect(tmp_rect.x, tmp_rect.y, in_image_size.height - tmp_rect.x - 4, in_image_size.width - tmp_rect.y - 4);
            outs.push_back(tmp_rect);
        }
        std::cout << "OCVR_O_NetPreProcGetROIs!!! input faces number " << in_faces.size() <<
            " output faces number " << outs.size() << std::endl;
    }
};// GAPI_OCV_KERNEL(R_O_NetPreProcGetROIs)


GAPI_OCV_KERNEL(OCVRNetPostProc, RNetPostProc) {
    static void run(const std::vector<Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const double threshold,
                    std::vector<Face> &out_faces) {
        out_faces.clear();
        std::cout << "OCVRNetPostProc!!! input scores number " << in_scores.size() <<
            " input regressions number " << in_regresssions.size() <<
            " input faces size " << in_faces.size() << std::endl;
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                std::cout << "OCVRNetPostProc!!! scores_data[0] " << scores_data[0] << " scores_data[1] " << scores_data[1] << std::endl;
                std::cout << "OCVRNetPostProc!!! reg_data[0] " << reg_data[0] << " reg_data[1] " << reg_data[1] <<
                    "reg_data[2] " << reg_data[2] << " reg_data[3] " << reg_data[3] << std::endl;
                Face info = in_faces[k];
                info.score = scores_data[1];
                std::copy_n(reg_data, NUM_REGRESSIONS, info.regression.begin());
                out_faces.push_back(info);
            }
        }
        std::cout << "OCVRNetPostProc!!! out faces number " << out_faces.size() <<
            " for threshold " << threshold << std::endl;
    }
};// GAPI_OCV_KERNEL(RNetPostProc)

GAPI_OCV_KERNEL(OCVONetPostProc, ONetPostProc) {
    static void run(const std::vector<Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const std::vector<cv::Mat> &in_landmarks,
                    const double threshold,
                    std::vector<Face> &out_faces) {
        out_faces.clear();
        std::cout << "OCVONetPostProc!!! input scores number " << in_scores.size() <<
            " input regressions number " << in_regresssions.size() <<
            " input landmarks number " << in_landmarks.size() <<
            " input faces size " << in_faces.size() << std::endl;
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            const float* landmark_data = in_landmarks[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                std::cout << "OCVONetPostProc!!! scores_data[0] " << scores_data[0] << " scores_data[1] " << scores_data[1] << std::endl;
                std::cout << "OCVONetPostProc!!! reg_data[0] " << reg_data[0] << " reg_data[1] " << reg_data[1] <<
                    " reg_data[2] " << reg_data[2] << " reg_data[3] " << reg_data[3] << std::endl;
                Face info = in_faces[k];
                info.score = scores_data[1];
                for (int i = 0; i < 4; ++i) {
                    info.regression[i] = reg_data[i];
                }
                double w = info.bbox.x2 - info.bbox.x1 + 1.0;
                double h = info.bbox.y2 - info.bbox.y1 + 1.0;

                for (int p = 0; p < NUM_PTS; ++p) {
                    info.ptsCoords[2 * p] =
                        info.bbox.x1 + static_cast<double>(landmark_data[NUM_PTS + p]) * w - 1;
                    info.ptsCoords[2 * p + 1] = info.bbox.y1 + static_cast<double>(landmark_data[p]) * h - 1;
                }

                out_faces.push_back(info);
            }
        }
        std::cout << "OCVONetPostProc!!! out faces number " << out_faces.size() << " for threshold " << threshold << std::endl;
    }
};// GAPI_OCV_KERNEL(ONetPostProc)

GAPI_OCV_KERNEL(OCVSwapFaces, SwapFaces) {
    static void run(const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        out_faces.clear();
        if (!in_faces_copy.empty()) {
            for (size_t i = 0; i < in_faces_copy.size(); ++i) {
                std::cout << "OCVSwapFaces!!! score " << in_faces_copy[i].score << std::endl;
                std::cout << "OCVSwapFaces!!! regression[0] " << in_faces_copy[i].regression[0] << " regression[1] " << in_faces_copy[i].regression[1] <<
                    " regression[2] " << in_faces_copy[i].regression[2] << " regression[3] " << in_faces_copy[i].regression[3] << std::endl;
                std::swap(in_faces_copy[i].bbox.x1, in_faces_copy[i].bbox.y1);
                std::swap(in_faces_copy[i].bbox.x2, in_faces_copy[i].bbox.y2);
                for (int p = 0; p < NUM_PTS; ++p) {
                    std::swap(in_faces_copy[i].ptsCoords[2 * p], in_faces_copy[i].ptsCoords[2 * p + 1]);
                }
            }
            out_faces = in_faces_copy;
        }
    }
};// GAPI_OCV_KERNEL(SwapFaces)

GAPI_OCV_KERNEL(OCVTranspose, Transpose) {
    static void run(const cv::Mat &in_mat,
                    cv::Mat &out_mat) {
        cv::transpose(in_mat, out_mat);
    }
};// GAPI_OCV_KERNEL(Transpose)
} // anonymous namespace
} // namespace custom

namespace vis {
namespace {
void bbox(const cv::Mat& m, const cv::Rect& rc) {
    std::cout << "Final rectangle " << "x1 = " << rc.x << " y1 = " << rc.y << " x2 = " << rc.x + rc.width << " y2 = " << rc.y + rc.height << std::endl;
    cv::rectangle(m, rc, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
};

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat& img,
    const std::vector<rectPoints> data) {
    cv::Mat outImg;
    img.copyTo(outImg);

    for (const auto& el : data) {
        vis::bbox(outImg, el.first);
        auto pts = el.second;
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
        }
    }
    return outImg;
}
} // anonymous namespace
} // namespace vis


//Infer helper function
namespace {
static inline std::tuple<cv::GMat, cv::GMat> run_mtcnn_p(cv::GMat &in, const std::string &id) {
    cv::GInferInputs inputs;
    inputs["data"] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(id, inputs);
    auto regressions = outputs.at("conv4-2");
    auto scores = outputs.at("prob1");
    return std::make_tuple(regressions, scores);
}

//Operator fot PNet network package creation in the loop
inline cv::gapi::GNetPackage& operator += (cv::gapi::GNetPackage& lhs, const cv::gapi::GNetPackage& rhs) {
    lhs.networks.reserve(lhs.networks.size() + rhs.networks.size());
    lhs.networks.insert(lhs.networks.end(), rhs.networks.begin(), rhs.networks.end());
    return lhs;
}

static inline std::string get_pnet_level_name(const cv::Size &in_size) {
    return "MTCNNProposal_" + std::to_string(in_size.width) + "x" + std::to_string(in_size.height);
}

int calculate_scales(const cv::Size &input_size, std::vector<double> &out_scales, std::vector<cv::Size> &out_sizes ) {
    //calculate multi - scale and limit the maxinum side to 1000
    //pr_scale: limit the maxinum side to 1000, < 1.0
    double pr_scale = 1.0;
    double h = static_cast<double>(input_size.height);
    double w = static_cast<double>(input_size.width);
    if (std::min(w, h) > 1000)
    {
        pr_scale = 1000.0 / std::min(h, w);
        w = w * pr_scale;
        h = h * pr_scale;
    }
    else if (std::max(w, h) < 1000)
    {
        w = w * pr_scale;
        h = h * pr_scale;
    }
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.709;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12)
    {
        const double current_scale = pr_scale * std::pow(factor, factor_count);
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        std::cout << "current_scale " << current_scale << std::endl;
        std::cout << "current_size " << current_size << std::endl;
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
    }
    std::cout << "factor_count " << factor_count << std::endl;
    return factor_count;
}

int calculate_half_scales(const cv::Size &input_size, std::vector<double>& out_scales, std::vector<cv::Size>& out_sizes) {
    double pr_scale = 0.5;
    const double h = static_cast<double>(input_size.height);
    const double w = static_cast<double>(input_size.width);
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.5;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12.0*2.0)
    {
        const double current_scale = pr_scale;
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        std::cout << "current_scale " << current_scale << std::endl;
        std::cout << "current_size " << current_size << std::endl;
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
        pr_scale *= 0.5;
    }
    std::cout << "factor_count " << factor_count << std::endl;
    return factor_count;
}

const int MAX_PYRAMID_LEVELS = 13;
//////////////////////////////////////////////////////////////////////
} // anonymous namespace

int main(int argc, char* argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const auto input_file_name = cmd.get<std::string>("input");
    const auto model_path_p = cmd.get<std::string>("mtcnnpm");
    const auto target_dev_p = cmd.get<std::string>("mtcnnpd");
    const auto conf_thresh_p = cmd.get<double>("thrp");
    const auto model_path_r = cmd.get<std::string>("mtcnnrm");
    const auto target_dev_r = cmd.get<std::string>("mtcnnrd");
    const auto conf_thresh_r = cmd.get<double>("thrr");
    const auto model_path_o = cmd.get<std::string>("mtcnnom");
    const auto target_dev_o = cmd.get<std::string>("mtcnnod");
    const auto conf_thresh_o = cmd.get<double>("thro");
    const auto use_half_scale = cmd.get<bool>("half_scale");

    std::vector<cv::Size> level_size;
    std::vector<double> scales;
    //MTCNN input size
    cv::VideoCapture cap;
    cap.open(input_file_name);
    if (!cap.isOpened())
        CV_Assert(false);
    auto in_rsz = cv::Size{ static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) };
    //Calculate scales, number of pyramid levels and sizes for PNet pyramid
    auto pyramid_levels = use_half_scale ? calculate_half_scales(in_rsz, scales, level_size) :
                                           calculate_scales(in_rsz, scales, level_size);
    CV_Assert(pyramid_levels <= MAX_PYRAMID_LEVELS);

    //Proposal part of MTCNN graph
    //Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
    cv::GMat in_original;
    cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_original);
    cv::GOpaque<cv::Size> in_sz = cv::gapi::streaming::size(in_original);
    cv::GMat in_resized[MAX_PYRAMID_LEVELS];
    cv::GMat in_transposed[MAX_PYRAMID_LEVELS];
    cv::GMat regressions[MAX_PYRAMID_LEVELS];
    cv::GMat scores[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> nms_p_faces[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> total_faces[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> faces_init(std::vector<custom::Face>{});

    //The very first PNet pyramid layer to init total_faces[0]
    in_resized[0] = cv::gapi::resize(in_originalRGB, level_size[0]);
    in_transposed[0] = custom::Transpose::on(in_resized[0]);
    std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposed[0], get_pnet_level_name(level_size[0]));
    cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores[0], regressions[0], scales[0], conf_thresh_p);
    cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(faces0, false);
    cv::GArray<custom::Face> final_faces_pnet0 = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);
    nms_p_faces[0] = custom::RunNMS::on(final_faces_pnet0, 0.5, false);
    total_faces[0] = custom::AccumulatePyramidOutputs::on(faces_init, nms_p_faces[0]);
    //The rest PNet pyramid layers to accumlate all layers result in total_faces[PYRAMID_LEVELS - 1]]
#if 1
    for (int i = 1; i < pyramid_levels; ++i)
    {
        in_resized[i] = cv::gapi::resize(in_originalRGB, level_size[i]);
        in_transposed[i] = custom::Transpose::on(in_resized[i]);
        std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposed[i], get_pnet_level_name(level_size[i]));
        cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i], regressions[i], scales[i], conf_thresh_p);
        cv::GArray<custom::Face> final_p_faces_for_bb2squares_i = custom::ApplyRegression::on(faces, false);
        cv::GArray<custom::Face> final_faces_pnet_i = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares_i);
        nms_p_faces[i] = custom::RunNMS::on(final_faces_pnet_i, 0.5, false);
        total_faces[i] = custom::AccumulatePyramidOutputs::on(total_faces[i - 1], nms_p_faces[i]);
    }

    //Proposal post-processing
    //cv::GArray<custom::Face> nms07_p_faces_total = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7, false);
    //Python example
    //////////
    cv::GArray<custom::Face> final_faces_pnet = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7, true);
    //////////
    //cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(nms07_p_faces_total, false);
    //cv::GArray<custom::Face> final_faces_pnet = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);

    //Refinement part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet, in_sz);
    cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
    cv::GMat in_originalRGB_transposed = custom::Transpose::on(in_originalRGB);
    std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<custom::MTCNNRefinement>(faces_roi_pnet, in_originalRGB_transposed);

    //Refinement post-processing
    cv::GArray<custom::Face> rnet_post_proc_faces = custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, conf_thresh_r);
    cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7, false);
    cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
    cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

    //Output part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet, in_sz);
    cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
    std::tie(regressionsONet, landmarksONet, scoresONet) = cv::gapi::infer<custom::MTCNNOutput>(faces_roi_rnet, in_originalRGB_transposed);

    //Output post-processing
    cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet, scoresONet, regressionsONet, landmarksONet, conf_thresh_o);
    cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
    cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7, true);
    cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);
#endif
    cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), final_faces_onet));
    //cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), total_faces[0]));

    // MTCNN Refinement detection network
    auto mtcnnr_net = cv::gapi::ie::Params<custom::MTCNNRefinement>{
        model_path_r,                // path to topology IR
        weights_path(model_path_r),  // path to weights
        target_dev_r,                // device specifier
    }.cfgOutputLayers({ "conv5-2", "prob1" }).cfgInputLayers({ "data" });

    // MTCNN Output detection network
    auto mtcnno_net = cv::gapi::ie::Params<custom::MTCNNOutput>{
        model_path_o,                // path to topology IR
        weights_path(model_path_o),  // path to weights
        target_dev_o,                // device specifier
    }.cfgOutputLayers({ "conv6-2", "conv6-3", "prob1" }).cfgInputLayers({ "data" });

    auto networks_mtcnn = cv::gapi::networks(mtcnnr_net, mtcnno_net);

    // MTCNN Proposal detection network
    for (int i = 0; i < pyramid_levels; ++i)
    {
        std::string net_id = get_pnet_level_name(level_size[i]);
        std::vector<size_t> reshape_dims = { 1, 3, (size_t)level_size[i].width, (size_t)level_size[i].height };
        cv::gapi::ie::Params<cv::gapi::Generic> mtcnnp_net{
                    net_id,                      // tag
                    model_path_p,                // path to topology IR
                    weights_path(model_path_p),  // path to weights
                    target_dev_p,                // device specifier
        };
        mtcnnp_net.cfgInputReshape({ {"data", reshape_dims} });
        networks_mtcnn += cv::gapi::networks(mtcnnp_net);
    }

    auto kernels_mtcnn = cv::gapi::kernels< custom::OCVBuildFaces
                                          , custom::OCVRunNMS
                                          , custom::OCVAccumulatePyramidOutputs
                                          , custom::OCVApplyRegression
                                          , custom::OCVBBoxesToSquares
                                          , custom::OCVR_O_NetPreProcGetROIs
                                          , custom::OCVRNetPostProc
                                          , custom::OCVONetPostProc
                                          , custom::OCVSwapFaces
                                          , custom::OCVTranspose
    >();
    auto pipeline_mtcnn = graph_mtcnn.compileStreaming(cv::compile_args(networks_mtcnn, kernels_mtcnn));

    std::cout << "Reading " << input_file_name << std::endl;
#if 1
    // Input image
    cv::TickMeter tm;
    //static (comment out for video cap) 
    auto in_orig = cv::imread(input_file_name);
    // remove comment out for video cap
    //cv::Mat in_orig;
    //cap.read(in_orig);

    cv::Mat in_src;
    in_orig.copyTo(in_src);
    auto graph_mtcnn_compiled = graph_mtcnn.compile(descr_of(gin(in_src)), cv::compile_args(networks_mtcnn, kernels_mtcnn));
    tm.start();
    int frames = 0;
    std::vector<custom::Face> out_faces;
    while (cv::waitKey(1) < 0) {
    //while (cap.read(in_orig)) {
        frames++;
        std::cout << "Frame " << frames << std::endl;
        tm.stop();
        //auto in_orig1 = cv::imread(input_file_name);
        //auto graph_mtcnn_compiled1 = graph_mtcnn.compile(descr_of(gin(in_orig)), cv::compile_args(networks_mtcnn, kernels_mtcnn));
        tm.start();
        //graph_mtcnn_compiled1(gin(in_orig), gout(in_orig, out_faces));
        graph_mtcnn_compiled(gin(in_orig), gout(in_orig, out_faces));
        tm.stop();
        std::cout << "Final Faces Size " << out_faces.size() << std::endl;
        std::vector<vis::rectPoints> data;
        // show the image with faces in it
        for (size_t i = 0; i < out_faces.size(); ++i) {
            std::vector<cv::Point> pts;
            for (int p = 0; p < NUM_PTS; ++p) {
                pts.push_back(
                    cv::Point(out_faces[i].ptsCoords[2 * p], out_faces[i].ptsCoords[2 * p + 1]));
            }

            auto rect = out_faces[i].bbox.getRect();
            auto d = std::make_pair(rect, pts);
            data.push_back(d);
        }
        auto resultImg = vis::drawRectsAndPoints(in_orig, data);
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        cv::putText(resultImg, fps_str, { 0,32 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
        cv::imshow("Out", resultImg);
        // remove comment out for video cap
        //cv::waitKey(1);
        out_faces.clear();
        tm.start();
    }

#else
    // Input stream
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name);

    // Set the pipeline source & start the pipeline
    pipeline_mtcnn.setSource(cv::gin(in_src));
    pipeline_mtcnn.start();

    // Declare the output data & run the processing loop
    cv::TickMeter tm;
    cv::Mat image;
    std::vector<custom::Face> out_faces;

    tm.start();
    int frames = 0;
    while (pipeline_mtcnn.pull(cv::gout(image, out_faces))) {
        frames++;
        std::cout << "Final Faces Size " << out_faces.size() << std::endl;
        std::vector<vis::rectPoints> data;
        // show the image with faces in it
        for (const auto& out_face : out_faces) {
            std::vector<cv::Point> pts;
            for (int p = 0; p < NUM_PTS; ++p) {
                pts.push_back(
                    cv::Point(static_cast<int>(out_face.ptsCoords[2 * p]), static_cast<int>(out_face.ptsCoords[2 * p + 1])));
            }
            auto rect = out_face.bbox.getRect();
            auto d = std::make_pair(rect, pts);
            data.push_back(d);
        }
        // Visualize results on the frame
        auto resultImg = vis::drawRectsAndPoints(image, data);
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        cv::putText(resultImg, fps_str, { 0,32 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
        cv::imshow("Out", resultImg);
        cv::waitKey(1);
        out_faces.clear();
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
        << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
#endif
    return 0;
}
