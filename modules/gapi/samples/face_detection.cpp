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
#include <opencv2/core/utility.hpp>

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
    ;

namespace {
std::string weights_path(const std::string &model_path) {
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
    float x1;
    float y1;
    float x2;
    float y2;

    cv::Rect getRect() const { return cv::Rect(x1, y1, x2 - x1, y2 - y1); }

    BBox getSquare() const {
        BBox bbox;
        float bboxWidth = x2 - x1;
        float bboxHeight = y2 - y1;
        float side = std::max(bboxWidth, bboxHeight);
        bbox.x1 = static_cast<int>(x1 + (bboxWidth - side) * 0.5f);
        bbox.y1 = static_cast<int>(y1 + (bboxHeight - side) * 0.5f);
        bbox.x2 = static_cast<int>(bbox.x1 + side);
        bbox.y2 = static_cast<int>(bbox.y1 + side);
        return bbox;
    }
};

struct Face {
    BBox bbox;
    float score;
    float regression[NUM_REGRESSIONS];
    float ptsCoords[2 * NUM_PTS];

    static void applyRegression(std::vector<Face>& faces, bool addOne = false) {
        for (size_t i = 0; i < faces.size(); ++i) {
            float bboxWidth =
                faces[i].bbox.x2 - faces[i].bbox.x1 + static_cast<float>(addOne);
            float bboxHeight =
                faces[i].bbox.y2 - faces[i].bbox.y1 + static_cast<float>(addOne);
            faces[i].bbox.x1 = faces[i].bbox.x1 + faces[i].regression[1] * bboxWidth;
            faces[i].bbox.y1 = faces[i].bbox.y1 + faces[i].regression[0] * bboxHeight;
            faces[i].bbox.x2 = faces[i].bbox.x2 + faces[i].regression[3] * bboxWidth;
            faces[i].bbox.y2 = faces[i].bbox.y2 + faces[i].regression[2] * bboxHeight;
        }
    }

    static void bboxes2Squares(std::vector<Face>& faces) {
        for (size_t i = 0; i < faces.size(); ++i) {
            faces[i].bbox = faces[i].bbox.getSquare();
        }
    }

    static std::vector<Face> runNMS(std::vector<Face>& faces, float threshold,
        bool useMin = false) {
        std::vector<Face> facesNMS;
        if (faces.empty()) {
            return facesNMS;
        }

        std::sort(faces.begin(), faces.end(), [](const Face& f1, const Face& f2) {
            return f1.score > f2.score;
            });

        std::vector<int> indices(faces.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        while (indices.size() > 0) {
            int idx = indices[0];
            facesNMS.push_back(faces[idx]);
            std::vector<int> tmpIndices = indices;
            indices.clear();
            for (size_t i = 1; i < tmpIndices.size(); ++i) {
                int tmpIdx = tmpIndices[i];
                float interX1 = std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1);
                float interY1 = std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1);
                float interX2 = std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2);
                float interY2 = std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2);

                float bboxWidth = std::max(0.f, (interX2 - interX1 + 1));
                float bboxHeight = std::max(0.f, (interY2 - interY1 + 1));

                float interArea = bboxWidth * bboxHeight;
                // TODO: compute outside the loop
                float area1 = (faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) *
                    (faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
                float area2 = (faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) *
                    (faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
                float o = 0.f;
                if (useMin) {
                    o = interArea / std::min(area1, area2);
                }
                else {
                    o = interArea / (area1 + area2 - interArea);
                }
                if (o <= threshold) {
                    indices.push_back(tmpIdx);
                }
            }
        }
        return facesNMS;
    }
};

const float P_NET_WINDOW_SIZE = 12.f;
const int P_NET_STRIDE = 2;

std::vector<Face> buildFaces(const cv::Mat& scores,
    const cv::Mat& regressions,
    const float scaleFactor,
    const float threshold) {

    auto w = scores.size[3];
    auto h = scores.size[2];
    auto size = w * h;

    const float* scores_data = (float*)(scores.data);
    scores_data += size;

    const float* reg_data = (float*)(regressions.data);

    std::vector<Face> boxes;

    for (int i = 0; i < size; i++) {
        if (scores_data[i] >= (threshold)) {
            int y = i / w;
            int x = i - w * y;

            Face faceInfo;
            BBox& faceBox = faceInfo.bbox;

            faceBox.x1 = (float)(x * P_NET_STRIDE) / scaleFactor;
            faceBox.y1 = (float)(y * P_NET_STRIDE) / scaleFactor;
            faceBox.x2 =
                (float)(x * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
            faceBox.y2 =
                (float)(y * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
            faceInfo.regression[0] = reg_data[i];
            faceInfo.regression[1] = reg_data[i + size];
            faceInfo.regression[2] = reg_data[i + 2 * size];
            faceInfo.regression[3] = reg_data[i + 3 * size];
            faceInfo.score = scores_data[i];
            boxes.push_back(faceInfo);
        }
    }

    return boxes;
}

// Define networks for this sample
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
using GMats = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;

G_API_NET(MTCNNProposal_960x540,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_960x540");

G_API_NET(MTCNNProposal_480x270,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_480x270");

G_API_NET(MTCNNProposal_240x135,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_240x135");

G_API_NET(MTCNNProposal_120x67,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_120x67");

G_API_NET(MTCNNProposal_60x33,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_60x33");

G_API_NET(MTCNNProposal_30x16,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_proposal_30x16");


G_API_NET(MTCNNRefinement,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_refinement");


G_API_NET(MTCNNOutput,
          <GMat3(cv::GMat)>,
          "sample.custom.mtcnn_output");

using GFaces = cv::GArray<Face>;
G_API_OP(BuildFaces,
         <GFaces(cv::GMat, cv::GMat, float, float)>,
          "sample.custom.mtcnn.build_faces") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                  const cv::GMatDesc&,
                                  float,
                                  float) {
          return cv::empty_array_desc();
    }
};

G_API_OP(RunNMSAccum,
         <GFaces(GFaces, float)>,
          "sample.custom.mtcnn.run_nms_accum") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                  float) {
        return cv::empty_array_desc();
    }
};

G_API_OP(RunNMS,
    <GFaces(GFaces, float, bool)>,
    "sample.custom.mtcnn.run_nms") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
        float, bool) {
        return cv::empty_array_desc();
    }
};

G_API_OP(MergePyramidOutputs,
    <GFaces(GFaces, GFaces, GFaces, GFaces, GFaces, GFaces)>,
    "sample.custom.mtcnn.merge_pyramid_outputs") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                  const cv::GArrayDesc&,
                                  const cv::GArrayDesc&,
                                  const cv::GArrayDesc&,
                                  const cv::GArrayDesc&,
                                  const cv::GArrayDesc&
        ) {
        return cv::empty_array_desc();
    }
};

G_API_OP(ApplyRegression,
    <GFaces(GFaces, bool)>,
    "sample.custom.mtcnn.apply_regression") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
        bool) {
        return cv::empty_array_desc();
    }
};

G_API_OP(BBoxesToSquares,
    <GFaces(GFaces)>,
    "sample.custom.mtcnn.bboxes_to_squares") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&
    ) {
        return cv::empty_array_desc();
    }
};

G_API_OP(R_O_NetPreProcGetROIs,
    <GRects(GFaces)>,
    "sample.custom.mtcnn.bboxes_r_o_net_preproc_get_rois") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&
    ) {
        return cv::empty_array_desc();
    }
};


G_API_OP(RNetPostProc,
    <GFaces(GFaces, GMats, GMats, float)>,
    "sample.custom.mtcnn.rnet_postproc") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        float
    ) {
        return cv::empty_array_desc();
    }
};

G_API_OP(ONetPostProc,
    <GFaces(GFaces, GMats, GMats, GMats, float)>,
    "sample.custom.mtcnn.onet_postproc") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        float
    ) {
        return cv::empty_array_desc();
    }
};

G_API_OP(SwapFaces,
    <GFaces(GFaces)>,
    "sample.custom.mtcnn.swap_faces") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&
    ) {
        return cv::empty_array_desc();
    }
};

//Custom kernels implementation

GAPI_OCV_KERNEL(OCVBuildFaces, BuildFaces) {
    static void run(const cv::Mat & in_scores,
                    const cv::Mat & in_regresssions,
                    float scaleFactor,
                    float threshold,
                    std::vector<Face> &out_faces) {
        out_faces = buildFaces(in_scores, in_regresssions, scaleFactor, threshold);
        std::cout << "OCVBuildFaces!!! faces number " << out_faces.size() << std::endl;
    }
};// GAPI_OCV_KERNEL(BuildFaces)


GAPI_OCV_KERNEL(OCVRunNMSAccum, RunNMSAccum) {
    static void run(const std::vector<Face> &in_faces,
                    float threshold,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        std::vector<Face> nms_faces = Face::runNMS(in_faces_copy, threshold);
        if (!nms_faces.empty()) {
            out_faces.insert(out_faces.end(), nms_faces.begin(), nms_faces.end());
        }
    }
};// GAPI_OCV_KERNEL(RunNMSAccum)

GAPI_OCV_KERNEL(OCVRunNMS, RunNMS) {
    static void run(const std::vector<Face> &in_faces,
        float threshold,
        bool useMin,
        std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        out_faces = Face::runNMS(in_faces_copy, threshold, useMin);
    }
};// GAPI_OCV_KERNEL(RunNMS)


GAPI_OCV_KERNEL(OCVMergePyramidOutputs, MergePyramidOutputs) {
    static void run(const std::vector<Face> &in_faces0,
                    const std::vector<Face> &in_faces1,
                    const std::vector<Face> &in_faces2,
                    const std::vector<Face> &in_faces3,
                    const std::vector<Face> &in_faces4,
                    const std::vector<Face> &in_faces5,
                    std::vector<Face> &out_faces) {
        if (!in_faces0.empty()) {
            out_faces.insert(out_faces.end(), in_faces0.begin(), in_faces0.end());
        }
        if (!in_faces1.empty()) {
            out_faces.insert(out_faces.end(), in_faces1.begin(), in_faces1.end());
        }
        if (!in_faces2.empty()) {
            out_faces.insert(out_faces.end(), in_faces2.begin(), in_faces2.end());
        }
        if (!in_faces3.empty()) {
            out_faces.insert(out_faces.end(), in_faces3.begin(), in_faces3.end());
        }
        if (!in_faces4.empty()) {
            out_faces.insert(out_faces.end(), in_faces4.begin(), in_faces4.end());
        }
        if (!in_faces5.empty()) {
            out_faces.insert(out_faces.end(), in_faces5.begin(), in_faces5.end());
        }
    }
};// GAPI_OCV_KERNEL(MergePyramidOutputs)

GAPI_OCV_KERNEL(OCVApplyRegression, ApplyRegression) {
    static void run(const std::vector<Face> &in_faces,
        bool addOne,
        std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::applyRegression(in_faces_copy, addOne);
        out_faces.clear();
        if (!in_faces_copy.empty()) {
            out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
        }
    }
};// GAPI_OCV_KERNEL(ApplyRegression)


GAPI_OCV_KERNEL(OCVBBoxesToSquares, BBoxesToSquares) {
    static void run(const std::vector<Face> &in_faces,
        std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::bboxes2Squares(in_faces_copy);
        out_faces.clear();
        if (!in_faces_copy.empty()) {
            out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
        }
    }
};// GAPI_OCV_KERNEL(BBoxesToSquares)


GAPI_OCV_KERNEL(OCVR_O_NetPreProcGetROIs, R_O_NetPreProcGetROIs) {
    static void run(const std::vector<Face> &in_faces,
        std::vector<cv::Rect> &outs) {
        outs.clear();
        for (auto& f : in_faces) {
            cv::Rect tmp_rect = f.bbox.getRect();
            if (tmp_rect.x + tmp_rect.width >= 1920) tmp_rect.width = 1920 - tmp_rect.x - 4;
            if (tmp_rect.y + tmp_rect.height >= 1080) tmp_rect.height = 1080 - tmp_rect.y - 4;
            outs.push_back(tmp_rect);
            //outs.push_back(f.bbox.getRect());
        }
        std::cout << "OCVR_O_NetPreProcGetROIs!!! input faces number " << in_faces.size() <<  " output faces number " << outs.size() << std::endl;
    }
};// GAPI_OCV_KERNEL(R_O_NetPreProcGetROIs)


GAPI_OCV_KERNEL(OCVRNetPostProc, RNetPostProc) {
    static void run(const std::vector<Face> &in_faces,
        const std::vector<cv::Mat> &in_scores,
        const std::vector<cv::Mat> & in_regresssions,
        float threshold,
        std::vector<Face> &out_faces) {
        out_faces.clear();
        std::cout << "OCVRNetPostProc!!! input scores number " << in_scores.size() << " input regressions number " << in_regresssions.size() << " input faces size " << in_faces.size() << std::endl;
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = (float*)in_scores[k].data;
            const float* reg_data = (float*)in_regresssions[k].data;
            if (scores_data[1] >= threshold) {
                Face info = in_faces[k];
                info.score = scores_data[1];
                for (int i = 0; i < 4; ++i) {
                    info.regression[i] = reg_data[i];
                }
                out_faces.push_back(info);
            }
        }
        std::cout << "OCVRNetPostProc!!! out faces number " << out_faces.size() << " for threshold " << threshold << std::endl;
    }
};// GAPI_OCV_KERNEL(RNetPostProc)

GAPI_OCV_KERNEL(OCVONetPostProc, ONetPostProc) {
    static void run(const std::vector<Face> &in_faces,
        const std::vector<cv::Mat> &in_scores,
        const std::vector<cv::Mat> &in_regresssions,
        const std::vector<cv::Mat> &in_landmarks,
        float threshold,
        std::vector<Face> &out_faces) {
        out_faces.clear();
        std::cout << "OCVONetPostProc!!! input scores number " << in_scores.size() <<
                     " input regressions number " <<  in_regresssions.size() <<
                     " input landmarks number " << in_landmarks.size() <<
                     " input faces size " << in_faces.size() << std::endl;
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = (float*)in_scores[k].data;
            const float* reg_data = (float*)in_regresssions[k].data;
            const float* landmark_data = (float*)in_landmarks[k].data;
            if (scores_data[1] >= threshold) {
                Face info = in_faces[k];
                info.score = scores_data[1];
                for (int i = 0; i < 4; ++i) {
                    info.regression[i] = reg_data[i];
                }
                float w = info.bbox.x2 - info.bbox.x1 + 1.f;
                float h = info.bbox.y2 - info.bbox.y1 + 1.f;

                for (int p = 0; p < NUM_PTS; ++p) {
                    info.ptsCoords[2 * p] =
                        info.bbox.x1 + landmark_data[NUM_PTS + p] * w - 1;
                    info.ptsCoords[2 * p + 1] = info.bbox.y1 + landmark_data[p] * h - 1;
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


} // anonymous namespace
} // namespace custom

namespace vis {
namespace {
void bbox(cv::Mat& m, const cv::Rect& rc) {
    cv::rectangle(m, rc, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
};

} // anonymous namespace
} // namespace vis

const float P_NET_WINDOW_SIZE = 12.f;
int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const auto input_file_name = cmd.get<std::string>("input");
    const auto tmcnnp_model_path  = cmd.get<std::string>("mtcnnpm");
    const auto tmcnnp_target_dev = cmd.get<std::string>("mtcnnpd");
    const auto tmcnnp_conf_thresh = cmd.get<double>("thrp");
    const auto tmcnnr_model_path = cmd.get<std::string>("mtcnnrm");
    const auto tmcnnr_target_dev = cmd.get<std::string>("mtcnnrd");
    const auto tmcnnr_conf_thresh = cmd.get<double>("thrr");
    const auto tmcnno_model_path = cmd.get<std::string>("mtcnnom");
    const auto tmcnno_target_dev = cmd.get<std::string>("mtcnnod");
    const auto tmcnno_conf_thresh = cmd.get<double>("thro");


    //Proposal part of MTCNN graph
    //TODO: replace with generic infer PNet
    //960x540
    cv::GMat in_original;
    cv::GMat in0 = cv::gapi::resize(in_original, cv::Size(960, 540));
    cv::GMat regressions0, scores0;
    std::tie(regressions0, scores0) = cv::gapi::infer<custom::MTCNNProposal_960x540>(in0);
    float currentScale = 0.5f;
    cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores0, regressions0, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces0 = custom::RunNMS::on(faces0, 0.5f, false);
    //480x270
    cv::GMat in1 = cv::gapi::resize(in0, cv::Size(480, 270));
    cv::GMat regressions1, scores1;
    std::tie(regressions1, scores1) = cv::gapi::infer<custom::MTCNNProposal_480x270>(in1);
    currentScale = currentScale / 2.0f;
    cv::GArray<custom::Face> faces1 = custom::BuildFaces::on(scores1, regressions1, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces1 = custom::RunNMS::on(faces1, 0.5f, false);
    //240x135
    cv::GMat in2 = cv::gapi::resize(in1, cv::Size(240, 135));
    cv::GMat regressions2, scores2;
    std::tie(regressions2, scores2) = cv::gapi::infer<custom::MTCNNProposal_240x135>(in2);
    currentScale = currentScale / 2.0f;
    cv::GArray<custom::Face> faces2 = custom::BuildFaces::on(scores2, regressions2, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces2 = custom::RunNMS::on(faces2, 0.5f, false);
    //120x67
    cv::GMat in3 = cv::gapi::resize(in2, cv::Size(120, 67));
    cv::GMat regressions3, scores3;
    std::tie(regressions3, scores3) = cv::gapi::infer<custom::MTCNNProposal_120x67>(in3);
    currentScale = currentScale / 2.0f;
    cv::GArray<custom::Face> faces3 = custom::BuildFaces::on(scores3, regressions3, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces3 = custom::RunNMS::on(faces3, 0.5f, false);
    //60x34
    cv::GMat in4 = cv::gapi::resize(in3, cv::Size(60, 33));
    cv::GMat regressions4, scores4;
    std::tie(regressions4, scores4) = cv::gapi::infer<custom::MTCNNProposal_60x33>(in4);
    currentScale = currentScale / 2.0f;
    cv::GArray<custom::Face> faces4 = custom::BuildFaces::on(scores4, regressions4, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces4 = custom::RunNMS::on(faces4, 0.5f, false);
    //30x17
    cv::GMat in5 = cv::gapi::resize(in4, cv::Size(30, 16));
    cv::GMat regressions5, scores5;
    std::tie(regressions5, scores5) = cv::gapi::infer<custom::MTCNNProposal_30x16>(in5);
    currentScale = currentScale / 2.0f;
    cv::GArray<custom::Face> faces5 = custom::BuildFaces::on(scores5, regressions5, currentScale, tmcnnp_conf_thresh);
    cv::GArray<custom::Face> nms_p_faces5 = custom::RunNMS::on(faces5, 0.5f, false);
    cv::GArray<custom::Face> nms_p_faces_total = custom::MergePyramidOutputs::on(nms_p_faces0, nms_p_faces1, nms_p_faces2, nms_p_faces3, nms_p_faces4, nms_p_faces5);
    //Proposal post-processing
    cv::GArray<custom::Face> nms07_p_faces_total = custom::RunNMS::on(nms_p_faces_total, 0.7f, false);
    cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(nms07_p_faces_total, false);
    cv::GArray<custom::Face> final_faces_pnet = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);

    //Refinement part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet);
    cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
    std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<custom::MTCNNRefinement>(faces_roi_pnet, in_original);

    //Refinement post-processing
    cv::GArray<custom::Face> rnet_post_proc_faces = custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, tmcnnr_conf_thresh);
    cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
    cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
    cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

    //Output part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet);
    cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
    std::tie(regressionsONet, landmarksONet, scoresONet) = cv::gapi::infer<custom::MTCNNOutput>(faces_roi_rnet, in_original);

    //Output post-processing
    cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet, scoresONet, regressionsONet, landmarksONet, tmcnno_conf_thresh);
    cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
    cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
    cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

    cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), final_faces_onet));
    //cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), final_faces_rnet));
    //cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), final_faces_pnet));


    // MTCNN Proposal detection network
    std::vector<size_t> reshape_dims_960_530 = {1, 3, 540, 960};
    auto mtcnnp_net_960x540 = cv::gapi::ie::Params<custom::MTCNNProposal_960x540>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_960_530} });

    std::vector<size_t> reshape_dims_480x270 = { 1, 3, 270, 480 };
    auto mtcnnp_net_480x270 = cv::gapi::ie::Params<custom::MTCNNProposal_480x270>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_480x270} });

    std::vector<size_t> reshape_dims_240x135 = { 1, 3, 135, 240 };
    auto mtcnnp_net_240x135 = cv::gapi::ie::Params<custom::MTCNNProposal_240x135>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_240x135} });

    std::vector<size_t> reshape_dims_120x67 = { 1, 3, 67, 120 };
    auto mtcnnp_net_120x67 = cv::gapi::ie::Params<custom::MTCNNProposal_120x67>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_120x67} });

    std::vector<size_t> reshape_dims_60x33 = { 1, 3, 33, 60 };
    auto mtcnnp_net_60x33 = cv::gapi::ie::Params<custom::MTCNNProposal_60x33>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_60x33} });

    std::vector<size_t> reshape_dims_30x16 = { 1, 3, 16, 30 };
    auto mtcnnp_net_30x16 = cv::gapi::ie::Params<custom::MTCNNProposal_30x16>{
        tmcnnp_model_path,                // path to topology IR
        weights_path(tmcnnp_model_path),  // path to weights
        tmcnnp_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv4-2", "prob1" }).cfgInputReshape({ {"data", reshape_dims_30x16} });


    // MTCNN Refinement detection network
    std::vector<size_t> reshape_dims_24x24 = { 1, 3, 24, 24 };
    auto mtcnnr_net = cv::gapi::ie::Params<custom::MTCNNRefinement>{
        tmcnnr_model_path,                // path to topology IR
        weights_path(tmcnnr_model_path),  // path to weights
        tmcnnr_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv5-2", "prob1" }).cfgInputLayers({ "data" });

    // MTCNN Output detection network
    std::vector<size_t> reshape_dims_48x48 = { 1, 3, 48, 48 };
    auto mtcnno_net = cv::gapi::ie::Params<custom::MTCNNOutput>{
        tmcnno_model_path,                // path to topology IR
        weights_path(tmcnno_model_path),  // path to weights
        tmcnno_target_dev,                // device specifier
    }.cfgOutputLayers({ "conv6-2", "conv6-3", "prob1" }).cfgInputLayers({ "data" });

    auto networks_mtcnn = cv::gapi::networks(mtcnnp_net_960x540
                                             , mtcnnp_net_480x270
                                             , mtcnnp_net_240x135
                                             , mtcnnp_net_120x67
                                             , mtcnnp_net_60x33
                                             , mtcnnp_net_30x16
                                             , mtcnnr_net
                                             , mtcnno_net);

    auto kernels_mtcnn = cv::gapi::kernels< custom::OCVBuildFaces
        , custom::OCVRunNMS
        , custom::OCVRunNMSAccum
        , custom::OCVMergePyramidOutputs
        , custom::OCVApplyRegression
        , custom::OCVBBoxesToSquares
        , custom::OCVR_O_NetPreProcGetROIs
        , custom::OCVRNetPostProc
        , custom::OCVONetPostProc
        , custom::OCVSwapFaces
    >();
    auto pipeline_mtcnn = graph_mtcnn.compileStreaming(cv::compile_args(networks_mtcnn, kernels_mtcnn));


    std::cout << "Reading " << input_file_name << std::endl;

    // Input stream
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name);

    // Text recognition input size (also an input parameter to the graph)
    auto in_rsz = cv::Size{ 1920, 1080 };

    // Set the pipeline source & start the pipeline
    pipeline_mtcnn.setSource(cv::gin(in_src, in_rsz));
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
        // Visualize results on the frame
        for (auto&& rc : out_faces) vis::bbox(image, rc.bbox.getRect());
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        cv::putText(image, fps_str, { 0,32 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
        cv::imshow("Out", image);
        cv::waitKey(1);
        out_faces.clear();
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
