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
        const int IMAGE_WIDTH = 1920;
        const int IMAGE_HEIGHT = 1080;
        const int TRANSPOSED_IMAGE_WIDTH = 1080;
        const int TRANSPOSED_IMAGE_HEIGHT = 1920;

        std::vector<Face> buildFaces(const cv::Mat& scores,
            const cv::Mat& regressions,
            const float scaleFactor,
            const float threshold) {

            auto w = scores.size[3];
            auto h = scores.size[2];
            auto size = w * h;

            std::cout << "scores_data w "  << w <<  " scores_data h "  << h << std::endl;


            const float* scores_data = (float*)(scores.data);
            for(int i = 0; i < 200; i++)
            {
                std::cout << scores_data[i] << " ";
            }
            std::cout << std::endl;
            scores_data += size;

            //std::cout << "scores_data shifted by size  "  << size << std::endl;

            //for(int i = 0; i < 200; i++)
            //{
            //    std::cout << scores_data[i] << " ";
            //}

            const float* reg_data = (float*)(regressions.data);
            //auto wr = regressions.size[3];
            //auto hr = regressions.size[2];
            //std::cout << "regressions_data w "  << wr <<  " regressions_data h "  << hr << std::endl;
            //for(int i = 0; i < 200; i++)
            //{
            //    std::cout << reg_data[i] << " ";
            //}
            //std::cout << std::endl;


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

        G_API_NET(MTCNNProposal_1777x1000,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_1777x1000");

        G_API_NET(MTCNNProposal_1260x709,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_1260x709");

        G_API_NET(MTCNNProposal_893x502,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_893x502");

        G_API_NET(MTCNNProposal_633x356,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_633x356");

        G_API_NET(MTCNNProposal_449x252,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_449x252");

        G_API_NET(MTCNNProposal_318x179,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_318x179");

        G_API_NET(MTCNNProposal_225x127,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_225x127");

        G_API_NET(MTCNNProposal_160x90,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_160x90");

        G_API_NET(MTCNNProposal_113x63,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_113x63");

        G_API_NET(MTCNNProposal_80x45,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_80x45");

        G_API_NET(MTCNNProposal_57x32,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_57x32");

        G_API_NET(MTCNNProposal_40x22,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_40x22");

        G_API_NET(MTCNNProposal_28x16,
            <GMat2(cv::GMat)>,
            "sample.custom.mtcnn_proposal_28x16");

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

        G_API_OP(RunNMS,
            <GFaces(GFaces, float, bool)>,
            "sample.custom.mtcnn.run_nms") {
            static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                float, bool) {
                return cv::empty_array_desc();
            }
        };

        G_API_OP(MergePyramidOutputs,
            <GFaces(GFaces, GFaces, GFaces, GFaces, GFaces, GFaces,
                    GFaces, GFaces, GFaces, GFaces, GFaces, GFaces, GFaces)>,
            "sample.custom.mtcnn.merge_pyramid_outputs") {
            static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&,
                const cv::GArrayDesc&
            ) {
                return cv::empty_array_desc();
            }
        };

        G_API_OP(AccumulatePyramidOutputs,
            <GFaces(GFaces, GFaces)>,
            "sample.custom.mtcnn.merge_pyramid_outputs") {
            static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
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

        G_API_OP(Transpose,
            <cv::GMat(cv::GMat)>,
            "sample.custom.mtcnn.transpose") {
            static cv::GMatDesc outMeta(const cv::GMatDesc in
            ) {
                const cv::GMatDesc out_desc = { in.depth, in.chan, cv::Size(in.size.height,
                                                                     in.size.width) };
                return out_desc;
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
                std::cout << "OCVBuildFaces!!! faces number " << out_faces.size() <<
                             " scaleFactor " << scaleFactor <<
                             " threshold " << threshold << std::endl;
            }
        };// GAPI_OCV_KERNEL(BuildFaces)

        GAPI_OCV_KERNEL(OCVRunNMS, RunNMS) {
            static void run(const std::vector<Face> &in_faces,
                float threshold,
                bool useMin,
                std::vector<Face> &out_faces) {
                std::vector<Face> in_faces_copy = in_faces;
                out_faces = Face::runNMS(in_faces_copy, threshold, useMin);
                std::cout << "OCVRunNMS!!! in_faces size " << in_faces.size() <<
                    " out_faces size " << out_faces.size() <<
                    " for threshold " << threshold <<
                    " and useMin " << useMin << std::endl;
            }
        };// GAPI_OCV_KERNEL(RunNMS)


        GAPI_OCV_KERNEL(OCVMergePyramidOutputs, MergePyramidOutputs) {
            static void run(const std::vector<Face> &in_faces0,
                const std::vector<Face> &in_faces1,
                const std::vector<Face> &in_faces2,
                const std::vector<Face> &in_faces3,
                const std::vector<Face> &in_faces4,
                const std::vector<Face> &in_faces5,
                const std::vector<Face> &in_faces6,
                const std::vector<Face> &in_faces7,
                const std::vector<Face> &in_faces8,
                const std::vector<Face> &in_faces9,
                const std::vector<Face> &in_faces10,
                const std::vector<Face> &in_faces11,
                const std::vector<Face> &in_faces12,
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
                if (!in_faces6.empty()) {
                    out_faces.insert(out_faces.end(), in_faces6.begin(), in_faces6.end());
                }
                if (!in_faces7.empty()) {
                    out_faces.insert(out_faces.end(), in_faces7.begin(), in_faces7.end());
                }
                if (!in_faces8.empty()) {
                    out_faces.insert(out_faces.end(), in_faces8.begin(), in_faces8.end());
                }
                if (!in_faces9.empty()) {
                    out_faces.insert(out_faces.end(), in_faces9.begin(), in_faces9.end());
                }
                if (!in_faces10.empty()) {
                    out_faces.insert(out_faces.end(), in_faces10.begin(), in_faces10.end());
                }
                if (!in_faces11.empty()) {
                    out_faces.insert(out_faces.end(), in_faces11.begin(), in_faces11.end());
                }
                if (!in_faces12.empty()) {
                    out_faces.insert(out_faces.end(), in_faces12.begin(), in_faces12.end());
                }
                std::cout << "OCVMergePyramidOutputs!!! output faces number " << out_faces.size() << std::endl;
            }
        };// GAPI_OCV_KERNEL(MergePyramidOutputs)

        GAPI_OCV_KERNEL(OCVAccumulatePyramidOutputs, AccumulatePyramidOutputs) {
            static void run(const std::vector<Face> &total_faces,
                const std::vector<Face> &in_faces,
                std::vector<Face> &out_faces) {
                out_faces = total_faces;
                if (!in_faces.empty()) {
                    out_faces.insert(out_faces.end(), in_faces.begin(), in_faces.end());
                }
                std::cout << "OCVAccumulatePyramidOutputs!!! output faces number " << out_faces.size() << std::endl;
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
                if (!in_faces_copy.empty()) {
                    out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
                }
                std::cout << "OCVBBoxesToSquares!!! input faces number " << in_faces.size() <<
                    " output faces number " << out_faces.size() << std::endl;
            }
        };// GAPI_OCV_KERNEL(BBoxesToSquares)

        GAPI_OCV_KERNEL(OCVR_O_NetPreProcGetROIs, R_O_NetPreProcGetROIs) {
            static void run(const std::vector<Face> &in_faces,
                std::vector<cv::Rect> &outs) {
                outs.clear();
                for (auto& f : in_faces) {
                    cv::Rect tmp_rect = f.bbox.getRect();
                    if (tmp_rect.x + tmp_rect.width >= TRANSPOSED_IMAGE_WIDTH) tmp_rect.width = TRANSPOSED_IMAGE_WIDTH - tmp_rect.x - 4;
                    if (tmp_rect.y + tmp_rect.height >= TRANSPOSED_IMAGE_HEIGHT) tmp_rect.height = TRANSPOSED_IMAGE_HEIGHT - tmp_rect.y - 4;
                    outs.push_back(tmp_rect);
                    //outs.push_back(f.bbox.getRect());
                }
                std::cout << "OCVR_O_NetPreProcGetROIs!!! input faces number " << in_faces.size() <<
                    " output faces number " << outs.size() << std::endl;
            }
        };// GAPI_OCV_KERNEL(R_O_NetPreProcGetROIs)


        GAPI_OCV_KERNEL(OCVRNetPostProc, RNetPostProc) {
            static void run(const std::vector<Face> &in_faces,
                const std::vector<cv::Mat> &in_scores,
                const std::vector<cv::Mat> &in_regresssions,
                float threshold,
                std::vector<Face> &out_faces) {
                out_faces.clear();
                std::cout << "OCVRNetPostProc!!! input scores number " << in_scores.size() <<
                    " input regressions number " << in_regresssions.size() <<
                    " input faces size " << in_faces.size() << std::endl;
                for (unsigned int k = 0; k < in_faces.size(); ++k) {
                    const float* scores_data = (float*)in_scores[k].data;
                    const float* reg_data = (float*)in_regresssions[k].data;
                    //std::cout << "OCVRNetPostProc!!! scores_data[0] " << scores_data[0] << " scores_data[1] " << scores_data[1] << std::endl;
                    //std::cout << "OCVRNetPostProc!!! reg_data[0] " << reg_data[0] << " reg_data[1] " << reg_data[1] <<
                    //    "reg_data[2] " << reg_data[2] << " reg_data[3] " << reg_data[3] << std::endl;
                    if (scores_data[1] >= threshold) {
                        std::cout << "OCVRNetPostProc!!! scores_data[0] " << scores_data[0] << " scores_data[1] " << scores_data[1] << std::endl;
                        std::cout << "OCVRNetPostProc!!! reg_data[0] " << reg_data[0] << " reg_data[1] " << reg_data[1] <<
                                     "reg_data[2] " << reg_data[2] << " reg_data[3] " << reg_data[3] << std::endl;
                        Face info = in_faces[k];
                        info.score = scores_data[1];
                        for (int i = 0; i < 4; ++i) {
                            info.regression[i] = reg_data[i];
                        }
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
                float threshold,
                std::vector<Face> &out_faces) {
                out_faces.clear();
                std::cout << "OCVONetPostProc!!! input scores number " << in_scores.size() <<
                    " input regressions number " << in_regresssions.size() <<
                    " input landmarks number " << in_landmarks.size() <<
                    " input faces size " << in_faces.size() << std::endl;
                for (unsigned int k = 0; k < in_faces.size(); ++k) {
                    const float* scores_data = (float*)in_scores[k].data;
                    const float* reg_data = (float*)in_regresssions[k].data;
                    const float* landmark_data = (float*)in_landmarks[k].data;
                    if (scores_data[1] >= threshold) {
                       std::cout << "OCVONetPostProc!!! scores_data[0] " << scores_data[0] << " scores_data[1] " << scores_data[1] << std::endl;
                       std::cout << "OCVONetPostProc!!! reg_data[0] " << reg_data[0] << " reg_data[1] " << reg_data[1] <<
                                     " reg_data[2] " << reg_data[2] << " reg_data[3] " << reg_data[3] << std::endl;

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
        void bbox(cv::Mat& m, const cv::Rect& rc) {
            cv::rectangle(m, rc, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
        };

    } // anonymous namespace
} // namespace vis

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat& img,
    const std::vector<rectPoints> data) {
    cv::Mat outImg;
    img.convertTo(outImg, CV_8UC3);

    for (auto& d : data) {
        //cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
        vis::bbox(outImg, d.first);
        std::cout << "drawRectsAndPoints!!!  " << d.first.x << " " << d.first.y << "  " << d.first.x +d.first.width << " " << d.first.y + d.first.height << std::endl;
        auto pts = d.second;
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
        }
    }
    return outImg;
}


static std::tuple<cv::GMat, cv::GMat> run_mtcnn_p(cv::GMat in, std::string id) {
    cv::GInferInputs inputs;
    inputs["data"] = in;
    //auto id = "net" + sz;
    std::cout << "run_mtcnn_p " << id << std::endl;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(id, inputs);
    auto regressions = outputs.at("conv4-2");
    auto scores = outputs.at("prob1");
    return std::make_tuple(regressions, scores);
}

const int PYRAMID_LEVELS = 13;

int main(int argc, char* argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const auto input_file_name = cmd.get<std::string>("input");
    const auto tmcnnp_model_path = cmd.get<std::string>("mtcnnpm");
    const auto tmcnnp_target_dev = cmd.get<std::string>("mtcnnpd");
    const auto tmcnnp_conf_thresh = cmd.get<double>("thrp");
    const auto tmcnnr_model_path = cmd.get<std::string>("mtcnnrm");
    const auto tmcnnr_target_dev = cmd.get<std::string>("mtcnnrd");
    const auto tmcnnr_conf_thresh = cmd.get<double>("thrr");
    const auto tmcnno_model_path = cmd.get<std::string>("mtcnnom");
    const auto tmcnno_target_dev = cmd.get<std::string>("mtcnnod");
    const auto tmcnno_conf_thresh = cmd.get<double>("thro");


    cv::GMat in_originalBGR;
    cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_originalBGR);
    //Proposal part of MTCNN graph
    //Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
    cv::Size level_size[PYRAMID_LEVELS] =
    {
        {1777, 1000},  {1260, 709}, {893, 502}, {633, 356},
        {449, 252}, {318, 179}, {225, 127}, {160, 90},
        {113, 63}, {80, 45}, {57, 32}, {40, 22}, {28, 16}
    };
    float scales[PYRAMID_LEVELS]
    {
        0.9259259259259259f, 0.6564814814814814f, 0.4654453703703703f,
        0.3300007675925925f, 0.23397054422314809f, 0.165885115854212f,
        0.1176125471406363f, 0.08338729592271113f, 0.059121592809202185f,
        0.041917209301724344f, 0.029719301394922563f, 0.021070984689000097f,
        0.014939328144501067f
    };

    cv::GMat in_resized[PYRAMID_LEVELS];
    cv::GMat in_transposed[PYRAMID_LEVELS];
    cv::GMat regressions[PYRAMID_LEVELS];
    cv::GMat scores[PYRAMID_LEVELS];
    //cv::GArray<custom::Face> faces[PYRAMID_LEVELS];
    cv::GArray<custom::Face> nms_p_faces[PYRAMID_LEVELS];
    cv::GArray<custom::Face> total_faces;
    in_resized[0] = cv::gapi::resize(in_originalRGB, level_size[0]);
    in_transposed[0] = custom::Transpose::on(in_resized[0]);
    std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposed[0], "MTCNNProposal_" + std::to_string(level_size[0].width) + "x" + std::to_string(level_size[0].height));
    cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[0], regressions[0], scales[0], tmcnnp_conf_thresh);
    nms_p_faces[0] = custom::RunNMS::on(faces, 0.5f, false);
    for (int i = 1; i < PYRAMID_LEVELS; ++i)
    {
        in_resized[i] = cv::gapi::resize(in_originalRGB, level_size[i]);
        in_transposed[i] = custom::Transpose::on(in_resized[i]);
        std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposed[i], "MTCNNProposal_" + std::to_string(level_size[i].width) + "x" + std::to_string(level_size[i].height));
        cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i], regressions[i], scales[i], tmcnnp_conf_thresh);
        nms_p_faces[i] = custom::RunNMS::on(faces, 0.5f, false);
        total_faces = custom::AccumulatePyramidOutputs::on(nms_p_faces[i-1], nms_p_faces[i]);
    }
    //cv::GArray<custom::Face> nms_p_faces_total = custom::MergePyramidOutputs::on(nms_p_faces[0],
    //    nms_p_faces[1],
    //    nms_p_faces[2],
    //    nms_p_faces[3],
    //    nms_p_faces[4],
    //    nms_p_faces[5],
    //    nms_p_faces[6],
    //    nms_p_faces[7],
    //    nms_p_faces[8],
    //    nms_p_faces[9],
    //    nms_p_faces[10],
    //    nms_p_faces[11],
    //    nms_p_faces[12]);
    //Proposal post-processing
    //cv::GArray<custom::Face> nms07_p_faces_total = custom::RunNMS::on(nms_p_faces_total, 0.7f, false);
    cv::GArray<custom::Face> nms07_p_faces_total = custom::RunNMS::on(total_faces, 0.7f, false);
    cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(nms07_p_faces_total, false);
    cv::GArray<custom::Face> final_faces_pnet = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);

    //Refinement part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet);
    cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
    cv::GMat in_originalRGB_transposed = custom::Transpose::on(in_originalRGB);
    std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<custom::MTCNNRefinement>(faces_roi_pnet, in_originalRGB_transposed);
    //std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<custom::MTCNNRefinement>(faces_roi_pnet, in_originalRGB);

    //Refinement post-processing
    cv::GArray<custom::Face> rnet_post_proc_faces = custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, tmcnnr_conf_thresh);
    cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
    cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
    cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

    //Output part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet);
    cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
    std::tie(regressionsONet, landmarksONet, scoresONet) = cv::gapi::infer<custom::MTCNNOutput>(faces_roi_rnet, in_originalRGB_transposed);

    //Output post-processing
    cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet, scoresONet, regressionsONet, landmarksONet, tmcnno_conf_thresh);
    cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
    cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
    cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

    cv::GComputation graph_mtcnn(cv::GIn(in_originalBGR), cv::GOut(cv::gapi::copy(in_originalBGR), final_faces_onet));
    //cv::GComputation graph_mtcnn(cv::GIn(in_originalBGR), cv::GOut(cv::gapi::copy(in_originalBGR), final_faces_rnet));
    //cv::GComputation graph_mtcnn(cv::GIn(in_originalBGR), cv::GOut(cv::gapi::copy(in_originalBGR), final_faces_pnet));

    // MTCNN Proposal detection network
    std::vector<cv::gapi::ie::Params<cv::gapi::Generic>> mtcnnp_net;
    for (int i = 0; i < PYRAMID_LEVELS; ++i)
    {
        std::string net_id = "MTCNNProposal_" + std::to_string(level_size[i].width) + "x" + std::to_string(level_size[i].height);
        std::cout << "mtcnnp_net " << net_id << std::endl;
        std::vector<size_t> reshape_dims = { 1, 3, (size_t)level_size[i].width, (size_t)level_size[i].height };
        mtcnnp_net.push_back(cv::gapi::ie::Params<cv::gapi::Generic>{
            net_id,                           // tag
            tmcnnp_model_path,                // path to topology IR
            weights_path(tmcnnp_model_path),  // path to weights
            tmcnnp_target_dev,                // device specifier
        }.cfgInputReshape({ {"data", reshape_dims} }));
    }
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

    auto networks_mtcnn = cv::gapi::networks(
        mtcnnp_net[0]
        , mtcnnp_net[1]
        , mtcnnp_net[2]
        , mtcnnp_net[3]
        , mtcnnp_net[4]
        , mtcnnp_net[5]
        , mtcnnp_net[6]
        , mtcnnp_net[7]
        , mtcnnp_net[8]
        , mtcnnp_net[9]
        , mtcnnp_net[10]
        , mtcnnp_net[11]
        , mtcnnp_net[12]
        , mtcnnr_net
        , mtcnno_net);

    auto kernels_mtcnn = cv::gapi::kernels< custom::OCVBuildFaces
        , custom::OCVRunNMS
        , custom::OCVMergePyramidOutputs
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
    auto in_original = cv::imread(input_file_name);
    cv::Mat in_src;
    if (in_original.cols != custom::IMAGE_WIDTH || in_original.rows != custom::IMAGE_HEIGHT)
    {
       cv::resize(in_original, in_src, cv::Size(custom::IMAGE_WIDTH, custom::IMAGE_HEIGHT));
    }
    else
    {
        in_original.copyTo(in_src);
    }

    cv::Mat image;
    std::vector<custom::Face> out_faces;
    auto graph_mtcnn_compiled = graph_mtcnn.compile(descr_of(gin(in_src)), cv::compile_args(networks_mtcnn, kernels_mtcnn));
    graph_mtcnn_compiled(gin(in_src), gout(image, out_faces));
    std::cout << "Final Faces Size " << out_faces.size() << std::endl;

    std::vector<rectPoints> data;
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
    auto resultImg = drawRectsAndPoints(image, data);
    cv::imshow("Out", resultImg);
    //for (auto&& rc : out_faces) vis::bbox(image, rc.bbox.getRect());
    //cv::imshow("Out", image);
    cv::waitKey(-1);
#else
    // Input stream
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name);

    // Text recognition input size (also an input parameter to the graph)
    auto in_rsz = cv::Size{ custom::IMAGE_WIDTH, custom::IMAGE_HEIGHT };

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
        std::vector<rectPoints> data;
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
        // Visualize results on the frame
        //for (auto&& rc : out_faces) vis::bbox(image, rc.bbox.getRect());
        auto resultImg = drawRectsAndPoints(image, data);
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        //cv::putText(image, fps_str, { 0,32 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
        //cv::imshow("Out", image);
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
