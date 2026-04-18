
#include "p2v_model.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"

#include <Eigen/src/Core/Matrix.h>
#include <memory>
#include <omp.h>

using namespace std;

extern int g_cpu_number;


P2V_Encoder_ONNX::P2V_Encoder_ONNX(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "p2v_header"),
      session_(nullptr),
      meminfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions so;
    // if your CPU has enough core, this intra-op can be 2; otherwise, just 1. (Because we also need cpu for decoder inference)
    const int intra_num_cpu = (g_cpu_number >= 8) ? 2 : 1;
    const int inter_num_cpu = 1;
    cout << "Intra-op number for VE-Net is: " << intra_num_cpu << endl;
    so.SetIntraOpNumThreads(intra_num_cpu);
    so.SetInterOpNumThreads(inter_num_cpu);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, onnx_path.c_str(), so);
}


std::vector<float> P2V_Encoder_ONNX::infer(const float* pts, const uint8_t* mask, int K){

    const int64_t pts_shape[3]  = {1, K, 3};
    const int64_t mask_shape[2] = {1, K};

    // pts tensor
    Ort::Value ort_pts = Ort::Value::CreateTensor<float>(
        meminfo_,
        const_cast<float*>(pts),
        1 * K * 3,
        pts_shape,
        3
    );

    std::unique_ptr<bool[]> mask_bool(new bool[K]);
    for (int i = 0; i < K; ++i)
        mask_bool[i] = (mask[i] != 0);

    Ort::Value ort_mask = Ort::Value::CreateTensor<bool>(
        meminfo_,
        mask_bool.get(),
        K,
        mask_shape,
        2
    );

    std::array<Ort::Value, 2> inputs = {
        std::move(ort_pts),
        std::move(ort_mask)
    };

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_,
        inputs.data(),
        2,
        output_names_,
        1
    );

    float* out = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(out, out + 64);
}



P2V_Decoder_ONNX::P2V_Decoder_ONNX(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "p2v_decoder"),
      session_(nullptr),
      meminfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions so;
    // if your CPU has enough core, this intra-op can be large. Reserve 1 core for main thread, and 1-2 core(s) for encoder.
    // Maybe you need to set different intra-op for VE-Net (encoder) and IR-Net (decoder) to get the best efficiency.
    // inter-op number do not improve the efficiency, so just set inter-op to 1.
    const int intra_num_cpu = (g_cpu_number >= 8) ? (g_cpu_number - 2 - 1) : (g_cpu_number - 1 - 1);
    const int inter_num_cpu = 1;
    cout << "Intra-op number for IR-Net is: " << intra_num_cpu << endl;
    so.SetIntraOpNumThreads(intra_num_cpu);
    so.SetInterOpNumThreads(inter_num_cpu);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, onnx_path.c_str(), so);
}


void P2V_Decoder_ONNX::infer(const float* feat, const float* u, const uint8_t* mask, int B, int Nn, float* out) {
// std::vector<std::vector<float>> P2V_Decoder_ONNX::infer(const float* feat, const float* u, const uint8_t* mask, int B, int Nn) {

    const int64_t feat_shape[3] = {B, Nn, 64}; // feat: [B, Nn, 64]
    const int64_t u_shape[2] = {B, 3};         // u: [B, 3]
    const int64_t mask_shape[2] = {B, Nn};     // mask: [B, Nn]

    // feat tensor
    Ort::Value ort_feat = Ort::Value::CreateTensor<float>(
        meminfo_,
        const_cast<float*>(feat),
        B * Nn * 64,
        feat_shape,
        3
    );

    // u tensor
    Ort::Value ort_u = Ort::Value::CreateTensor<float>(
        meminfo_,
        const_cast<float*>(u),
        B * 3,
        u_shape,
        2
    );

    std::unique_ptr<bool[]> mask_bool(new bool[B * Nn]);
    for (int i = 0; i < B * Nn; ++i)
        mask_bool[i] = (mask[i] != 0);

    Ort::Value ort_mask = Ort::Value::CreateTensor<bool>(
        meminfo_,
        mask_bool.get(),
        B * Nn,
        mask_shape,
        2
    );

    // Create input.
    std::array<Ort::Value, 3> inputs = {        // names should keep the same as python's model output
        std::move(ort_feat),
        std::move(ort_u),
        std::move(ort_mask)
    };

    // parallel inference
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_,
        inputs.data(),
        3,
        output_names_,
        4
    );

    // Get output data
    float* p2v_data = outputs[0].GetTensorMutableData<float>();  // [B, 3]
    float* sigma_data = outputs[1].GetTensorMutableData<float>();  // [B, 1]
    float* score_data = outputs[2].GetTensorMutableData<float>();  // [B, 1]
    float* phi_data = outputs[3].GetTensorMutableData<float>();  // [B, 1]

    // Extract output data.
    std::vector<std::vector<float>> result(B);
    // --- pack into out[B,6] ---
    // out[i*6 + 0..2] = p2v, out[i*6+3]=sigma, +4 score, +5 phi
    for (int i = 0; i < B; ++i) {
        out[i*6 + 0] = p2v_data[i*3 + 0];
        out[i*6 + 1] = p2v_data[i*3 + 1];
        out[i*6 + 2] = p2v_data[i*3 + 2];
        out[i*6 + 3] = sigma_data[i];
        out[i*6 + 4] = score_data[i];
        out[i*6 + 5] = phi_data[i];
    }
}

