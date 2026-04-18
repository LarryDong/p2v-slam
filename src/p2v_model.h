#ifndef P2V_MODEL_H
#define P2V_MODEL_H

#include "common_lib.h"
#include "torch/csrc/jit/api/module.h"
#include <Eigen/Core>
#include <torch/script.h>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>


// IR-Net (Encoder)
class P2V_Encoder_ONNX {
public:
    P2V_Encoder_ONNX(const std::string& onnx_path);
    // pts: [1,K,3] float32
    // mask: [1,K] uint8 or bool
    // return: [64] float
    std::vector<float> infer(const float* pts, const uint8_t* mask, int K);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo meminfo_;

    // names should keep the same as `convert_to_onnx.py` output
    const char* input_names_[2] = {"pts", "mask"};
    const char* output_names_[1] = {"feat"};
};



// VE-Net (Decoder)
class P2V_Decoder_ONNX {
public:
    P2V_Decoder_ONNX(const std::string& onnx_path);

    // feat: [B, Nn, 64] float32
    // u: [B, 3] float32
    // mask: [B, Nn] uint8 or bool
    // return: [B, p2v+sigma+score+phi] float
    void infer(const float* feat, const float* u, const uint8_t* mask, int B,  int Nn, float* out);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo meminfo_;

    // names should keep the same as `convert_to_onnx.py` output
    const char* input_names_[3] = {"feat27", "u", "exist_mask_27"};
    const char* output_names_[4] = {"p2v_vec", "p2v_sigma", "score", "phi"};
};


#endif
