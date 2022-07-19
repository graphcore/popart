// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_BUILDER_HELPER_HPP_
#define POPART_WILLOW_SRC_BUILDER_HELPER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class BuilderImpl;
class any;

void verifyWindowParameters(std::unique_ptr<BuilderImpl> &impl,
                            TensorId input,
                            std::vector<int64_t> strides,
                            std::vector<int64_t> padding,
                            std::vector<int64_t> outPadding,
                            const std::vector<int64_t> &kernel_shape,
                            std::vector<int64_t> dilation,
                            std::vector<int64_t> inDilation,
                            const std::string &auto_pad,
                            bool ceil_mode);
void verifyPoolBase(std::unique_ptr<BuilderImpl> &impl,
                    std::vector<TensorId> inputs,
                    std::map<std::string, popart::any> attributes);

// Functions that are expected by generated code when the verifyInput
// is set to true.
void verifyConvBase(std::unique_ptr<BuilderImpl> &impl,
                    std::vector<TensorId> inputs,
                    std::map<std::string, popart::any> attributes);
void verify_AiOnnxOpset6_Conv_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset11_Conv_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset6_AveragePool_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset7_AveragePool_7(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset10_AveragePool_10(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset11_AveragePool_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset6_MaxPool_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset8_MaxPool_8(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset10_MaxPool_10(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verify_AiOnnxOpset11_MaxPool_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

void verifyPadBase(std::unique_ptr<BuilderImpl> &impl,
                   std::vector<TensorId> inputs,
                   std::map<std::string, popart::any> attributes);

void verify_AiOnnxOpset6_Pad_2(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes);

} // namespace popart

#endif // POPART_WILLOW_SRC_BUILDER_HELPER_HPP_
