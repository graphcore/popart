// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "builder_helper.hpp"

#include <algorithm>
#include <builder_impl.hpp>
#include <cstdint>
#include <popart/op/receptive.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/vendored/any.hpp"

namespace popart {

void verifyWindowParameters(std::unique_ptr<BuilderImpl> &impl,
                            TensorId input,
                            std::vector<int64_t> strides,
                            std::vector<int64_t> padding,
                            std::vector<int64_t> outPadding,
                            const std::vector<int64_t> &kernel_shape,
                            std::vector<int64_t> dilation   = {},
                            std::vector<int64_t> inDilation = {},
                            const std::string &auto_pad     = "NOTSET",
                            bool ceil_mode                  = false) {

  // TODO T17932 : We do not have a mechanism for infering the output shape
  // of custom ops, so this set of checks can only be applied if the tensor
  // shape is known
  if (impl->hasTensorShape(input)) {
    auto num_spatial_dims = impl->getTensorShape(input).size() - 2;
    if (num_spatial_dims < 1) {
      throw error("Input tensor has no spatial dimensions");
    }
    if (strides.size() != 0 && strides.size() != num_spatial_dims) {
      throw error(
          "Length of strides vector {} != number of spatial dimensions {}",
          strides.size(),
          num_spatial_dims);
    }
    if (padding.size() != 0 && padding.size() != num_spatial_dims * 2) {
      throw error("Padding vector (length {}) does not have 2 values for each "
                  "spatial dimension {}",
                  padding.size(),
                  num_spatial_dims);
    }
    if (dilation.size() != 0 && dilation.size() != num_spatial_dims) {
      throw error(
          "Length of dilations vector {} != number of spatial dimensions {}",
          dilation.size(),
          num_spatial_dims);
    }
    if (inDilation.size() != 0 && inDilation.size() != num_spatial_dims) {
      throw error(
          "Length of inDilations vector {} != number of spatial dimensions {}",
          inDilation.size(),
          num_spatial_dims);
    }

    // Validate that the input shape, kernel shape, strides, padding, and
    // optional dilation combine to produce a valid output shape
    if (kernel_shape.size()) {
      Shape inShape = impl->getTensorShape(input);
      inShape.erase(inShape.begin(), inShape.begin() + 2);

      // Default 'zeros'
      if (padding.empty()) {
        padding.resize(2 * num_spatial_dims, 0);
      }
      if (outPadding.empty()) {
        outPadding.resize(2 * num_spatial_dims, 0);
      }
      // Default 'ones'
      if (dilation.empty()) {
        dilation.resize(num_spatial_dims, 1);
      }
      if (inDilation.empty()) {
        inDilation.resize(num_spatial_dims, 1);
      }
      if (strides.empty()) {
        strides.resize(num_spatial_dims, 1);
      }

      Shape spatialOutShape = HasReceptiveFieldOp::getSpatialOutShape(
          inShape,
          kernel_shape,
          padding,
          outPadding,
          strides,
          dilation,
          inDilation,
          HasReceptiveFieldOp::getAutoPad(auto_pad),
          ceil_mode);

      if (std::any_of(spatialOutShape.begin(),
                      spatialOutShape.end(),
                      [](int64_t i) { return i < 0; })) {
        throw error("Window parameter values combine to give invalid spatial "
                    "output shape: {}",
                    spatialOutShape);
      }
    }
  }
}

void verifyPoolBase(std::unique_ptr<BuilderImpl> &impl,
                    std::vector<TensorId> inputs,
                    std::map<std::string, popart::any> attributes) {
  // Prepare attributes for verifyWindowParameters:
  // If attributes are unspecified (i.e. they do not
  // exist in the 'attributes' map) then set as empty
  std::vector<int64_t> emptyVec;
  if (!attributes.count("strides")) {
    attributes["strides"] = emptyVec;
  }
  if (!attributes.count("pads")) {
    attributes["pads"] = emptyVec;
  }
  if (!attributes.count("outPads")) {
    attributes["outPads"] = emptyVec;
  }
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
  }
  if (!attributes.count("inDilations")) {
    attributes["inDilations"] = emptyVec;
  }
  std::string emptyString;
  if (!attributes.count("auto_pad")) {
    attributes["auto_pad"] = emptyString;
  }
  bool ceil_mode = false;
  if (attributes.count("ceil_mode")) {
    ceil_mode = popart::any_cast<int64_t>(attributes["ceil_mode"]);
  }

  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["outPads"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["kernel_shape"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["inDilations"]),
      popart::any_cast<const std::string &>(attributes["auto_pad"]),
      ceil_mode);
}

// Functions that are expected by the generated code when the verifyInput
// is set to true.

void verifyConvBase(std::unique_ptr<BuilderImpl> &impl,
                    std::vector<TensorId> inputs,
                    std::map<std::string, popart::any> attributes) {
  // TODO T17932 : We do not have a mechanism for infering the output shape
  // of custom ops, so this check can only be applied if the tensor shape
  // is known
  Shape weightsKShape;
  if ((inputs.size() > 1) && impl->hasTensorShape(inputs[1])) {
    weightsKShape = impl->getTensorShape(inputs[1]);
    weightsKShape.erase(weightsKShape.begin(), weightsKShape.begin() + 2);

    // Verify that the optional kernel_shape attribute matches the inferred
    // shape from the weight tensor's shape
    if (attributes.count("kernel_shape")) {
      auto userKShape =
          popart::any_cast<const Shape &>(attributes["kernel_shape"]);

      if (userKShape != weightsKShape) {
        throw error(
            "kernel_shape, {}, does not match inferred shape from weight "
            "input '{}', {}",
            userKShape,
            inputs[1],
            weightsKShape);
      }
    }
  } else if (inputs.size() < 2) {
    throw error("Conv requires at least two inputs: data, and weights. {} "
                "inputs provided.",
                inputs.size());
  }

  // Prepare attributes for verifyWindowParameters:
  // If attributes are unspecified (i.e. they do not
  // exist in the 'attributes' map) then set as empty
  std::vector<int64_t> emptyVec;
  if (!attributes.count("strides")) {
    attributes["strides"] = emptyVec;
  }
  if (!attributes.count("pads")) {
    attributes["pads"] = emptyVec;
  }
  if (!attributes.count("outPads")) {
    attributes["outPads"] = emptyVec;
  }
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
  }
  if (!attributes.count("inDilations")) {
    attributes["inDilations"] = emptyVec;
  }
  std::string emptyString;
  if (!attributes.count("auto_pad")) {
    attributes["auto_pad"] = emptyString;
  }

  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["outPads"]),
      weightsKShape,
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["inDilations"]),
      popart::any_cast<std::string>(attributes["auto_pad"]));
}

void verify_AiOnnxOpset6_Conv_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyConvBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset11_Conv_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyConvBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset6_AveragePool_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset7_AveragePool_7(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset10_AveragePool_10(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset11_AveragePool_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset6_MaxPool_1(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset8_MaxPool_8(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset10_MaxPool_10(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verify_AiOnnxOpset11_MaxPool_11(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

void verifyPadBase(std::unique_ptr<BuilderImpl> &impl,
                   std::vector<TensorId> inputs,
                   std::map<std::string, popart::any> attributes) {

  auto rank = impl->getTensorShape(inputs[0]).size();
  auto &pads =
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]);
  if (pads.size() != rank * 2) {
    throw error(
        "Padding vector (length {}) doesn't contain 2 entries per input "
        "dimension {}",
        pads.size(),
        rank);
  }
}

void verify_AiOnnxOpset6_Pad_2(
    std::unique_ptr<BuilderImpl> &impl,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> &attributes) {
  verifyPadBase(impl, inputs, attributes);
}

} // namespace popart
