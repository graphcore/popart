// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_GEMM_HPP
#define GUARD_NEURALNET_ONNXTOONNX_GEMM_HPP

#include <memory>
#include <onnxpasses/nodepattern.hpp>
#include <string>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {
class PatternTarget;

// Gemm(A, B, C) = alpha * transpose(A) * transpose(B) + beta * C.
// A, B, C are tensors. C is optional. alpha and beta are values.
// Transpose(s) is optional.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm

class Gemm : public NodePattern {
public:
  Gemm(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  /** Compute transpose of tensor.
   *
   * \param node The NodeProto to copy (attributes etc.) from.
   *
   * \param in The input to transpose.
   *
   * \param out The output of the Node to create transposed tensor name.
   *
   * Return string which represents the transposed tensor - name.
   * */
  std::string transposeTensor(const NodeProto &node,
                              const std::string &in,
                              const std::string &out);

  /**
   * Multiply the Tensor with name \a in by a constant scalar value.
   * */
  void scale(const NodeProto &node,
             const std::string &in,
             const std::string &out,
             float value);

  bool go(const NodeProto &node) final;
};

} // namespace onnxpasses
} // namespace popart

#endif
