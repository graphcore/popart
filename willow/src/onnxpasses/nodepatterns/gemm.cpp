// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/gemm.hpp>
#include <vector>

#include "popart/attributes.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

// When input size is 3 inputs are A, B, C and
// we compute alpha * A * B + beta * C.
// When input size is 2 inputs are A, B and
// we compute alpha * A * B.
bool Gemm::go(const NodeProto &node) {
  if (node.op_type() == "Gemm") {
    int nSize          = node.input_size();
    const auto inA     = node.input(0);
    const auto inB     = node.input(1);
    const auto outName = node.output(0);

    auto attr   = Attributes(node.attribute());
    bool transA = attr.getAttribute<Attributes::Int>("transA", false);
    bool transB = attr.getAttribute<Attributes::Int>("transB", false);
    float alpha = attr.getAttribute<Attributes::Float>("alpha", 1.0);

    // Compute transpose of tensors A and B if needed.
    const auto transposeA = transA ? transposeTensor(node, inA, outName) : inA;
    const auto transposeB = transB ? transposeTensor(node, inB, outName) : inB;

    // mulAB matrix multiplication of tensors A and B.
    const auto mulABName = withUniqueSuffix(outName);
    binary(node, {transposeA, transposeB}, mulABName, "MatMul");

    if (nSize == 3) {
      // Multiply tensor mulAB by value alpha. Multiply tensor C by value beta.
      // Add them together.
      // alpha * mulAB.
      const auto alphaABName = withUniqueSuffix(outName);
      scale(node, mulABName, alphaABName, alpha);

      auto inC   = node.input(2);
      float beta = attr.getAttribute<Attributes::Float>("beta", 1.0);

      // beta * C.
      const auto betaCName = withUniqueSuffix(outName);
      scale(node, inC, betaCName, beta);

      // alpha * mulAB + beta * C.
      binary(node, {alphaABName, betaCName}, outName, "Add");
    } else if (nSize == 2) {
      // Multiply tensor mulAB by value alpha.
      scale(node, mulABName, outName, alpha);
    } else {
      throw error("Node pattern, number of Gemm op inputs is {}. "
                  "Number of Gemm inputs can be 2 or 3.",
                  nSize);
    }

    return true;
  }
  return false;
}

std::string Gemm::transposeTensor(const NodeProto &node,
                                  const std::string &in,
                                  const std::string &out) {
  const auto transposeOutName = withUniqueSuffix(out);
  auto &transposeNode         = unary(node, in, transposeOutName, "Transpose");
  transposeNode.clear_attribute();
  copyUnderscorePrefixedAttributes(node, transposeNode);
  std::vector<int64_t> perm{1, 0};
  addIntsAttribute(transposeNode, "perm", perm);
  transposeNode.set_domain("ai.onnx");

  return transposeOutName;
}

void Gemm::scale(const NodeProto &node,
                 const std::string &in,
                 const std::string &out,
                 float value) {
  auto &scaleNode = unary(node, in, out, "Scale");
  scaleNode.clear_attribute();
  copyUnderscorePrefixedAttributes(node, scaleNode);
  auto &att = *scaleNode.add_attribute();
  att.set_name("scale");
  att.set_f(value);
  scaleNode.set_domain("ai.graphcore");
}

} // namespace onnxpasses
} // namespace popart
