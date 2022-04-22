// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/trig.hpp>
#include <string>

#include "onnxpasses/nodepattern.hpp"

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

bool Tan::go(const NodeProto &node) {
  if (node.op_type() == "Tan") {
    const auto inName     = node.input(0);
    const auto outName    = node.output(0);
    const auto sinOutName = withUniqueSuffix(outName);
    unary(node, inName, sinOutName, "Sin");
    const auto cosOutName = withUniqueSuffix(outName);
    unary(node, inName, cosOutName, "Cos");
    binary(node, {sinOutName, cosOutName}, node.output(0), "Div");
    return true;
  }
  return false;
}

bool Asinh::go(const NodeProto &node) {
  if (node.op_type() == "Asinh") {
    const auto asinhInName  = node.input(0);
    const auto outName      = node.output(0);
    const auto pow2OutName  = withUniqueSuffix(outName);
    const auto plus1OutName = withUniqueSuffix(outName);
    const auto sqrtOutName  = withUniqueSuffix(outName);
    const auto preLogName   = withUniqueSuffix(outName);
    powConstScalar(node, asinhInName, pow2OutName, ScalarInIndex::One, 2.0);
    addConstScalar(node, pow2OutName, plus1OutName, ScalarInIndex::One, 1.0);
    unary(node, plus1OutName, sqrtOutName, "Sqrt");
    binary(node, {asinhInName, sqrtOutName}, preLogName, "Add");
    unary(node, preLogName, outName, "Log");
    return true;
  }
  return false;
}

bool Acosh::go(const NodeProto &node) {
  if (node.op_type() == "Acosh") {
    const auto asinhInName   = node.input(0);
    const auto outName       = node.output(0);
    const auto pow2OutName   = withUniqueSuffix(outName);
    const auto minus1OutName = withUniqueSuffix(outName);
    const auto sqrtOutName   = withUniqueSuffix(outName);
    const auto sumOutName    = withUniqueSuffix(outName);
    powConstScalar(node, asinhInName, pow2OutName, ScalarInIndex::One, 2.0);
    subConstScalar(node, pow2OutName, minus1OutName, ScalarInIndex::One, 1.0);
    unary(node, minus1OutName, sqrtOutName, "Sqrt");
    binary(node, {asinhInName, sqrtOutName}, sumOutName, "Add");
    unary(node, sumOutName, outName, "Log");
    return true;
  }
  return false;
}

bool Acos::go(const NodeProto &node) {
  if (node.op_type() == "Acos") {
    const auto acosInName = node.input(0);
    const auto outName    = node.output(0);
    const auto asinOut    = withUniqueSuffix(outName);
    const auto diffOut    = withUniqueSuffix(outName);
    unary(node, acosInName, asinOut, "Asin");
    constexpr double piBy2 = 3.14159265358979323846 / 2.0;
    subConstScalar(node, asinOut, outName, ScalarInIndex::Zero, piBy2);
    return true;
  }
  return false;
}

bool Atanh::go(const NodeProto &node) {
  if (node.op_type() == "Atanh") {
    const auto atanhInName  = node.input(0);
    const auto outName      = node.output(0);
    const auto onePlus      = withUniqueSuffix(outName);
    const auto oneMinus     = withUniqueSuffix(outName);
    const auto ratioName    = withUniqueSuffix(outName);
    const auto logRatioName = withUniqueSuffix(outName);
    addConstScalar(node, atanhInName, onePlus, ScalarInIndex::Zero, 1.0);
    subConstScalar(node, atanhInName, oneMinus, ScalarInIndex::Zero, 1.0);
    binary(node, {onePlus, oneMinus}, ratioName, "Div");
    unary(node, ratioName, logRatioName, "Log");
    mulConstScalar(node, logRatioName, outName, ScalarInIndex::Zero, 0.5);
    return true;
  }
  return false;
}

} // namespace onnxpasses
} // namespace popart
