#ifndef GUARD_NEURALNET_NAMES_HPP
#define GUARD_NEURALNET_NAMES_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx_pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <ostream>
#include <sstream>

namespace willow {

using TensorId = std::string;

using OpId = int;
// using TensorId = std::string;

// The position at which a Tensor is consumed by an Op
using InIndex = int;

// The position at which a Tensor is output by an Op
using OutIndex = int;

using Node = onnx::NodeProto;

using onnxAttPtr = decltype(&onnx::NodeProto().attribute(0));

using OnnxTensors = std::map<TensorId, onnx::TensorProto>;

using OnnxTensorPtrs = std::map<TensorId, const onnx::TensorProto *>;

using DataType = decltype(onnx::TensorProto().data_type());

using TP = onnx::TensorProto;

class DataFlow;
class Device;
class EarlyInfo;
class Graph;
class Loss;
class Op;
class Optimizer;
class Pattern;
class Tensor;

} // namespace willow

#endif
