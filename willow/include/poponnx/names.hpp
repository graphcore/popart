#ifndef GUARD_NEURALNET_NAMES_HPP
#define GUARD_NEURALNET_NAMES_HPP

#include <onnx/onnx_pb.h>

#include <ostream>
#include <sstream>

namespace poponnx {

using TensorId = std::string;
using TaskId   = std::string;

using OpName   = std::string;
using OpDomain = std::string;

using OpId = int;

// The position at which a Tensor is consumed by an Op
using InIndex = int;

// The position at which a Tensor is output by an Op
using OutIndex = int;

using Rank = int;

using Node = onnx::NodeProto;

using onnxAttPtr = decltype(&onnx::NodeProto().attribute(0));

using OnnxTensors = std::map<TensorId, onnx::TensorProto>;

using OnnxTensorPtrs = std::map<TensorId, const onnx::TensorProto *>;

using DataType = decltype(onnx::TensorProto().data_type());

using TP = onnx::TensorProto;

using Shape = std::vector<int64_t>;

class DataFlow;
class Device;
class EarlyInfo;
class Ir;
class Loss;
class Op;
class Optimizer;
class Pattern;
class Tensor;
class ConstVoidData;
class MutableOutData;
class StepIO;

using OpsBeforeKey = std::map<Op *, std::vector<Op *>>;

} // namespace poponnx

#endif
