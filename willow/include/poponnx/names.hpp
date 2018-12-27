#ifndef GUARD_NEURALNET_NAMES_HPP
#define GUARD_NEURALNET_NAMES_HPP

#include <map>
#include <vector>

// forward declaring onnx classes, done to
// help reduce the number of compilation units
// including the onnx headers.
namespace onnx {
class NodeProto;
class TensorProto;
class AttributeProto;
class GraphProto;
class ValueInfoProto;
class ModelProto;
class TypeProto;
} // namespace onnx

namespace google {
namespace protobuf {
template <class T> class RepeatedPtrField;
}
} // namespace google

namespace poponnx {

// Some type aliases, which hopefully
// make poponnx code clearer
using Shape        = std::vector<int64_t>;
using Rank         = int;
using TensorId     = std::string;
using TaskId       = std::string;
using OpName       = std::string;
using OpDomain     = std::string;
using OpType       = std::string;
using OpVersion    = unsigned;
using OpId         = int;
using ReturnPeriod = int;
// The position at which a Tensor is consumed by an Op
using InIndex = int;
// The position at which a Tensor is output by an Op
using OutIndex = int;

// forward declaring several poponnx classes
class DataFlow;
class Device;
class InputShapeInfo;
class Ir;
class Loss;
class Op;
class Optimizer;
class Pattern;
class Tensor;
class Scheduler;
class TensorIndexMap;
class TensorInfo;
class ConstVoidData;
class MutableOutData;
class IStepIO;
class GradInOutMapper;
class InputMapWrapper;
class OutputMapWrapper;
class Vertex;
enum class TensorType;

// equivalent to decltype(&onnx::NodeProto().attribute(0))
using onnxAttPtr     = const onnx::AttributeProto *;
using NodeAttributes = google::protobuf::RepeatedPtrField<onnx::AttributeProto>;
using OnnxTensors    = std::map<TensorId, onnx::TensorProto>;
using Node           = onnx::NodeProto;
using OnnxTensorPtrs = std::map<TensorId, const onnx::TensorProto *>;
using OpsBeforeKey   = std::map<Op *, std::vector<Op *>>;

// Custom node attribute names
static constexpr const char *sRecomputeOutputAttribute =
    "__recomputeOutputInBackwardPass";
static constexpr const char *sVirtualGraphAttribute = "__ipuNumber";

} // namespace poponnx

#endif
