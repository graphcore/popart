#ifndef GUARD_NEURALNET_CONSTEXPR_HPP
#define GUARD_NEURALNET_CONSTEXPR_HPP

#include <map>
#include <poponnx/attributes.hpp>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

// Base class for processing NodeProtos as constant expressions
class ConstExprOp {
public:
  ConstExprOp(const onnx::NodeProto &, Ir *);
  virtual ~ConstExprOp() = default;
  // insert the output of the Node as a constant tensor
  virtual void insertOutput() = 0;
  Tensor *atInIndex(InIndex) const;
  TensorId atOutIndex0() const;
  void
  addConstInitTensor(const TensorId &, const TensorInfo &, const void *) const;

  template <typename OpFunctor, typename... Args>
  static std::vector<char> callOpFunctor(DataType dtype, Args &&... args);

protected:
  // The NodeProto to process as a constant expression
  const onnx::NodeProto &node;
  Ir *ir;
  const Attributes nAtts;
};

class ConstExprClassifier {
public:
  ConstExprClassifier(std::map<TensorId, bool> &&M_) : M(M_) {}
  bool isConstExprTensor(TensorId id) const;

private:
  std::map<TensorId, bool> M;
};

// for every Tensor which is the output of a
// Node in an onnx::Graph, can its value be
// computed just once, on host? If so, we say that
// it is a ConstExprTensor and that its producing
// Node is a ConstExprNode.

// This class exposes functions for determining which
// Tensors are ConstExprTensors, and another for processing
// ConstExprNodes.
class ConstExprUtil {

public:
  // Determine which Tensors are ConstExprTensors.
  // The rule for determining if a tensor is a ConstExprTensor:
  // For eval and infer, a tensor is NOT ConstExprTensor if:
  //      1) there is a path from a Stream Tensor to it
  // For training, a tensor is NOT ConstExprTensor if either:
  //      1) there is a path from a Stream Tensor to it
  //      2) there is a path from a Variable Tensor to it
  // The initial Stream/Variable tensors are passed in as sourceTensors
  ConstExprClassifier getClassifier(const onnx::GraphProto &,
                                    const std::vector<TensorId> &sourceTensors);

  // process a ConstExprNode "node", modfying the Ir pointed to by "ir"
  void processNode(const onnx::NodeProto &node, Ir *ir);

private:
  static int getOutIndex(const onnx::NodeProto &, const TensorId &);
  static bool isNodeOutputAlwaysConstExpr(const OpType &, OutIndex);
};

template <typename OpFunctor, typename... Args>
std::vector<char> ConstExprOp::callOpFunctor(DataType dtype, Args &&... args) {
  switch (dtype) {
  case DataType::DOUBLE:
    return OpFunctor().template operator()<double>(std::forward<Args>(args)...);
  case DataType::FLOAT:
    return OpFunctor().template operator()<float>(std::forward<Args>(args)...);
  case DataType::INT64:
    return OpFunctor().template operator()<int64_t>(
        std::forward<Args>(args)...);
  case DataType::INT32:
    return OpFunctor().template operator()<int32_t>(
        std::forward<Args>(args)...);
  case DataType::INT16:
    return OpFunctor().template operator()<int16_t>(
        std::forward<Args>(args)...);
  case DataType::INT8:
    return OpFunctor().template operator()<int8_t>(std::forward<Args>(args)...);
  case DataType::UINT64:
    return OpFunctor().template operator()<uint64_t>(
        std::forward<Args>(args)...);
  case DataType::UINT32:
    return OpFunctor().template operator()<uint32_t>(
        std::forward<Args>(args)...);
  case DataType::UINT16:
    return OpFunctor().template operator()<uint16_t>(
        std::forward<Args>(args)...);
  case DataType::UINT8:
    return OpFunctor().template operator()<uint8_t>(
        std::forward<Args>(args)...);
  case DataType::FLOAT16:
  case DataType::BOOL:
  case DataType::BFLOAT16:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::STRING:
  case DataType::UNDEFINED:
  default:
    throw error("functor {} does not support DataType::{}",
                typeid(OpFunctor).name(),
                getDataTypeInfoMap().at(dtype).name());
  }
}

} // namespace poponnx

#endif
