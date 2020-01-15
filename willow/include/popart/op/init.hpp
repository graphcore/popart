#ifndef GUARD_NEURALNET_INIT_HPP
#define GUARD_NEURALNET_INIT_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

enum class InitType { NONE = 0, ZERO };

class InitOp : public Op {
public:
  InitOp(const OperatorIdentifier &,
         const TensorInfo &,
         const TensorType &,
         const InitType &,
         const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getOutIndex() { return 0; }

  TensorInfo getTensorInfo() { return tensor_info; }
  TensorType getTensorType() { return tensor_type; }
  InitType getInitType() { return init_type; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool isOutlineable() const final { return tensor_type != TensorType::Cache; }

private:
  TensorInfo tensor_info;
  TensorType tensor_type;
  InitType init_type;
};

} // namespace popart

#endif
