// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_INIT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_INIT_HPP_

#include <memory>
#include <popart/op.hpp>
#include <popart/tensor.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// Initialisation behaviour has consequences for the liveness of tensors,
// as well as their values before each new iteration.

// None: Do not populate tensor in each iteration;
//       only safe if tensor is overwritten by it's consumer before use.
//       For example RemoteLoad and DynamicUpdate are safe consumers, while
//       DynamicAdd needs a zero-initialised tensor.
// Zero: Populate tensor with zeros in each iteration (like np.zeros)
enum class InitType { NoInit = 0, Zero };

// Initialize a new tensor given
// shape, data type, tensor type and initialization type.
// Allows to create initialized tensors in any graph.
// The InitOp has no tensor inputs and one tensor output.
class InitOp : public Op {
public:
  InitOp(const OperatorIdentifier &,
         const TensorInfo &,
         const TensorType &,
         const InitType &,
         const Op::Settings &,
         const int = -1);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getOutIndex() { return 0; }

  TensorInfo getTensorInfo() const { return tensor_info; }
  TensorType getTensorType() const { return tensor_type; }
  InitType getInitType() const { return init_type; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool isOutlineable() const final {
    // The deal here is that off-chip variables have an on-chip counterpart.
    // Variables, for now, can only exist in the main graph.
    // This is such that the behaviour between InitOp created on-chip variables
    // and normal variables is consistent: They must exist in the main graph.
    return tensor_type != TensorType::Variable;
  }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  int getOutBatchAxis(OutIndex) const override { return batch_axis; }

  bool canShard() const override { return batch_axis != -1; }

private:
  TensorInfo tensor_info;
  TensorType tensor_type;
  InitType init_type;
  int batch_axis;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_INIT_HPP_
