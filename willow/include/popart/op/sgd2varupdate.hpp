// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD2VARUPDATE_HPP
#define GUARD_NEURALNET_SGD2VARUPDATE_HPP

#include <memory>
#include <popart/op/sgd1varupdate.hpp>

namespace popart {
class Op;

/**
 * This Op is by design exactly equivalent to an SGD1VarUpdateOp. Any logic
 * based on an SGD1VarUpdateOp, like transform code or lowering into Opx, can be
 * applied to an SGD2VarUpdateOp. This includes the OperatorIdentifer being
 * Onnx::CustomOperators::SGD1VarUpdate.
 */
class SGD2VarUpdateOp : public SGD1VarUpdateOp {
  /*
   * This equivalence with SGD1VarUpdateOp is because, for SGD1 and SGD2, the Op
   * required is exactly the same, so an SGD1VarUpdateOp could be used, but for
   * naming clarity, we define this op that extends it (in name only). We do not
   * rename SGD1VarUpdateOp to something more generic like SGDVarUpdateOp as
   * that would be a breaking change.
   *
   * If this assumption changes, then an SGD2VarUpdateOp is no longer an
   * SGD1VarUpdateOp, so this is-a relationship should be reconsidered. A
   * separate SGD2VarUpdateOpx may also be required.
   */
public:
  using SGD1VarUpdateOp::SGD1VarUpdateOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
