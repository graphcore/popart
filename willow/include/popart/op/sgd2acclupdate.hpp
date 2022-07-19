// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SGD2ACCLUPDATE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SGD2ACCLUPDATE_HPP_

#include <memory>
#include <popart/op/sgd1acclupdate.hpp>

namespace popart {
class Op;

/**
 * This Op is by design exactly equivalent to an SGD1AcclUpdateOp. Any logic
 * based on an SGD1AcclUpdateOp, like transform code or lowering into Opx, can
 * be applied to an SGD2PartialAcclUpdateOp. This includes the OperatorIdentifer
 * being Onnx::CustomOperators::SGD1AcclUpdateOp.
 *
 * For SGD2, the entire v update equation could be done in one op (see equation
 * derivation in optimizer.hpp); however, we reuse the SG1AcclUpdateOp and
 * AccumulateOp to implement the equation in the two steps.
 */
class SGD2PartialAcclUpdateOp : public SGD1AcclUpdateOp {
  /*
   * SGD2 uses a two-step update, instead of implementing a single
   * SGD2AcclUpdateOp for the following reasons:
   *   1. Re-use existing Ir-level functionality. Do need to maintain entire
   *      "stack" for implementing an Op, like an Opx implementation.
   *   2. Re-useing existing Ir-level functionality also gives us for free all
   *      Ir-level logic (for correctness or optimisation) that is already
   *      defined for the reused ops.
   *   3. If we choose to remove these special optimiser ops in the future and
   *      use normal inplace ops (or any other refactoring) then there is one
   *      less op to consider.
   *
   * The equivalence with SGD1AcclUpdateOp is because, for SGD1 and SGD2, the
   * Op required is exactly the same, so an SGD1AcclUpdateOp could be used, but
   * for naming clarity, we define this op that extends it (in name only). We do
   * not rename SGD1AcclUpdateOp to something more generic like SGDAcclUpdateOp
   * as that would be a breaking change.
   *
   * If this assumption changes, then an SGD2AcclUpdateOp is no longer an
   * SGD1AcclUpdateOp, so this is-a relationship should be reconsidered. A
   * separate SGD2AcclUpdateOpx may also be required.
   */
public:
  using SGD1AcclUpdateOp::SGD1AcclUpdateOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SGD2ACCLUPDATE_HPP_
