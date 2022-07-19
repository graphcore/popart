// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_INPLACE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_INPLACE_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/patterns/pattern.hpp>

#include "popart/names.hpp"

namespace popart {
class Op;
class Tensor;
struct OperatorIdentifier;

// A class for creating an Op and Tensors without inserting them
// into the Ir. It is useful for testing what the effects of an
// Inplace Op are before applying the Pattern
class ExternOpTensorBundle {

public:
  // copyOp : the op whose inputs and outputs will be cloned
  // testOp : the op which will have the cloned tensors connected to it
  //          ownership of opNew is given to this new object
  ExternOpTensorBundle(Op *copyOp, std::unique_ptr<Op> testOp);
  ~ExternOpTensorBundle();

  // return the "test" Op
  Op *getOp();

private:
  std::unique_ptr<Op> up_op;
  std::map<TensorId, std::unique_ptr<Tensor>> tensors;
};

// for all cases with 0 outputs
class Inplace : public Pattern {
public:
  Inplace();

  // which tensors are touched if "op" is replaced by type "inpl"
  std::vector<const Tensor *> touches(Op *op, OperatorIdentifier inpl) const;

  // Replace "op" with an Op of type "inpl", using already calculated
  // constraints (with "op" in places where "inpl" should be) cons
  bool apply(Op *op, OperatorIdentifier inpl, const OpsBeforeKey &cons) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_INPLACE_HPP_
