#ifndef GUARD_NEURALNET_INPLACE_HPP
#define GUARD_NEURALNET_INPLACE_HPP

#include <poponnx/patterns/pattern.hpp>

namespace poponnx {

// A class for creating an Op and Tensors without inserting them
// into the Ir. It is useful for testing what the effects of an
// Inplace Op are before applying the Pattern
class ExternOpTensorBundle {

public:
  // copyOp : the op whose inputs and outputs will be cloned
  // testOp : the op which will have the cloned tensors connected to it
  //          ownership of opNew is given to this new object
  ExternOpTensorBundle(Op *copyOp, std::unique_ptr<Op> testOp);

  // return the "test" Op
  Op *getOp();

private:
  std::unique_ptr<Op> up_op;
  std::map<TensorId, std::unique_ptr<Tensor>> tensors;
};

// for all cases with 0 outputs
class Inplace : public Pattern {
public:
  bool matches(Op *op) const final;
  // which input indices does this Inplace Pattern target?
  virtual std::vector<InIndex> targetInIndices(Op *) const = 0;
  std::vector<const Tensor *> touches(Op *op) const final;
  bool apply(Op *op) const final;
  std::tuple<bool, OperatorIdentifier> firstGoodVariant(Op *op) const;
  // What are the additional topological constraints
  // required if oldOp is replaced by newOp?
  OpsBeforeKey getNewTopoCons(Op *oldOp, Op *newOp) const;
  PatternPhase phase() const final { return PatternPhase::WITHTOPOCONS; }
};

// For an Op "op" with
//    N inputs and 1 output,
// replace it with an in-place Op with
//    N inputs and 0 outputs,
// with the in-place Op modifying input 0 to have the
// value that the output of op would have had.
class Inplace0 : public Inplace {
public:
  virtual std::vector<InIndex> targetInIndices(Op *) const final { return {0}; }
};

class InplaceAll : public Inplace {
public:
  virtual std::vector<InIndex> targetInIndices(Op *op) const final {
    std::vector<InIndex> indices;
    for (auto &x : op->input->tensorMap()) {
      indices.push_back(x.first);
    }
    return indices;
  }
};

} // namespace poponnx

#endif
