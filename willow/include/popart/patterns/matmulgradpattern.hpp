#ifndef GUARD_NEURALNET_MATMULGRADPATTERN_HPP
#define GUARD_NEURALNET_MATMULGRADPATTERN_HPP

#include <popart/op/matmul.hpp>
#include <popart/patterns/pattern.hpp>

namespace popart {

/*
  The intention of this pattern is to make sure that all matmuls have 3D inputs
  of the form  [g x n x m ] i.e. groups x row x colum

                                [a,b]     [b,c]
                                  |         |
                               RESHAPE   RESHAPE
    [a,b] [b,c]                   |         |
      |     |                  [1,a,b]   [1,b,c]
      |     |                       |     |
      MAT MUL      ------>          MAT MUL
         |                             |
         |                          [1,a,c]
       [a,c]                           |
                                    RESHAPE
                                       |
                                     [a,c]
 */

class MatMulPattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override;

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override;
};

// The following pattern will expand matmul(lhs/rhs)grad to a transpose and and
// a matmul Additionally it may need to add a squeeze/reduce/reshape to the
// output of the matmul to match the output of the grad op.

class MatMulGradPattern : public PreAliasPattern {
public:
  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *) const override;

  virtual popart::Tensor *getIn(Op *op) const      = 0;
  virtual popart::Tensor *getGradIn(Op *op) const  = 0;
  virtual popart::Tensor *getGradOut(Op *op) const = 0;

  virtual InIndex getInIndex() const     = 0;
  virtual InIndex getGradInIndex() const = 0;
};

class MatMulLhsGradPattern : public MatMulGradPattern {
public:
  bool matches(Op *op) const override;

  virtual popart::Tensor *getIn(Op *op) const override {
    return op->inTensor(MatMulLhsGradOp::getRhsInIndex());
  }
  virtual popart::Tensor *getGradIn(Op *op) const override {
    return op->inTensor(MatMulLhsGradOp::getGradInIndex());
  }
  virtual popart::Tensor *getGradOut(Op *op) const override {
    return op->outTensor(MatMulLhsGradOp::getOutIndex());
  }

  virtual InIndex getInIndex() const override {
    return MatMulOp::getRhsInIndex();
  }
  virtual InIndex getGradInIndex() const override {
    return MatMulOp::getLhsInIndex();
  }
};

class MatMulRhsGradPattern : public MatMulGradPattern {
public:
  bool matches(Op *op) const override;

  virtual popart::Tensor *getIn(Op *op) const override {
    return op->inTensor(MatMulRhsGradOp::getLhsInIndex());
  }
  virtual popart::Tensor *getGradIn(Op *op) const override {
    return op->inTensor(MatMulRhsGradOp::getGradInIndex());
  }
  virtual popart::Tensor *getGradOut(Op *op) const override {
    return op->outTensor(MatMulRhsGradOp::getOutIndex());
  }

  virtual InIndex getInIndex() const override {
    return MatMulOp::getLhsInIndex();
  }
  virtual InIndex getGradInIndex() const override {
    return MatMulOp::getRhsInIndex();
  }
};

} // namespace popart

#endif
