#ifndef GUARD_NEURALNET_MAXPOOL_HPP
#define GUARD_NEURALNET_MAXPOOL_HPP

#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/receptive.hpp>

namespace poponnx {

// c++ note : the conditions are suitable here
// for the compiler to generate defaults for
// "the 3": destructor, copy constructor, assigment op.
class MaxPoolOp : public HasReceptiveFieldOp {
public:
  MaxPoolOp(const OperatorIdentifier &_opid,
            const std::vector<int64_t> &kernelShape_,
            int64_t storageOrder,
            int64_t _ceilMode,
            const HasReceptiveFieldOp::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  void setup0() final;
  void setSpatialK() final;

  int64_t storageOrder;
  int64_t ceilMode;
  std::vector<int64_t> kernelShape;
};

class MaxPoolGradOp : public Op {
public:
  MaxPoolGradOp(const MaxPoolOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getPrePooledInIndex() { return 0; }
  static InIndex getPooledInIndex() { return 1; }
  static InIndex getGradPooledInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  const MaxPoolOp *getCloneOfCreator() const;

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // The shape and type of the input to the
  // forward op which creates this backwards op
  TensorInfo unpooledInfo;
  // A copy of the forward op which creates
  // this backwards op. Note
  // 1) backends will need a copy of this op to determine
  //    how to do the backwards pass (padding, striding, etc)
  // 2) we DON'T store a pointer to the creating forward op,
  //    which might be optimised out and deleted
  std::shared_ptr<Op> cloneOfCreator;
};

} // namespace poponnx

#endif
