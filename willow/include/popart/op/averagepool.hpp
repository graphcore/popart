#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/receptive.hpp>

namespace popart {

// c++ note : the conditions are suitable here
// for the compiler to generate defaults for
// "the 3": destructor, copy constructor, assigment op.
class AveragePoolOp : public HasReceptiveFieldOp {
public:
  AveragePoolOp(const OperatorIdentifier &_opid,
                int64_t _countIncludePad,
                int64_t _ceilMode,
                const std::vector<int64_t> &_kernelShape,
                const HasReceptiveFieldOp::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool canBeReplacedByIdentity() override;

private:
  void setup0() final;
  void setSpatialK() final;

  std::vector<int64_t> kernelShape;
  int64_t countIncludePad;
  int64_t ceilMode;
};

class AveragePoolGradOp : public Op {
public:
  AveragePoolGradOp(const AveragePoolOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;
  // Op for computing the gradient of the pre-pooled activations.
  // In theory, all we need to do this is the gradient of the
  // pooled activations. But we are requiring that all 3 of,
  //   - activations before pooling,
  //   - activations after pooling, and
  //   - gradient of activations after pooling, are inputs.
  // The reason for connecting to all 3 of these is that the
  // poplibs API requires all them.
  // We MUST provide an alternative as this is
  // kind of a bug in the poplibs API (see T5079), any optimised
  // backend will want just 1 input (gradient of pooling output)
  static InIndex getPrePooledInIndex() { return 0; }
  static InIndex getPooledInIndex() { return 1; }
  static InIndex getGradPooledInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  const AveragePoolOp *getCloneOfCreator() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

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

} // namespace popart

#endif
