#ifndef GUARD_NEURALNET_GLOBALAVERAGEPOOL_HPP
#define GUARD_NEURALNET_GLOBALAVERAGEPOOL_HPP

#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/receptive.hpp>

namespace poponnx {

class GlobalAveragePoolOp : public Op {
public:
  GlobalAveragePoolOp(const OperatorIdentifier &_opid,
                      const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;

  void setup() override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  Shape getSpatialK() const { return kernel; }
  Shape getStrides() const;
  Shape getLowerPads() const;
  Shape getUpperPads() const;

private:
  Shape kernel;
};

class GlobalAveragePoolGradOp : public Op {
public:
  GlobalAveragePoolGradOp(const GlobalAveragePoolOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
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

  const GlobalAveragePoolOp *getCloneOfCreator();

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
  std::unique_ptr<Op> cloneOfCreator;
};

} // namespace poponnx

#endif
