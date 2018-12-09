#ifndef GUARD_NEURALNET_SUBSAMPLE_HPP
#define GUARD_NEURALNET_SUBSAMPLE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class SubsampleOp : public Op {
public:
  SubsampleOp(const OpConstructorBundle &);
  SubsampleOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // The stride is a vector whose length is the rank of the input tensor
  // If strides is defined as {1,..,1} the the input tensor will not be changed
  std::vector<uint32_t> strides_u32() const;

  // Returns true if all the strides at 1
  bool strideSizeOne() const;

public:
  std::vector<int64_t> strides;
};

class SubsampleGradOp : public Op {
public:
  SubsampleGradOp(SubsampleOp *fwdOp);
  std::unique_ptr<Op> clone() const final;
  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  SubsampleOp *getFwdOp() { return fwdOp; }

private:
  SubsampleOp *fwdOp;
  TensorInfo fwdOpInfo;
};

} // namespace poponnx

#endif
