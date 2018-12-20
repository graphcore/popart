#ifndef GUARD_NEURALNET_SCALE_HPP
#define GUARD_NEURALNET_SCALE_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

// y = scale_factor * x
class ScaleOp : public ElementWiseUnaryOp {
public:
  ScaleOp(const OperatorIdentifier &_opid,
          Ir *_ir,
          const std::string &name = "",
          const Attributes &_attr = {});

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setScaleFactor(float value) { scale_factor = value; }
  float getScaleFactor() const;

private:
  float scale_factor;
};

class ScaleGradOp : public ScaleOp {
public:
  ScaleGradOp(ScaleOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
