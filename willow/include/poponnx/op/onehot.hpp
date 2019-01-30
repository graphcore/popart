#ifndef GUARD_NEURALNET_ONEHOT_HPP
#define GUARD_NEURALNET_ONEHOT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class OnehotOp : public Op {
public:
  OnehotOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() override;

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

  static InIndex getIndicesInIndex() { return 0; }
  static InIndex getDepthInIndex() { return 1; }
  static InIndex getValuesInIndex() { return 2; }

  static OutIndex getOutIndex() { return 0; }

  virtual void connectInTensor(InIndex inIndex, TensorId tenId) override;

  int64_t getAxis() const { return axis; }

private:
  int64_t axis;
  int64_t onehotAxisDim;
};

class OnehotGradOp : public Op {
public:
  OnehotGradOp(const OnehotOp &fwdOp_);
  std::unique_ptr<Op> clone() const final;

  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getIndicesInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  const Shape &getOutputShape() const { return outputShape; }

  int64_t getAxis() const { return axis; }

private:
  int64_t axis;
  Shape outputShape;
};

} // namespace poponnx

#endif
