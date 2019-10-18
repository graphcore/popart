#ifndef GUARD_NEURALNET_SQUEEZE_HPP
#define GUARD_NEURALNET_SQUEEZE_HPP

#include <popart/op.hpp>

namespace popart {

class SqueezeBaseOp : public Op {
public:
  SqueezeBaseOp(const OperatorIdentifier &_opid,
                const std::vector<int64_t> &axes_,
                const Op::Settings &settings_);

  void setup() final;

  void setAxes(const std::vector<int64_t> &value) { axes = value; }
  std::vector<int64_t> getAxes() const { return axes; }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // currently these are conservative TODO T6973
  view::RegMap fwdRegMap(InIndex) const final;
  view::RegMap bwdRegMap(InIndex) const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendAttributes(OpSerialiserBase &) const override;

protected:
  std::vector<int64_t> axes;

  void setAxesToDefault();
};

class SqueezeOp : public SqueezeBaseOp {
public:
  SqueezeOp(const OperatorIdentifier &_opid,
            const std::vector<int64_t> &axes_,
            const Op::Settings &settings_);
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::unique_ptr<Op> clone() const final;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
};

class SqueezeInplaceOp : public SqueezeBaseOp {
public:
  SqueezeInplaceOp(const SqueezeOp &);
  std::unique_ptr<Op> clone() const final;
};

class SqueezeGradOp : public Op {
public:
  SqueezeGradOp(const SqueezeOp &);

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // The shape and type of the input to the constructing forward op
  TensorInfo unsqueezedInfo;
};

} // namespace popart

#endif
