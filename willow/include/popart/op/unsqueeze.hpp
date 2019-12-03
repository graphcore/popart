#ifndef GUARD_NEURALNET_UNSQUEEZE_HPP
#define GUARD_NEURALNET_UNSQUEEZE_HPP

#include <popart/op.hpp>

namespace popart {

class UnsqueezeOp : public Op {
public:
  UnsqueezeOp(const OperatorIdentifier &_opid,
              const std::vector<int64_t> &axes_,
              const Op::Settings &settings_);
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::vector<int64_t> axes;
};

class UnsqueezeGradOp : public Op {
public:
  UnsqueezeGradOp(const UnsqueezeOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // The shape and type of the input to the constructing forward op
  TensorInfo squeezedInfo;
};

} // namespace popart

#endif
