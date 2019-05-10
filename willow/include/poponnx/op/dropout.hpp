#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Currently only support dropout in testing mode in which case it
// becomes an identity, so this op does not have any grad op's

// TODO : T8559 Add support for Dropout in training

class DropoutOp : public Op {
public:
  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() override;

  void appendAttributes(OpSerialiserBase &) const override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() override;

  // T8559 : disable outlining by default? TODO

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  float ratio;
};

} // namespace poponnx

#endif
