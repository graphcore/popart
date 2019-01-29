#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Currently only support dropout in testing mode in which case it
// becomes an identity, so this op does not have any grad op's

// TODO : T6625 Add support for Dropout  in training

class DropoutOp : public Op {
public:
  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup();

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  float ratio;
};

} // namespace poponnx

#endif
