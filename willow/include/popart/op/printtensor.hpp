#ifndef GUARD_NEURALNET_PRINTTENSOR_HPP
#define GUARD_NEURALNET_PRINTTENSOR_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class PrintTensorOp : public ElementWiseUnaryOp {
public:
  PrintTensorOp(const OperatorIdentifier &,
                bool printSelf,
                bool printGradient,
                const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void appendOutlineAttributes(OpSerialiserBase &os) const final;
  bool canBeReplacedByIdentity() final { return !printSelf; }

  bool shouldPrint() const { return printSelf; }

private:
  bool printSelf;
  bool printGradient;
};

} // namespace popart

#endif
