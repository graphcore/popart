// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PRINTTENSOR_HPP
#define GUARD_NEURALNET_PRINTTENSOR_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class PrintTensorOp : public ElementWiseUnaryOp {
public:
  PrintTensorOp(const OperatorIdentifier &,
                bool printSelf,
                bool printGradient,
                const std::string &title,
                const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void appendOutlineAttributes(OpSerialiserBase &os) const final;
  bool canBeReplacedByIdentity() const final { return !printSelf; }

  bool shouldPrint() const { return printSelf; }
  const std::string &getTitle() const { return title; }

private:
  bool printSelf;
  bool printGradient;
  std::string title;
};

} // namespace popart

#endif
