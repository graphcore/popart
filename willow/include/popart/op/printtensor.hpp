// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_PRINTTENSOR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_PRINTTENSOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/elementwise.hpp>
#include <popart/printtensorfmt.hpp>

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

  PrintTensorOp(const OperatorIdentifier &,
                bool printSelf,
                bool printGradient,
                const std::string &title,
                const PrintTensorFmt &fmt,
                const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void appendOutlineAttributes(OpSerialiserBase &os) const final;
  bool canBeReplacedByIdentity() const final { return !printSelf; }
  bool hasSideEffect() const override { return true; }

  bool shouldPrint() const { return printSelf; }
  const std::string &getTitle() const { return title; }
  void setTitle(std::string title_) { title = std::move(title_); }
  const PrintTensorFmt &getFmt() const { return fmt; }

private:
  bool printSelf;
  bool printGradient;
  std::string title;
  const PrintTensorFmt fmt{};
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_PRINTTENSOR_HPP_
