#ifndef GUARD_NEURALNET_LOG_HPP
#define GUARD_NEURALNET_LOG_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class LogOp : public ElementWiseUnaryOp {
public:
  LogOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class LogGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  LogGradOp(LogOp *fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
