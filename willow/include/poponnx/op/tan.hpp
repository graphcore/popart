#ifndef GUARD_NEURALNET_TAN_HPP
#define GUARD_NEURALNET_TAN_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class TanOp : public ElementWiseUnaryOp {
public:
  TanOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace poponnx

#endif
