#ifndef GUARD_NEURALNET_CONSTEXPRS_ADDCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_ADDCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprAdd : public ConstExprOp {
public:
  ConstExprAdd(const onnx::NodeProto &n, Ir *i) : ConstExprOp(n, i) {}
  void insertOutput() final;
};

} // namespace poponnx

#endif
