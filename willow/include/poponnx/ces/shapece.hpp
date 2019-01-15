#ifndef GUARD_NEURALNET_CONSTEXPRS_SHAPECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SHAPECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprShape : public ConstExprOp {
public:
  ConstExprShape(const onnx::NodeProto &n, Ir *i) : ConstExprOp(n, i) {}
  void insertOutput() final;
};

} // namespace poponnx

#endif
