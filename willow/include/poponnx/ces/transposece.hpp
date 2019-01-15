#ifndef GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprTranspose : public ConstExprOp {
public:
  ConstExprTranspose(const onnx::NodeProto &n, Ir *i) : ConstExprOp(n, i) {}
  void insertOutput() final;
};

} // namespace poponnx

#endif
