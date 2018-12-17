#ifndef GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class CastCe : public ConstExprOp {
public:
  CastCe(const onnx::NodeProto &n, Ir *i) : ConstExprOp(n, i) {}
  void insertOutput() final;
};
} // namespace poponnx

#endif
