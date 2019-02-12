#ifndef GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprUnsqueeze : public ConstExprOp {
public:
  ConstExprUnsqueeze(const onnx::NodeProto &n, Ir *i);
  void insertOutput() final;
};

} // namespace poponnx

#endif
