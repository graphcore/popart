#ifndef GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprConcat : public ConstExprOp {
public:
  ConstExprConcat(const onnx::NodeProto &n, Ir *i);
  void insertOutput() final;

private:
  int64_t input_count;
  int64_t axis;
};

} // namespace poponnx

#endif
