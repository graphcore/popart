#ifndef GUARD_NEURALNET_GEMM_HPP
#define GUARD_NEURALNET_GEMM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// out = alpha * transA(A) * transB(B) + beta * C
class GemmOp : public Op {
public:
  GemmOp(const OperatorIdentifier &_opid,
         Ir *_ir,
         const std::string &name = "",
         const Attributes &_attr = {});

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  float getAlpha() const;
  float getBeta() const;
  bool getTransA() const;
  bool getTransB() const;

  static InIndex getAInIndex() { return 0; }
  static InIndex getBInIndex() { return 1; }
  static InIndex getCInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

private:
  float alpha = 1.0;
  float beta  = 1.0;
  bool transA = false;
  bool transB = false;

  Shape getOutputShape();
};

} // namespace poponnx

#endif
