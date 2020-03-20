// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GEMM_HPP
#define GUARD_NEURALNET_GEMM_HPP

#include <popart/op.hpp>

namespace popart {

// out = alpha * transA(A) * transB(B) + beta * C
class GemmOp : public Op {
public:
  GemmOp(const OperatorIdentifier &_opid,
         float alpha_,
         float beta_,
         bool transA_,
         bool transB_,
         bool broadcast_,
         const Op::Settings &);

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

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  float alpha = 1.0;
  float beta  = 1.0;
  bool transA = false;
  bool transB = false;

  // broadcast in defined in version 6 of the op. If version 6 we should adhere
  // to the value of the attribute or throw an exception that we do not support
  // it. (T6328)
  bool broadcast = false;

  Shape getOutputShape();
};

} // namespace popart

#endif
