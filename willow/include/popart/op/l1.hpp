// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_L1_HPP
#define GUARD_NEURALNET_L1_HPP

#include <popart/op.hpp>
#include <popart/op/loss.hpp>

namespace popart {

class L1Loss : public Loss {
public:
  // where lambda*|"input"|_1 = "output" (so output has rank 0)
  L1Loss(TensorId input, TensorId output, float lambda, ReductionType rt);
  // There are no tensors streamed into this loss layer (unlike NLL for
  // example which has a label streamed in)
  std::vector<TensorId> getStreamTensorNames() const final;
  std::unique_ptr<Op> getOp(const Op::Settings &settings_) const final;
  const OperatorIdentifier &op_type() const final;
  TensorId getInputId() const;

  float getLambda() const;
  std::unique_ptr<Loss> clone() const final {
    return std::unique_ptr<Loss>(new L1Loss(*this));
  }

private:
  float lambda;
};

class L1Op : public LossOp {
public:
  L1Op(const OperatorIdentifier &_opid,
       const L1Loss *l1loss,
       const Op::Settings &settings_);
  L1Op(const OperatorIdentifier &_opid,
       const float lambda_,
       const ReductionType reduction_,
       const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  float getLambda() const { return lambda; }
  ReductionType getReductionType() const { return reduction; }

private:
  float lambda;
  ReductionType reduction;
};

class L1GradOp : public Op {

public:
  L1GradOp(const L1Op &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getInIndex() { return 0; }
  static InIndex getLossScalingInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  float getLambda() const { return lambda; }
  ReductionType getReductionType() const { return reduction; }

private:
  float lambda;
  ReductionType reduction;
};

} // namespace popart

#endif
