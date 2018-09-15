#ifndef GUARD_NEURALNET_NEGLOGLIKELIHOOD_AKA_NLL_HPP
#define GUARD_NEURALNET_NEGLOGLIKELIHOOD_AKA_NLL_HPP

#include <neuralnet/graph.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// Used for defining onnx Nodes
#include <onnx/defs/schema.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace neuralnet {

onnx::OpSchema createNegLogLikeOpSchema();
const onnx::OpSchema &getNegLogLikeOpSchema();

class NegLogLikeOp : public Op {
public:
  NegLogLikeOp(const OpConstructorBundle &);
  virtual void setup() override final;
};

class NegLogLikeLoss : public Loss {
public:
  // takes in of a tensor to apply NLL to
  // (the pre-soft-max tensor)
  // and the label Tensor
  NegLogLikeLoss(TensorId X_, TensorId Y_);
  // determine X from the onnx model
  NegLogLikeLoss(const onnx::ModelProto &, TensorId Y_);
  virtual ~NegLogLikeLoss() override = default;
  virtual std::unique_ptr<Op> getOp() const override final;
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  virtual TensorId getLossId() const override final;
  virtual std::string op_type() const override final;

private:
  // The tensor on which the loss is applied,
  TensorId X;
  // The correct label,
  TensorId Y;
  virtual void setInOut(std::vector<TensorId> &,
                        std::vector<TensorId> &) const override final;
};

} // namespace neuralnet

#endif
