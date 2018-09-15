#ifndef GUARD_NEURALNET_NEGLOGLIKELIHOOD_AKA_NLL_HPP
#define GUARD_NEURALNET_NEGLOGLIKELIHOOD_AKA_NLL_HPP

#include  <neuralnet/graph.hpp>


#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// Used for defining onnx Nodes
#include <onnx/defs/schema.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace neuralnet {

// if model's graph has single output, return its name,
// otherwise throw an error
TensorId getUniqueOutId(const onnx::ModelProto &m);

onnx::OpSchema createNegLogLikeOpSchema();
const onnx::OpSchema & getNegLogLikeOpSchema();

class NegLogLikeOp : public Op {
public:
  NegLogLikeOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);
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
  virtual std::vector<std::unique_ptr<Node>> getNodes() const override final;
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  static TensorId getLossId();

  private:
  // The tensor on which the loss is applied,
  TensorId X;
  // The correct label,
  TensorId Y;
};


} // namespace neuralnet

#endif
