#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <neuralnet/graph.hpp>
#include <neuralnet/names.hpp>
#include <neuralnet/receptive.hpp>

namespace neuralnet {

class Tensor;

class AveragePoolOp : public HasReceptiveFieldOp {
public:
  AveragePoolOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual OpsAndIndices getGradOps() const override final;

private:
  virtual void setup0() override final;
  virtual void setSpatial() override final;
  int64_t getNOutChans() const override final;
};

class AveragePoolGradOp : public Op {

public:
  AveragePoolGradOp(const AveragePoolOp *);

private:
  const AveragePoolOp *averagePoolOp;
};

} // namespace neuralnet

#endif
