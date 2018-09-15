#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <neuralnet/graph.hpp>
#include <neuralnet/receptive.hpp>

namespace neuralnet {

class AveragePoolOp : public HasReceptiveFieldOp {
public:
  AveragePoolOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);

private:
  virtual void setup0() override final;
  virtual void setSpatial() override final;
  int64_t getNOutChans() const override final;
};
} // namespace neuralnet

#endif
