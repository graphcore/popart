#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <neuralnet/receptive.hpp>

namespace neuralnet {

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);
  int64_t nOutChans;
  int64_t group;

private:
  virtual int64_t getNOutChans() const override final;
  virtual void setup0() override final;
  virtual void setSpatial() override final;
};

} // namespace neuralnet

#endif
