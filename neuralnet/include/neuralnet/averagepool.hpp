#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <neuralnet/graph.hpp>
#include <neuralnet/receptive.hpp>
#include <neuralnet/names.hpp>

namespace neuralnet {

class Tensor;

class AveragePoolOp : public HasReceptiveFieldOp {
public:
  AveragePoolOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);
  virtual std::unique_ptr<Op> getGradOp(
      OpId) const override final;
      //const std::map<int, Tensor *> &gradientsIn) const override final;

private:
  virtual void setup0() override final;
  virtual void setSpatial() override final;
  int64_t getNOutChans() const override final;
};

class AveragePoolGradOp : public GradOp {

  public:
    
    AveragePoolGradOp(OpId,
                      const AveragePoolOp *);
                      //const std::map<int, Tensor *> &gradientsIn);

  private: 
    const AveragePoolOp * averagePoolOp;

};

}

#endif
