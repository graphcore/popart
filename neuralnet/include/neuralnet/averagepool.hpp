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
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;

private:
  virtual void setup0() override final;
  virtual void setSpatial() override final;
  int64_t getNOutChans() const override final;
  virtual bool readyToCreateGradients(std::set<int> &) const override final;
};

class AveragePoolGradOp : public Op {

public:
  AveragePoolGradOp(AveragePoolOp *);
  virtual const std::vector<GradInOutMapper> & gradInputInfo() const override final;
  virtual const std::map<int, int> & gradOutToNonGradIn() const override final;
  virtual Op * getNonGradOp() override final;
  virtual const std::map<int, Tensor *> & gradOutMap() override final;
  virtual int getNonGradInIndex(int gradOpOutIndex) const override final;




private:
  AveragePoolOp *averagePoolOp;
  std::vector<GradInOutMapper> createGradInputInfo() const;
  std::map<int, int> createGradOutToNonGradInInfo() const;
};

} // namespace neuralnet

#endif
