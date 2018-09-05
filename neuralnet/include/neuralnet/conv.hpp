#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <cblas.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <neuralnet/graph.hpp>

namespace neuralnet {

class ConvOp : public Op {
public:
  ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);
  virtual void inferInfo() override final;

private:
  Attributes atts; 

  int nSpatialDims;
  int64_t batchSize, nInChans, nOutChans;

  std::vector<int64_t> dilations;
  int64_t group;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;

};

} // namespace neuralnet

#endif
