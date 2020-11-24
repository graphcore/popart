// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <popart/op/convbase.hpp>
#include <popart/op/receptive.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class ConvOp : public MultiConvBaseOp {
public:
  ConvOp(const OperatorIdentifier &_opid,
         const Settings &settings_,
         std::vector<int64_t> strides,
         std::vector<int64_t> pads,
         std::vector<int64_t> dilations,
         int64_t group,
         const AutoPad &padType,
         const MultiConvOptions &convOpts);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv :
  // "data" at index 0, "weights" at index 1, "bias" as index 2.
  // popart's ConvOp does not support bias, but bias can be used when
  // ConvBiasPattern is used.
  static InIndex getBiasInIndex() { return 2; }

  int numConvs() const final { return 1; }
  static InIndex getDataInIndex() { return MultiConvBaseOp::getDataInIndex(0); }
  static InIndex getWeightsInIndex() {
    return MultiConvBaseOp::getWeightsInIndex(0);
  }
  static OutIndex getOutIndex() { return MultiConvBaseOp::getOutIndex(0); }
  int64_t getGroups() { return MultiConvBaseOp::getGroups(0); }
  int64_t getNInChans() { return MultiConvBaseOp::getNInChans(0); }
  int64_t getNOutChans() { return MultiConvBaseOp::getNOutChans(0); }
  const ConvParameters &getParameters() const {
    return MultiConvBaseOp::getParameters(0);
  }

private:
  // Can always be determined by input shapes. However, we check here that
  // the user-provided value matches.
  int64_t group;
};

class ConvWeightsGradOp : public MultiConvWeightsGradBaseOp {
public:
  ConvWeightsGradOp(const ConvOp &);
  std::unique_ptr<Op> clone() const final;
  ConvWeightsGradOp(const ConvWeightsGradOp &) = default;

  int numConvs() const final { return 1; }
  static InIndex getGradConvolvedInIndex() {
    return MultiConvWeightsGradBaseOp::getGradConvolvedInIndex(0);
  }
  static InIndex getPreConvolvedInIndex() {
    return MultiConvWeightsGradBaseOp::getPreConvolvedInIndex(0);
  }
  static OutIndex getOutIndex() {
    return MultiConvWeightsGradBaseOp::getOutIndex(0);
  }
  const ConvParameters &getParameters() const {
    return MultiConvWeightsGradBaseOp::getParameters(0);
  }
};

class ConvFlipWeightsOp : public Op {
public:
  ConvFlipWeightsOp(const ConvFlipWeightsOp &) = default;
  ConvFlipWeightsOp(const OperatorIdentifier &_opid,
                    const Op::Settings &settings_);
  ~ConvFlipWeightsOp() override;
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const ConvParameters &getParameters() const { return params; }
  void setParameters(const ConvParameters &p) { params = p; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const final;

  void setConvOptions(const MultiConvOptions &opts) { convOpts = opts; }
  std::map<std::string, std::string> getConvOptions() const {
    return convOpts.getConvOptions(0);
  }

private:
  ConvParameters params;
  MultiConvOptions convOpts;
};

class ConvDataGradOp : public MultiConvDataGradBaseOp {
public:
  ConvDataGradOp(const ConvOp &);
  std::unique_ptr<Op> clone() const final;

  int numConvs() const override { return 1; }
  static InIndex getWeightsInIndex() {
    return MultiConvDataGradBaseOp::getWeightsInIndex(0);
  }
  static InIndex getGradConvolvedInIndex() {
    return MultiConvDataGradBaseOp::getGradConvolvedInIndex(0);
  }
  static OutIndex getOutIndex() {
    return MultiConvDataGradBaseOp::getOutIndex(0);
  }
  const ConvParameters &getParameters() const {
    return MultiConvDataGradBaseOp::getParameters(0);
  }
};

} // namespace popart

#endif
