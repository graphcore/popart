// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CONV_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CONV_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/convbase.hpp>
#include <popart/op/receptive.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

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

  int numConvs() const final { return 1; }
  static InIndex getDataInIndex() { return MultiConvBaseOp::getDataInIndex(0); }
  static InIndex getWeightsInIndex() {
    return MultiConvBaseOp::getWeightsInIndex(0);
  }
  static InIndex getLog2ScaleInIndex() { return getWeightsInIndex() + 1; }
  static OutIndex getOutIndex() { return MultiConvBaseOp::getOutIndex(0); }
  int64_t getGroups() const { return MultiConvBaseOp::getGroups(0); }
  void setGroup() { group = MultiConvBaseOp::getGroups(0); }
  int64_t getNInChans() const { return MultiConvBaseOp::getNInChans(0); }
  int64_t getNOutChans() const { return MultiConvBaseOp::getNOutChans(0); }
  ConvParameters getParameters() const {
    return MultiConvBaseOp::getParameters(0);
  }
  void
  restoreAttributesFromParams(const std::vector<ConvParameters> &) override;

  /**
   * Returns true if and only if the inputs to the op constitute a
   * valid set of inputs for a fused (float8) convolution.
   */
  bool isPow2ScaledConv() const;

  std::set<InIndex> optionalInputs() const override {
    return {getLog2ScaleInIndex()};
  }

private:
  void verifyPartialsTypesAreHalf() const;
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
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const ConvParameters &getParameters() const { return params; }
  void setParameters(const ConvParameters &p) { params = p; }

  bool getGroupReshape() const { return groupReshape; }
  void setGroupReshape(bool reshape) { groupReshape = reshape; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const final;

  void setConvOptions(const MultiConvOptions &opts) { convOpts = opts; }
  const MultiConvOptions &getMultiConvOptions() const { return convOpts; }
  std::map<std::string, std::string> getConvOptions() const {
    return convOpts.getConvOptions(0);
  }

private:
  // Reshape the weight such that the data grad convolution receives a weight
  // shape which is valid
  bool groupReshape;
  ConvParameters params;
  MultiConvOptions convOpts;
};

class ConvFlipWeightsGradOp : public ConvFlipWeightsOp {
public:
  ConvFlipWeightsGradOp(const ConvFlipWeightsGradOp &) = default;
  ConvFlipWeightsGradOp(const ConvFlipWeightsOp &convFlipWeightsOp);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CONV_HPP_
