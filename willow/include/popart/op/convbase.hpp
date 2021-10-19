// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTICONVBASE_HPP
#define GUARD_NEURALNET_MULTICONVBASE_HPP

#include <cmath>

#include <popart/op.hpp>
#include <popart/op/receptive.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

// The detailed conv parameters at the ir level
struct ConvParameters {

  DataType type;
  int64_t batchSize;

  int64_t numInChannelsPerGroup;
  int64_t numOutChannelsPerGroup;
  int64_t numGroups;

  Shape inputShape;
  Shape kernelShape;

  struct Input {
    std::vector<int64_t> lowerTruncation;
    std::vector<int64_t> upperTruncation;
    std::vector<int64_t> dilation;
    std::vector<int64_t> lowerPadding;
    std::vector<int64_t> upperPadding;
    std::vector<bool> flip;
  } inputTransformation, kernelTransformation;

  struct Output {
    std::vector<int64_t> lowerTruncation;
    std::vector<int64_t> upperTruncation;
    std::vector<int64_t> stride;
    std::vector<int64_t> lowerPadding;
    std::vector<int64_t> upperPadding;
  } outputTransformation;
};

std::ostream &operator<<(std::ostream &os, const ConvParameters::Input &input);
std::ostream &operator<<(std::ostream &os, const ConvParameters::Output &input);
std::ostream &operator<<(std::ostream &os, const ConvParameters &params);

inline bool operator==(const ConvParameters::Input &a,
                       const ConvParameters::Input &b) {
  return std::tie(a.lowerTruncation,
                  a.upperTruncation,
                  a.dilation,
                  a.lowerPadding,
                  a.upperPadding,
                  a.flip) == std::tie(b.lowerTruncation,
                                      b.upperTruncation,
                                      b.dilation,
                                      b.lowerPadding,
                                      b.upperPadding,
                                      b.flip);
}

inline bool operator!=(const ConvParameters::Input &a,
                       const ConvParameters::Input &b) {
  return !(a == b);
}

inline bool operator==(const ConvParameters::Output &a,
                       const ConvParameters::Output &b) {
  return std::tie(a.lowerTruncation,
                  a.upperTruncation,
                  a.stride,
                  a.lowerPadding,
                  a.upperPadding) == std::tie(b.lowerTruncation,
                                              b.upperTruncation,
                                              b.stride,
                                              b.lowerPadding,
                                              b.upperPadding);
}

inline bool operator!=(const ConvParameters::Output &a,
                       const ConvParameters::Output &b) {
  return !(a == b);
}

inline bool operator==(const ConvParameters &a, const ConvParameters &b) {
  return std::tie(a.type,
                  a.batchSize,
                  a.numInChannelsPerGroup,
                  a.numOutChannelsPerGroup,
                  a.numGroups,
                  a.inputShape,
                  a.kernelShape,
                  a.inputTransformation,
                  a.kernelTransformation,
                  a.outputTransformation) == std::tie(b.type,
                                                      b.batchSize,
                                                      b.numInChannelsPerGroup,
                                                      b.numOutChannelsPerGroup,
                                                      b.numGroups,
                                                      b.inputShape,
                                                      b.kernelShape,
                                                      b.inputTransformation,
                                                      b.kernelTransformation,
                                                      b.outputTransformation);
}

inline bool operator!=(const ConvParameters &a, const ConvParameters &b) {
  return !(a == b);
}

// The user-options that control the performance of the convolution
class MultiConvOptions {
public:
  // ConvOptions();
  MultiConvOptions(const std::map<std::string, std::string> sessionConvOptions,
                   const Attributes &attr);
  std::map<std::string, std::string> getConvOptions(int convIndex) const;
  std::map<std::string, std::string> getGlobalOptions() const;

  // Per-conv options
  std::vector<float> availableMemoryProportions;
  std::vector<std::string> partialsTypes;

  // Global options
  nonstd::optional<std::string> planType;
  nonstd::optional<int> perConvReservedTiles;
  nonstd::optional<float> cycleBackOff;
  std::vector<int64_t> enableConvDithering;
};

class MultiConvBaseOp : public Op {
public:
  MultiConvBaseOp(const OperatorIdentifier &_opid,
                  const Op::Settings &settings_,
                  std::vector<int64_t> flatStrides_,
                  std::vector<int64_t> flatPads_,
                  std::vector<int64_t> flatDilations_,
                  const AutoPad &padType_,
                  const MultiConvOptions &convOpts_);

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  static void appendConvParameterAttributes(const ConvParameters &,
                                            const std::string &,
                                            OpSerialiserBase &);
  void setup() override;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  static InIndex getDataInIndex(int convIndex) { return 2 * convIndex; }
  static InIndex getWeightsInIndex(int convIndex) {
    return (2 * convIndex) + 1;
  }
  static OutIndex getOutIndex(int convIndex) { return convIndex; }
  static int getConvIndexFromInIndex(InIndex index) {
    return static_cast<int>(std::floor(index / 2));
  }

  virtual int numConvs() const { return static_cast<int>(inTensorCount() / 2); }
  int64_t getNSpatialDims(int convIndex) const {
    return inRank(getDataInIndex(convIndex)) - 2;
  }
  Shape getSpatialD(int convIndex) const {
    Shape dataShape = inShape(getDataInIndex(convIndex));
    return {dataShape.begin() + 2, dataShape.end()};
  }
  Shape getSpatialK(int convIndex) const {
    Shape kernelShape = inShape(getWeightsInIndex(convIndex));
    return {kernelShape.begin() + 2, kernelShape.end()};
  }
  int64_t getGroups(int convIndex) const {
    return inInfo(getDataInIndex(convIndex)).dim(1) /
           inInfo(getWeightsInIndex(convIndex)).dim(1);
  }
  int64_t getNOutChans(int convIndex) const {
    return inInfo(getWeightsInIndex(convIndex)).dim(0);
  }
  int64_t getNInChans(int convIndex) const {
    return inInfo(getDataInIndex(convIndex)).dim(1);
  }
  Shape lowerPads(int convIndex) const {
    return HasReceptiveFieldOp::lowerPads(
        getPads(convIndex),
        static_cast<int>(getNSpatialDims(convIndex)),
        padType);
  }
  Shape upperPads(int convIndex) const {
    return HasReceptiveFieldOp::upperPads(
        getPads(convIndex),
        static_cast<int>(getNSpatialDims(convIndex)),
        padType);
  }
  Shape lowerOutPads(int convIndex) const {
    return HasReceptiveFieldOp::lowerPads(
        getOutPads(convIndex),
        static_cast<int>(getNSpatialDims(convIndex)),
        AutoPad::NOTSET);
  }
  Shape upperOutPads(int convIndex) const {
    return HasReceptiveFieldOp::upperPads(
        getOutPads(convIndex),
        static_cast<int>(getNSpatialDims(convIndex)),
        AutoPad::NOTSET);
  }
  Shape getOutShape(int convIndex, const ConvPads &pads) const;

  // Conv parameters, packaged into a single struct
  ConvParameters getParameters(int convIndex) const;

  virtual void setParamsFromDataGradOp(const Op *dataGradOp);
  virtual void restoreAttributesFromParams(const std::vector<ConvParameters> &);
  const MultiConvOptions &getConvOptions() const { return convOpts; }
  void setConvOptions(const MultiConvOptions &opts) { convOpts = opts; }

  int64_t getCumulativeSpatialDims(int64_t i) const;

  ConvStrides getStrides(int64_t convIndex) const;
  ConvPads getPads(int64_t convIndex) const;
  ConvPads getOutPads(int64_t convIndex) const;
  ConvDilations getDilations(int64_t convIndex) const;

  // Used for ConvTranspose
  ConvDilations getInDilations(int64_t convIndex) const;

  // Usually all zero but can be set by restoreAttributesFromParams
  Shape lowerKernTruncs(int64_t convIndex) const;
  Shape upperKernTruncs(int64_t convIndex) const;
  Shape lowerInTruncs(int64_t convIndex) const;
  Shape upperInTruncs(int64_t convIndex) const;
  Shape lowerOutTruncs(int64_t convIndex) const;
  Shape upperOutTruncs(int64_t convIndex) const;

private:
  void checkParameters() const;

  // The options are "flat" versions of ConvParameters for multiple
  // convolutions as follows:
  // Sort priority:
  // 1. convIndex ascending
  // 2. lower than upper (padding and truncation only)
  // 3. dimension ascending
  // (i.e. the value for dimensions will be back-to-back in the vector)

  // Directly passed in from onnx model attributes
  ConvStrides flatStrides;
  ConvPads flatPads;
  ConvPads flatOutPads;
  ConvDilations flatDilations;
  ConvDilations flatInDilations;

  // Allows for the kernel to be truncated. Needed for the gradient in some
  // cases e.g. where the kernel size combined with stride and dilation, as well
  // as the input size, means that parts of the kernel are not used.
  ConvTruncs flatKernTruncs;

  // Allows the input dims and output to be directly truncated
  // Needed for some cases of restoreAttributesFromParams such as to calculate
  // the gradient of a convolution with (zeros) padding > kernel_size
  ConvTruncs flatInTruncs;
  ConvTruncs flatOutTruncs;

  // Encapsulates per-conv and global options
  MultiConvOptions convOpts;

  AutoPad padType;
};

class MultiConvWeightsGradBaseOp : public Op {
public:
  MultiConvWeightsGradBaseOp(const MultiConvBaseOp &,
                             const OperatorIdentifier &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final {
    return inInfo;
  }
  const std::map<int, int> &gradOutToNonGradIn() const final {
    return gradOutInfo;
  }
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // The input indices where the gradients of
  // the output of convolutions are inserted to this Op
  static InIndex getGradConvolvedInIndex(int convIndex) {
    return 2 * convIndex;
  }
  // The input indices where the inputs to the
  // convolutions are inserted to this Op
  static InIndex getPreConvolvedInIndex(int convIndex) {
    return (2 * convIndex) + 1;
  }
  static OutIndex getOutIndex(int convIndex) { return convIndex; }

  virtual int numConvs() const { return static_cast<int>(inTensorCount() / 2); }
  const ConvParameters &getParameters(int convIndex) const {
    return params[convIndex];
  }
  const MultiConvOptions &getConvOptions() const { return convOpts; }

private:
  std::vector<GradInOutMapper> inInfo;
  std::map<int, int> gradOutInfo;
  std::vector<TensorInfo> weightsInfo;

  // Per conv
  std::vector<ConvParameters> params;

  // Encapsulates per-conv and global options
  MultiConvOptions convOpts;
};

class MultiConvDataGradBaseOp : public Op {
public:
  MultiConvDataGradBaseOp(const MultiConvBaseOp &, const OperatorIdentifier &);
  void setup() final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const std::vector<GradInOutMapper> &gradInputInfo() const final {
    return inInfo;
  }
  const std::map<int, int> &gradOutToNonGradIn() const final {
    return gradOutInfo;
  }

  // The input indices where the weight tensors are inserted
  static InIndex getWeightsInIndex(int convIndex) { return 2 * convIndex; }
  // The input indices where the gradient of the outputs are inserted
  static InIndex getGradConvolvedInIndex(int convIndex) {
    return (2 * convIndex) + 1;
  }
  static OutIndex getOutIndex(int convIndex) { return convIndex; }
  const ConvParameters &getParameters(int convIndex) const {
    return params[convIndex];
  }
  virtual int numConvs() const { return static_cast<int>(inTensorCount() / 2); }

  const MultiConvOptions &getConvOptions() const { return convOpts; }
  void setConvOptions(const MultiConvOptions &opts) { convOpts = opts; }
  TensorInfo getDataInfo(int convIndex) const { return dataInfo[convIndex]; }

private:
  std::vector<GradInOutMapper> inInfo;
  std::map<int, int> gradOutInfo;
  std::vector<TensorInfo> dataInfo;

  // Per conv
  std::vector<ConvParameters> params;

  // Encapsulates per-conv and global options
  MultiConvOptions convOpts;
};

} // namespace popart

#endif
