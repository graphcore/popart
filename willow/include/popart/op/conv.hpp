#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#include <popart/op/receptive.hpp>
#include <popart/util.hpp>

namespace popart {

enum class ConvPartialsType { HALF, FLOAT };

std::string toString(const ConvPartialsType &);
std::ostream &operator<<(std::ostream &, const ConvPartialsType &);

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

class ConvOp : public HasReceptiveFieldOp {
public:
  ConvOp(const OperatorIdentifier &_opid,
         int64_t group,
         const ConvPartialsType &partialsType_,
         const float &availableMemoryProportion_,
         const HasReceptiveFieldOp::Settings &settings_);
  int64_t nOutChans;
  int64_t group;
  // convenience functions:
  const Tensor *dataIn() const;
  const Tensor *weightsIn() const;
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  void setup() override;

  // from github.com/onnx/onnx/blob/master/docs/Operators.md#Conv :
  // "data" at index 0, "weights" at index 1, "bias" as index 2.
  // popart's ConvOp does not support bias, but bias can be used when
  // ConvBiasPattern is used.
  static InIndex getDataInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getBiasInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  const ConvParameters &getParameters() const { return params; }
  void setParameters(const ConvParameters &p) { params = p; }

  // Rather than overrideing the outshape for the conv when used in the bwd pass
  // it would be better to figureout the formula.
  Shape getOutShape() const override;

  void setOutputShape(const Shape &s) { outputShape = s; }

  const Shape &getInputShape() const { return inputShape; }

  const ConvPartialsType &getPartialsType() const { return partialsType; }
  void setPartialsType(const ConvPartialsType &v) { partialsType = v; }
  float getAvailableMemoryProportion() const {
    return availableMemoryProportion;
  }
  void setAvailableMemoryProportion(const float &v) {
    availableMemoryProportion = v;
  }

private:
  ConvParameters params;

  // Override the outputshape of the conv
  Shape outputShape;

  // Saved shape of the input data
  Shape inputShape;

  ConvPartialsType partialsType;
  float availableMemoryProportion;

  void setup0() final;
  void setSpatialK() final;
};

class ConvWeightsGradOp : public Op {
public:
  ConvWeightsGradOp(const ConvOp &);
  std::unique_ptr<Op> clone() const final;
  ConvWeightsGradOp(const ConvWeightsGradOp &) = default;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // The input index where the gradient of
  // the output of convolution is inserted to this Op
  static InIndex getGradConvolvedInIndex() { return 0; }

  // The input index where the input to the
  // convolution (ConvOp) is inserted to this Op
  static InIndex getPreConvolvedInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  const ConvOp *getCloneOfCreator() const;

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  std::shared_ptr<Op> cloneOfCreator;
  TensorInfo weightsInfo;
};

class ConvFlipWeightsOp : public Op {
public:
  ConvFlipWeightsOp(const OperatorIdentifier &_opid,
                    const Op::Settings &settings_);
  ConvFlipWeightsOp(const ConvFlipWeightsOp &) = default;
  ~ConvFlipWeightsOp() override;
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const ConvParameters &getParameters() const { return params; }
  void setParameters(const ConvParameters &p) { params = p; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  const ConvPartialsType &getPartialsType() const { return partialsType; }
  void setPartialsType(const ConvPartialsType &v) { partialsType = v; }
  float getAvailableMemoryProportion() const {
    return availableMemoryProportion;
  }
  void setAvailableMemoryProportion(const float &v) {
    availableMemoryProportion = v;
  }

  void appendAttributes(OpSerialiserBase &os) const final;

private:
  ConvParameters params;
  ConvPartialsType partialsType;
  float availableMemoryProportion;
};

class ConvDataGradOp : public Op {
public:
  ConvDataGradOp(const ConvOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  const ConvOp *getCloneOfCreator() const;

  // The input index where the weight tensor is inserted
  static InIndex getWeightsInIndex() { return 0; }
  // The input index where the gradient of the output is inserted
  static InIndex getGradConvolvedInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  const ConvParameters &getParameters() const { return params; }
  void setParameters(const ConvParameters &p) { params = p; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  ConvParameters params;
  std::shared_ptr<Op> cloneOfCreator;
  TensorInfo dataInfo;
};

} // namespace popart

#endif
