// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CONVTRANSPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CONVTRANSPOSE_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op/convbase.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/receptive.hpp"

namespace popart {
struct OperatorIdentifier;

class ConvTransposeOp : public Op {
public:
  ConvTransposeOp(const OperatorIdentifier &_opid,
                  const Settings &settings_,
                  std::vector<int64_t> strides,
                  std::vector<int64_t> pads,
                  std::vector<int64_t> dilations,
                  int64_t group,
                  const AutoPad &padType,
                  std::vector<int64_t> outputPadding,
                  Shape outputShape,
                  const MultiConvOptions &convOpts);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getLog2ScaleInIndex() { return 2; }

  static OutIndex getOutIndex() { return 0; }

  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  int64_t group;
  const AutoPad padType;
  const MultiConvOptions convOpts;
  ConvParameters params;

  bool isPow2ScaledConvTranspose() const;

  std::set<InIndex> optionalInputs() const override {
    return {getLog2ScaleInIndex()};
  }

private:
  std::vector<int64_t> pads;
  std::vector<int64_t> outputPadding;
  std::vector<int64_t> outputShape;

  void setParams(const std::vector<int64_t> &lowerPadding,
                 const std::vector<int64_t> &upperPadding);
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CONVTRANSPOSE_HPP_
