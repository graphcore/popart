// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTICONV_HPP
#define GUARD_NEURALNET_MULTICONV_HPP

#include <cmath>

#include <popart/op/convbase.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class MultiConvOp : public MultiConvBaseOp {
public:
  // Constructed with flat* parameters, as 'vectors of vectors' are not
  // supported onnx attributes.
  // These are unflattened based on the ranks of the input tensors
  // of each convolution that the multiconv is comprised of.
  //
  // e.g. if  conv0, with inputs of rank 3, has strides {2}
  //      and conv1, with inputs of rank 4, has strides {1, 1}
  //      strides : [[2], [1, 1]]
  //      flatStrides = [2, 1, 1]
  MultiConvOp(const OperatorIdentifier &_opid,
              const Settings &settings_,
              const std::vector<int64_t> &flatStrides_,
              const std::vector<int64_t> &flatPads_,
              const std::vector<int64_t> &flatDilations_,
              const MultiConvOptions &mcOpts_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
};

class MultiConvWeightsGradOp : public MultiConvWeightsGradBaseOp {
public:
  MultiConvWeightsGradOp(const MultiConvOp &);
  std::unique_ptr<Op> clone() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
};

class MultiConvDataGradOp : public MultiConvDataGradBaseOp {
public:
  MultiConvDataGradOp(const MultiConvOp &);
  std::unique_ptr<Op> clone() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
};

} // namespace popart

#endif
