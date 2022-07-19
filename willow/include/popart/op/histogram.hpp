// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_HISTOGRAM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_HISTOGRAM_HPP_

#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// This Op gathers a histogram representing the statistics of an input tensor.
// It sorts each element of the input tensor into bins, the edges of which are
// specified by 'levels' attributed, and returns a tensor containing the
// counts in each bin.
//
// e.g. take the 1D input {-10, -0.1, 0.01, 0.09, 1.1, 4, 6.9, 7, 8.0, 900}
// and the levels {0.1, 3.1, 7}. The bins and their counts are as follows:
// x < 0.1        : 4
// 0.1 <= x < 3.1 : 1
// 3.1 <= x < 7   : 2
// x >= 7         : 3
//
// So the output will be {4, 1, 2, 3}.
//
// If the 'absoluteOfInput' attribute is set to 'true', then the same binning is
// applied to {10, 0.1, 0.01, 0.09, 1.1, 4, 6.9, 7, 8.0, 900}, in which case
// the output will be {2, 2, 2, 4}.

class HistogramOp : public Op {
public:
  HistogramOp(const OperatorIdentifier &_opid,
              const std::vector<float> &levels_,
              const bool absoluteOfInput_,
              const Op::Settings &settings_);

  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::unique_ptr<Op> clone() const override;

  std::vector<float> getLevels() const { return levels; }

  bool getAbsoluteOfInput() const { return absoluteOfInput; }

private:
  // The histogram's bin edges
  std::vector<float> levels;

  // If true, the absolute value of each input is calculated before
  // comparison with levels.
  bool absoluteOfInput;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_HISTOGRAM_HPP_
