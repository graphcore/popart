// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_UPSAMPLE_HPP
#define GUARD_NEURALNET_UPSAMPLE_HPP

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

enum class UpsampleMode { Nearest, Linear, N };
std::string toString(const UpsampleMode &);
std::ostream &operator<<(std::ostream &, const UpsampleMode &);

class UpsampleOp : public Op {
public:
  UpsampleOp(const OperatorIdentifier &,
             const Op::Settings &,
             UpsampleMode,
             const std::vector<float> &scales);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void connectInTensor(InIndex inIndex, TensorId tenId) final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  UpsampleMode getMode() const { return mode; }
  const std::vector<float> &getScales() const { return scales; }

private:
  const std::vector<float> scales;
  const UpsampleMode mode;
};

} // namespace popart

#endif
