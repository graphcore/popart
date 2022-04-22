// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COPYVARUPDATE_HPP
#define GUARD_NEURALNET_COPYVARUPDATE_HPP

#include <map>
#include <memory>
#include <popart/op/varupdate.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class CopyVarUpdateOp : public VarUpdateWithUpdaterOp {
public:
  CopyVarUpdateOp(const Op::Settings &);

  CopyVarUpdateOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // Modifies differs from base class since copy will
  // overwrite the tensor to update completely
  view::Regions modifies(InIndex) const override;
};

} // namespace popart

#endif
