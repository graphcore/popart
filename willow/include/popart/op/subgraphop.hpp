#ifndef GUARD_NEURALNET_SUBGRAPHOP_HPP
#define GUARD_NEURALNET_SUBGRAPHOP_HPP

#include <popart/op.hpp>

namespace popart {

class SubgraphOp : public Op {
public:
  // parent: Graph this CallOp belongs to
  SubgraphOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  virtual bool isInputModified(InIndex) const = 0;
};

} // namespace popart

#endif
