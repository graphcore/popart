#ifndef GUARD_NEURALNET_PLACEHOLDER_HPP
#define GUARD_NEURALNET_PLACEHOLDER_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class PlaceholderOp : public Op {
public:
  PlaceholderOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace poponnx

#endif
