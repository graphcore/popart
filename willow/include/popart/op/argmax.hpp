#ifndef GUARD_NEURALNET_ARGMAX_HPP
#define GUARD_NEURALNET_ARGMAX_HPP

#include <popart/op/argextrema.hpp>

namespace popart {

class ArgMaxOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
