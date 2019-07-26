#ifndef GUARD_NEURALNET_ARGMIN_HPP
#define GUARD_NEURALNET_ARGMIN_HPP

#include <popart/op/argextrema.hpp>

namespace popart {

class ArgMinOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
