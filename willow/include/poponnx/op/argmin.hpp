#ifndef GUARD_NEURALNET_ARGMIN_HPP
#define GUARD_NEURALNET_ARGMIN_HPP

#include <poponnx/op/argextrema.hpp>

namespace poponnx {

class ArgMinOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
