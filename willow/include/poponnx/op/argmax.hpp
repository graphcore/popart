#ifndef GUARD_NEURALNET_ARGMAX_HPP
#define GUARD_NEURALNET_ARGMAX_HPP

#include <poponnx/op/argextrema.hpp>

namespace poponnx {

class ArgMaxOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
