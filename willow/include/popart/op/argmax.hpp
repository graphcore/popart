// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ARGMAX_HPP
#define GUARD_NEURALNET_ARGMAX_HPP

#include <memory>
#include <popart/op/argextrema.hpp>

namespace popart {
class Op;

class ArgMaxOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
