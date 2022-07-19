// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ARGMIN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ARGMIN_HPP_

#include <memory>
#include <popart/op/argextrema.hpp>

namespace popart {
class Op;

class ArgMinOp : public ArgExtremaOp {
public:
  using ArgExtremaOp::ArgExtremaOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ARGMIN_HPP_
