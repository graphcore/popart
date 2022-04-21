// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP

#include <popart/ces/constexpr.hpp>
#include <popart/op/slice.hpp>

namespace popart {

class ConstExprSlice : public ConstExprOp {
public:
  ConstExprSlice(Op *);
  std::vector<char> compute() final;

private:
  std::vector<Slice> getAllSlices();
};

} // namespace popart

#endif
