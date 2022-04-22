// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;
struct Slice;

class ConstExprSlice : public ConstExprOp {
public:
  ConstExprSlice(Op *);
  std::vector<char> compute() final;

private:
  std::vector<Slice> getAllSlices();
};

} // namespace popart

#endif
