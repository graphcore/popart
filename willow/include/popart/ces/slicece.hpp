// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_SLICECE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_SLICECE_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_CES_SLICECE_HPP_
