// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/op.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>

namespace popart {

bool POpCmp::operator()(const Op *a, const Op *b) const {
  return a->id < b->id;
}

bool PTensorCmp::operator()(const Tensor *const &a,
                            const Tensor *const &b) const {
  return a->id < b->id;
}

bool POpBoolCmp::operator()(const std::pair<Op *, bool> &a,
                            const std::pair<Op *, bool> &b) const {
  return std::pair<OpId, bool>(a.first->id, a.second) <
         std::pair<OpId, bool>(b.first->id, b.second);
}

bool POpIntCmp::operator()(std::pair<Op *, int> const &a,
                           std::pair<Op *, int> const &b) const {
  return std::pair<OpId, int>(a.first->id, a.second) <
         std::pair<OpId, int>(b.first->id, b.second);
}

} // namespace popart
