// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <string>
#include <utility>
#include <popart/op.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/tensor.hpp>

#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

/* NOTE:
 * Invalid pointers are undefined behaviour, thus
 * POPART_STRICT_COMPARATOR_CHECKS does not guarantee to throw on invalid
 * pointers
 */

bool POpCmp::operator()(const Op *a, const Op *b) const {
#ifdef POPART_STRICT_COMPARATOR_CHECKS
  if (a == nullptr || b == nullptr) {
    throw internal_error("[POpCmp] Invalid pointer.");
  }
  if (!a->getGraph().getIr().getOp(a->id)) {
    throw internal_error("[POpCmp] Invalid op {}.", a->id);
  }
  if (!b->getGraph().getIr().getOp(b->id)) {
    throw internal_error("[POpCmp] Invalid op {}.", b->id);
  }
#endif
  return a->id < b->id;
}

bool PTensorCmp::operator()(const Tensor *const &a,
                            const Tensor *const &b) const {
#ifdef POPART_STRICT_COMPARATOR_CHECKS
  if (a == nullptr || b == nullptr) {
    throw internal_error("[PTensorCmp] Invalid pointer.");
  }
  if (!a->getGraph().getIr().containsTensor(a->id)) {
    throw internal_error("[PTensorCmp] Invalid tensor {}.", a->id);
  }
  if (!b->getGraph().getIr().containsTensor(b->id)) {
    throw internal_error("[PTensorCmp] Invalid tensor {}.", b->id);
  }
#endif
  return a->id < b->id;
}

bool VectorPTensorCmp::operator()(const std::vector<Tensor *> &a,
                                  const std::vector<Tensor *> &b) const {
  return std::lexicographical_compare(
      a.begin(), a.end(), b.begin(), b.end(), PTensorCmp());
}

bool POpBoolCmp::operator()(const std::pair<Op *, bool> &a,
                            const std::pair<Op *, bool> &b) const {
#ifdef POPART_STRICT_COMPARATOR_CHECKS
  if (a.first == nullptr || b.first == nullptr) {
    throw internal_error("[POpBoolCmp] Invalid pointer.");
  }
  if (!a.first->getGraph().getIr().getOp(a.first->id)) {
    throw internal_error("[POpBoolCmp] Invalid op {}.", a.first->id);
  }
  if (!b.first->getGraph().getIr().getOp(b.first->id)) {
    throw internal_error("[POpBoolCmp] Invalid op {}.", b.first->id);
  }
#endif
  return std::pair<OpId, bool>(a.first->id, a.second) <
         std::pair<OpId, bool>(b.first->id, b.second);
}

bool POpIntCmp::operator()(std::pair<Op *, int> const &a,
                           std::pair<Op *, int> const &b) const {
#ifdef POPART_STRICT_COMPARATOR_CHECKS
  if (a.first == nullptr || b.first == nullptr) {
    throw internal_error("[POpIntCmp] Invalid pointer.");
  }
  if (!a.first->getGraph().getIr().getOp(a.first->id)) {
    throw internal_error("[POpIntCmp] Invalid op {}.", a.first->id);
  }
  if (!b.first->getGraph().getIr().getOp(b.first->id)) {
    throw internal_error("[POpIntCmp] Invalid op {}.", b.first->id);
  }
#endif
  return std::pair<OpId, int>(a.first->id, a.second) <
         std::pair<OpId, int>(b.first->id, b.second);
}

bool PICreatorCandidateCmp::operator()(const popx::ICreatorCandidate *a,
                                       const popx::ICreatorCandidate *b) const {
#ifdef POPART_STRICT_COMPARATOR_CHECKS
  if (a == nullptr || b == nullptr) {
    throw internal_error("[PICreatorCandidateCmp] Invalid pointer.");
  }
#endif
  return std::tuple<double, int64_t, int64_t>(a->getMaxCreatorPriority(),
                                              a->getNumElems(),
                                              b->getScheduleIndex()) >
         std::tuple<double, int64_t, int64_t>(b->getMaxCreatorPriority(),
                                              b->getNumElems(),
                                              a->getScheduleIndex());
}

bool ICreatorCandidatePtrCmp::operator()(
    const popx::ICreatorCandidatePtr a,
    const popx::ICreatorCandidatePtr b) const {
  PICreatorCandidateCmp cmp;
  return cmp(a.get(), b.get());
}

} // namespace popart
