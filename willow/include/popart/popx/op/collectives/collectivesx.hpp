// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVESX_HPP
#define GUARD_NEURALNET_COLLECTIVESX_HPP

#include <popart/debugcontext.hpp>
#include <popart/names.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/popx/viewchangers.hpp>

#include <popops/CollectiveTypes.hpp>

#include <gcl/CollectiveBalancedReorder.hpp>

namespace popart {
namespace popx {

popops::CollectiveOperator getPoplarCollectiveOperator(CollectiveOperator op);

struct ReorderMetadata {
  ReorderMetadata(int64_t offset_,
                  int64_t rearranged_offset_,
                  int64_t size_,
                  int64_t tile_)
      : offset(offset_), rearranged_offset(rearranged_offset_), size(size_),
        tile(tile_) {}
  int64_t offset;
  int64_t rearranged_offset;
  int64_t size;
  int64_t tile;
};

// If the input/output to a collective op is padded,
// provide a cut-down tensor that fits IR specifications of that tensor
// Note: The cut-down tensor may not contain all regions of the tensor that are
// carrying data, and is therefore not suited to be operated on and only
// serves to provide an IR-compatible view of the tensor
class ReplicatedGatherInScatterOutViewChanger : public ViewChanger {
public:
  ReplicatedGatherInScatterOutViewChanger(int64_t nelms_,
                                          const std::set<TensorId> group_)
      : nelms(nelms_), group(group_) {}
  snap::Tensor apply(snap::Tensor tensor) const final {
    return tensor.slice(0, nelms, 0);
  }
  bool containsAllDataRegions() const final { return false; }
  bool operator==(const ViewChanger &rhs) const final {
    if (const ReplicatedGatherInScatterOutViewChanger *other =
            dynamic_cast<const ReplicatedGatherInScatterOutViewChanger *>(
                &rhs)) {
      return group == other->group;
    }
    return false;
  }

private:
  int64_t nelms;
  std::set<TensorId> group;
};

// If the (tile-balanced) input/output to a collective op is rearranged and/or
// padded, provide a view of the tenosr data that matches the IR expectations
// of the tensor.
// Note: The view is suitable for consumption by all ops, as all data carrying
// regions are included in the view and arranged correctly
class ReplicatedGatherOutScatterInViewChanger : public ViewChanger {
public:
  ReplicatedGatherOutScatterInViewChanger(
      const gcl::CollectiveBalancedReorder *cbr_,
      const std::set<TensorId> group_)
      : cbr(cbr_), group(group_) {}
  snap::Tensor apply(snap::Tensor tensor) const final {
    return snap::Tensor{
        cbr->undoRearrangeForCollective(tensor.getPoplarTensor())
            .reshape(cbr->getReferenceShape()),
        tensor};
  }
  bool operator==(const ViewChanger &rhs) const final {
    if (const ReplicatedGatherOutScatterInViewChanger *other =
            dynamic_cast<const ReplicatedGatherOutScatterInViewChanger *>(
                &rhs)) {
      return group == other->group;
    }
    return false;
  }

private:
  const gcl::CollectiveBalancedReorder *cbr;
  std::set<TensorId> group;
};

class CollectivesBaseOpx : public PopOpx {
public:
  CollectivesBaseOpx(Op *, Devicex *);
  // Return all linked tensors and their connected ops to coordinate tensor
  // mapping of collective inputs and outputs
  std::pair<std::set<TensorId>, std::vector<Op *>>
  getCollectiveLinkedGroup() const;
  gcl::CollectiveBalancedReorder *getCollectiveBalancedReorder() const;
  gcl::CollectiveBalancedReorder *
  createCollectiveBalancedReorder(snap::Tensor tensor) const;
};

} // namespace popx
} // namespace popart

#endif
