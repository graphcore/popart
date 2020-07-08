// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVESX_HPP
#define GUARD_NEURALNET_COLLECTIVESX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

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

// Helper class to reorder a tensor in a per-tile-balanced fashion such that
// each replica obtains (for inputs to AllGather or outputs of ReduceScatter)
// an equally sized 1D tensor with equally sized regions.
// The reordering process:
//  - Flattens the input tensor
//  - Analyzes the tile mapping
//  - Determines reordering strategy and required internal padding
//  - Can rearrange and undo the rearrangement on any tensor that
//    has the same tile mapping
//  - Can rearrange and undo the rearrangement on host tensors that are to be
//    copied into CBR-rearranged RemoteBuffers
class CollectiveBalancedReorder {
public:
  CollectiveBalancedReorder(poplar::Graph &graph_,
                            poplar::Tensor tensor_,
                            unsigned replicationFactor_);

  // Balanced reorder the tensor in a collective-friendly manner
  poplar::Tensor rearrangeForCollective(poplar::Tensor tensor) const;

  // Reorder tensor back into the expected IR tensor shape and order
  poplar::Tensor undoRearrangeForCollective(poplar::Tensor tensor) const;

  // Balanced reorder the tensor in a collective-friendly manner (host-side)
  void
  rearrangeForCollective(const char *in, char *out, int64_t elemByteSize) const;

  // Reorder tensor back into the expected IR tensor shape and order (host-side)
  void undoRearrangeForCollective(const char *in,
                                  char *out,
                                  int64_t elemByteSize) const;

  // Get a clone of the tensor that was used to create the CBR object
  poplar::Tensor getReferenceTensorClone(std::string name) const;

  // Get the tensor that was used to create the CBR object
  const poplar::Tensor &getReferenceTensor() const;

  // Number of elements in the collective balanced (reordered) tensor
  size_t getNumRearrangedTensorElems() const {
    return numRearrangedTensorElems;
  }

private:
  // Host tensor rearrangement routine
  void rearrange(const char *in,
                 char *out,
                 int64_t elemByteSize,
                 bool forCollective) const;

  // Graph or subgraph on which the tensor and reordered tensor are allocated
  poplar::Graph &graph;

  unsigned replicationFactor;

  poplar::Tensor referenceTensor;
  size_t numRearrangedTensorElems;

  // Tuple of: original offset, rearranged offset, size and tile
  std::vector<ReorderMetadata> reordering;

  // Proxy to simplify tensors to rearrange for collectives
  poplar::Tensor simplifyProxy;

  // Proxy to reverse simplfy tensors to rearrange for collectives
  poplar::Tensor simplifyReverseProxy;
};

// If the input/output to a collective op is padded,
// provide a cut-down tensor that fits IR specifications of that tensor
// Note: The cut-down tensor may not contain all regions of the tensor that are
// carrying data, and is therefore not suited to be operated on and only
// serves to provide an IR-compatible view of the tensor
class ReplicatedGatherInScatterOutViewChanger : public ViewChanger {
public:
  ReplicatedGatherInScatterOutViewChanger(int64_t nelms_) : nelms(nelms_) {}
  poplar::Tensor apply(poplar::Tensor tensor) const final {
    return tensor.slice(0, nelms, 0);
  }
  bool containsAllDataRegions() const final { return false; }

private:
  int64_t nelms;
};

// If the (tile-balanced) input/output to a collective op is rearranged and/or
// padded, provide a view of the tenosr data that matches the IR expectations
// of the tensor.
// Note: The view is suitable for consumption by all ops, as all data carrying
// regions are included in the view and arranged correctly
class ReplicatedGatherOutScatterInViewChanger : public ViewChanger {
public:
  ReplicatedGatherOutScatterInViewChanger(const CollectiveBalancedReorder *cbr_)
      : cbr(cbr_) {}
  poplar::Tensor apply(poplar::Tensor tensor) const final {
    return cbr->undoRearrangeForCollective(tensor).reshape(
        cbr->getReferenceTensor().shape());
  }

private:
  const CollectiveBalancedReorder *cbr;
};

class CollectivesBaseOpx : public Opx {
public:
  CollectivesBaseOpx(Op *, Devicex *);
  // Return all linked tensors and their connected ops to coordinate tensor
  // mapping of collective inputs and outputs
  std::pair<std::set<TensorId>, std::vector<Op *>>
  getCollectiveLinkedGroup() const;
  CollectiveBalancedReorder *getCollectiveBalancedReorder() const;
  CollectiveBalancedReorder *
  createCollectiveBalancedReorder(poplar::Tensor tensor) const;
};

} // namespace popx
} // namespace popart

#endif
