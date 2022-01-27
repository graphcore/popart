// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CREATOR_HPP
#define GUARD_NEURALNET_CREATOR_HPP

#include <utility>

#include <popart/popx/opx.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ICreatorCandidate;
using ICreatorCandidatePtr = std::shared_ptr<ICreatorCandidate>;

struct UnwindEndpoint;
using UnwindEndpointPtr = std::shared_ptr<UnwindEndpoint>;

struct TensorRegion {
  TensorRegion(view::Region offset_, view::Region region_, snap::Tensor tensor_)
      : offset(offset_), region(region_), tensor(tensor_) {}
  view::Region offset;
  view::Region region;
  snap::Tensor tensor;
};

using TensorRegions = std::vector<TensorRegion>;

// A bundle struct to represent the path a tensor
// takes through an Opx
struct OpxInAndOutIndex {
  OpxInAndOutIndex(const PopOpx *opx_, InIndex inIndex_, OutIndex outIndex_)
      : opx(opx_), inIndex(inIndex_), outIndex(outIndex_), isDelegate(false) {}
  OpxInAndOutIndex(const PopOpx *opx_)
      : opx(opx_), inIndex(-1), outIndex(-1), isDelegate(true) {}
  OpxInAndOutIndex() = default;

  bool operator==(const OpxInAndOutIndex &rhs) const {
    return opx == rhs.opx && inIndex == rhs.inIndex && outIndex == rhs.outIndex;
  }

  const PopOpx *opx;
  InIndex inIndex;
  OutIndex outIndex;
  bool isDelegate;
};

// An interface for a potential creator of a tensor
class ICreatorCandidate {
public:
  ICreatorCandidate();
  virtual ~ICreatorCandidate() = default;

  // Creates an input tensor
  virtual std::pair<snap::Tensor, ViewChangers>
  createInput(const poplar::DebugNameAndId &dnai) = 0;

  // Returns the list of tensors (DNF) that must be created before this one
  // Allows disjunctive normal form of must exist tensors, i.e.
  // at least one full set of TensorIds in the vector must exist
  virtual DnfTensorIds mustExistBeforeCreate() = 0;

  virtual double getMaxCreatorPriority() = 0;

  // Number of efficiently laid out tensor elements by the creator candidate
  // after unwinding
  virtual int64_t getNumElems() = 0;

  virtual std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() = 0;

  virtual std::string str() = 0;

  // Return unwound tensor and the view changer that can be applied to the
  // unwound tensor if the tensor does not match IR specifications.
  // Unwinding will currently stop and return when an unwinding Opx that
  // supplies a ViewChanger is reached
  virtual std::pair<snap::Tensor, ViewChangers> unwind(snap::Tensor) = 0;

  virtual std::vector<popart::view::Region> unwind(popart::view::Region) = 0;
  virtual std::vector<popart::view::Region> unwind()                     = 0;

  virtual int64_t getScheduleIndex() const = 0;

  static bool greaterThan(ICreatorCandidatePtr, ICreatorCandidatePtr);
};

class InputCreatorCandidate : public ICreatorCandidate {
public:
  InputCreatorCandidate(InIndex index_,
                        const PopOpx *opx_,
                        std::vector<OpxInAndOutIndex> pathFromInput_,
                        int64_t scheduleIndex_);
  InputCreatorCandidate()                   = default;
  virtual ~InputCreatorCandidate() override = default;

  std::pair<snap::Tensor, ViewChangers>
  createInput(const poplar::DebugNameAndId &dnai) override;

  DnfTensorIds mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;

  int64_t getNumElems() override;

  InIndex getIndex() const { return index; }
  const PopOpx *getOpx() const { return opx; }

  // Returns the unwind path from the tensor to the creator
  std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() final {
    return {pathFromInput};
  }
  void setPathFromInput(std::vector<OpxInAndOutIndex> &value) {
    pathFromInput = value;
  }

  std::pair<snap::Tensor, ViewChangers> unwind(snap::Tensor) override;
  std::vector<popart::view::Region> unwind(popart::view::Region) override;
  std::vector<popart::view::Region> unwind() override;

  std::string str() override;

  int64_t getScheduleIndex() const final { return scheduleIndex; }

protected:
  std::vector<OpxInAndOutIndex> pathFromInput;

private:
  std::pair<snap::Tensor, ViewChangers>
  unwindOnPath(const OpxInAndOutIndex &opxOnPath,
               const snap::Tensor &outTensor,
               const view::Regions &outRegions,
               view::Regions &inRegions);

  // Input index on the creating Op
  InIndex index;
  const PopOpx *opx;
  // Global schedule index to order the creators by global schedule position
  int64_t scheduleIndex;
  // Number of efficiently laid out tensor elements by the creator candidate
  // after unwinding
  int64_t numElements;
};

struct UnwindEndpoint {
  const Graph &graph;
  const Tensor *const tensor;
  std::vector<OpxInAndOutIndex> pathFromInput;

  UnwindEndpoint(const Graph &graph_,
                 const Tensor *const tensor_,
                 std::vector<OpxInAndOutIndex> pathFromInput_)
      : graph(graph_), tensor(tensor_), pathFromInput(pathFromInput_) {}
};

class InputMultiCreatorCandidate : public ICreatorCandidate {
public:
  InputMultiCreatorCandidate();
  virtual ~InputMultiCreatorCandidate() override = default;

  std::pair<snap::Tensor, ViewChangers>
  createInput(const poplar::DebugNameAndId &dnai) override;
  DnfTensorIds mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;
  int64_t getNumElems() override;

  std::string str() override;

  bool addCreatorCandidate(ICreatorCandidatePtr);

  // Returns the unwind path from the tensor to the creator
  std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() final;

  std::pair<snap::Tensor, ViewChangers> unwind(snap::Tensor) override;
  std::vector<popart::view::Region> unwind(popart::view::Region) override;
  std::vector<popart::view::Region> unwind() override;

  int64_t getScheduleIndex() const final;

private:
  view::Regions getAcceptedSubregions(view::Region);

  std::map<ICreatorCandidatePtr, view::Regions> candidates;
};

} // namespace popx
} // namespace popart

#endif
