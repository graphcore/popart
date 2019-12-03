#ifndef GUARD_NEURALNET_CREATOR_HPP
#define GUARD_NEURALNET_CREATOR_HPP

#include <utility>

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ICreatorCandidate;
using ICreatorCandidatePtr = std::shared_ptr<ICreatorCandidate>;

struct UnwindEndpoint;
using UnwindEndpointPtr = std::shared_ptr<UnwindEndpoint>;

// A bundle struct to represent the path a tensor
// takes through an Opx
struct OpxInAndOutIndex {
  OpxInAndOutIndex(const Opx *opx_, InIndex inIndex_, OutIndex outIndex_)
      : opx(opx_), inIndex(inIndex_), outIndex(outIndex_) {}
  OpxInAndOutIndex() = default;

  const Opx *opx;
  InIndex inIndex;
  OutIndex outIndex;
};

// An interface for a potential creator of a tensor
class ICreatorCandidate {
public:
  ICreatorCandidate();
  virtual ~ICreatorCandidate() = default;

  // Creates an input tensor
  virtual poplar::Tensor createInput(const std::string &name) = 0;

  // Returns the list of tensors that must be created before this one
  virtual std::vector<TensorId> mustExistBeforeCreate() = 0;

  virtual double getMaxCreatorPriority() = 0;

  virtual int64_t getNumElems() = 0;

  virtual std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() = 0;

  virtual std::string str() = 0;

  virtual poplar::Tensor unwind(poplar::Tensor)                          = 0;
  virtual std::vector<popart::view::Region> unwind(popart::view::Region) = 0;
  virtual std::vector<popart::view::Region> unwind()                     = 0;

  static bool greaterThan(ICreatorCandidatePtr, ICreatorCandidatePtr);
};

class InputCreatorCandidate : public ICreatorCandidate {
public:
  InputCreatorCandidate(int, const Opx *, std::vector<OpxInAndOutIndex>);
  InputCreatorCandidate()                   = default;
  virtual ~InputCreatorCandidate() override = default;

  poplar::Tensor createInput(const std::string &name) override;

  std::vector<TensorId> mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;
  int64_t getNumElems() override;

  int getIndex() const { return index; }
  const Opx *getOpx() const { return opx; }

  // Returns the unwind path from the tensor to the creator
  std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() final {
    return {pathFromInput};
  }
  void setPathFromInput(std::vector<OpxInAndOutIndex> &value) {
    pathFromInput = value;
  }

  poplar::Tensor unwind(poplar::Tensor) override;
  std::vector<popart::view::Region> unwind(popart::view::Region) override;
  std::vector<popart::view::Region> unwind() override;

  virtual std::string str() override;

protected:
  std::vector<OpxInAndOutIndex> pathFromInput;

private:
  int index;
  const Opx *opx;
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

  poplar::Tensor createInput(const std::string &name) override;
  std::vector<TensorId> mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;
  int64_t getNumElems() override;

  virtual std::string str() override;

  bool addCreatorCandidate(ICreatorCandidatePtr);

  // Returns the unwind path from the tensor to the creator
  std::vector<std::vector<OpxInAndOutIndex>> getPathsFromInput() final;

  poplar::Tensor unwind(poplar::Tensor) override;
  std::vector<popart::view::Region> unwind(popart::view::Region) override;
  std::vector<popart::view::Region> unwind() override;

private:
  view::Regions getAcceptedSubregions(view::Region);

  std::map<ICreatorCandidatePtr, view::Regions> candidates;
};

bool hasSmallerPriority(ICreatorCandidatePtr icc1, ICreatorCandidatePtr icc2);

} // namespace popx
} // namespace popart

#endif
