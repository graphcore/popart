#ifndef GUARD_NEURALNET_ALIASZEROCOPY_HPP
#define GUARD_NEURALNET_ALIASZEROCOPY_HPP

#include <popart/aliases.hpp>
#include <popart/liveness.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/transforms/transform.hpp>

// AliasZeroCopy:
// Optimisation pass to eliminate copies into and out of subgraphs, and to reuse
// Poplar tensors that are directly created as graph variables.
// Runs a global liveness analysis of relevant tensors over all graphs and
// all their call sites.
//
// Case 1: Reusing cached tensors:
//    InitOp can create tensors which can be used as not always live
//    TensorType::Variable tensors with streaming memory mode.
//    If two Cache-tensors are shape, type and liveness compatible, they will,
//    during lowering to Poplar, reuse the same underlying Poplar::Tensor.
//
// Case 2: Alias instead of copying into subgraphs:
//    If tb is the subgraph input copied from the parent scope tensor ta

//    Primary call site:
//      X -> ta -> CopyInput <-- to be eliminated -> ta == tb
//                         |
//                         tb -> Y
//
//    Secondary call site:
//      X -> tc -> CopyInput
//                         |
//                         tb -> Y
//
//    then ta can be aliased to tb if the liveness of tc (and any other
//    seconday call site) agree.
//    If possible, but not necessarily, tc will also alias to ta.
//
//
// Case 3: Alias instead of copying out of subgraphs:
//    If tb is the parent scope output copied from the subgraph tensor ta

//    Primary call site:
//                          tb -> Y
//                          |
//      X -> ta -> CopyOutput <-- to be eliminated -> ta == tb
//
//    Secondary call sites:
//                          tc -> Y
//                          |
//      X -> ta -> CopyOutput

//    then ta can be aliased to tb if the liveness of tc (and any other
//    secondary call site) agree.
//    If possible, but not necessarily, tc will also alias to tb.

namespace popart {
namespace liveness {

// Right-open intervals of tensor liveness
class IntervalsImpl;
class Intervals {
public:
  Intervals();
  Intervals(const Intervals &other);
  ~Intervals();
  void insert(int64_t s, int64_t e);
  bool empty() const;

  Intervals operator&(const Intervals &other) const;
  void operator+=(const Intervals &other);

private:
  std::unique_ptr<IntervalsImpl> intervals;
  friend class AliasZeroCopy;
};

class AliasZeroCopy {
public:
  static std::size_t id();

  AliasZeroCopy(const Ir *ir, const LivenessAnalyzer *analyzer);
  void apply();

  void removePostIRAliases(Tensor *);

  std::set<Tensor *, PTensorCmp> getPostIRAliases(Tensor *) const;

  std::set<Tensor *, PTensorCmp> getTensorsWithPostIRAliases() const;

  static bool doOverlap(const Intervals &aIntervals,
                        const Intervals &bIntervals);

  std::set<Tensor *, PTensorCmp>
  getProposedAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                            bool fullyAliased) const;

  std::set<Tensor *, PTensorCmp>
  getActiveAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                          bool fullyAliased) const;

  void activateAlias(Tensor *ta, Tensor *tb);

  bool copyModifiedRequired(Op *op, InIndex inIndex) const;

private:
  std::set<Tensor *, PTensorCmp>
  getAliasedTensors(const Aliases &aliases,
                    std::set<Tensor *, PTensorCmp> tensors,
                    bool fullyAliased) const;

  void insertAlias(Tensor *ta, Tensor *tb);

  // Find the liveness start of a consumed tensor
  int64_t findStart(Tensor *consumedTensor, int64_t scheduleIndex) const;

  // Find the liveness front of a consumed tensor
  int64_t findFront(Tensor *consumedTensor, int64_t scheduleIndex) const;

  // Returns when the tensor needs to be live in the schedule
  Intervals getLivenessIntervals(Tensor *);

  // Returns when the tensor (including all aliases)
  // needs to be live in the schedule
  Intervals getCandidateLivenessIntervals(Tensor *, bool cached = true);

  bool checkCandidatesCompatible(Tensor *ta, Tensor *tb);

  // If tb is the subgraph input copied from the parent scope tensor ta
  // Primary call site:
  // X -> ta -> CopyInput <-- to be eliminated -> ta == tb
  //                    |
  //                    tb -> Y
  //
  // Secondary call site:
  // X -> tc -> CopyInput
  //                    |
  //                    tb -> Y
  bool checkSubgraphInputCompatible(Tensor *ta, Tensor *tb);

  // If tb is the parent scope output copied from the subgraph tensor ta
  // Primary call site:
  //                     tb -> Y
  //                     |
  // X -> ta -> CopyOutput <-- to be eliminated -> ta == tb
  //
  // Secondary call sites:
  //                     tc -> Y
  //                     |
  // X -> ta -> CopyOutput
  bool checkSubgraphOutputCompatible(Tensor *ta, Tensor *tb);

  std::vector<Tensor *>
  processTensorAliasGroups(std::set<Tensor *, PTensorCmp> proposedTensor);

  Tensor *findAliasableTensor(Tensor *);

  void logPostIRAliases();

  // Debug printing liveness intervals
  void printLivenessIntervals(std::set<Tensor *, PTensorCmp> tensors);

  const Ir *ir;
  const LivenessAnalyzer *analyzer;

  std::map<std::pair<Tensor *, Tensor *>, bool> candidateCompatMap;
  std::map<Tensor *, Intervals, PTensorCmp> candidateLivenessIntervalsMap;

  // Aliases as inferred from all IR graphs
  Aliases irAliases;
  Aliases proposedAliases;
  Aliases activeAliases;

  // Required CopyModified
  std::map<std::pair<Op *, InIndex>, bool> requiredCopyModified;

  std::map<Tensor *, std::set<Tensor *, PTensorCmp>, PTensorCmp> postIRAliases;

  // Debugging tool to exclude alias zero copy of tensors with certain names
  // by substring matching
  std::set<std::string> excludeTensorByName;
};

} // namespace liveness
} // namespace popart

#endif
