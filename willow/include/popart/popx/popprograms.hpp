// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPPROGRAMS_HPP
#define GUARD_NEURALNET_POPPROGRAMS_HPP

#include <set>
#include <unordered_map>

#include <popart/names.hpp>

namespace popart {

enum class ScheduledPreLoss;

namespace popx {

class Devicex;
class PipelineInfo;

class PopPrograms {

public:
  // We may want to run some programs multiple times without having
  // to communicate with the host to call the 'run'. By supplying a
  // count, we can loop a repeatable program inside a Poplar repeat
  // program
  PopPrograms(Devicex *dv_p_);

  enum ProgramIndex {
    WeightsFromHost = 0,
    OptimizerFromHost,
    SetRandomSeedFromHost,
    RngStateFromHost,
    Program,
    RngStateToHost,
    WeightstoHost,
    CycleCountTensortoHost,
    N // The number of programs
  };

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    StreamWeightsFromHost = 0,
    StreamOptimizerFromHost,
    SetRandomSeedFromHost,
    RngStateFromHost,
    Init,
    PreForward,
    Forward,
    Backward,
    VarUpdateFromAccumulator,
    RngStateToHost,
    WeightstoHost,
    ToHostFinalCopy,
    CycleCountTensortoHost,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  const poplar::program::Sequence &streamWeightsFromHostFragment() const;
  poplar::program::Sequence &streamWeightsFromHostFragment();
  const poplar::program::Sequence &streamOptimizerFromHostFragment() const;
  poplar::program::Sequence &streamOptimizerFromHostFragment();
  const poplar::program::Sequence &setRandomSeedFromHostFragment() const;
  poplar::program::Sequence &setRandomSeedFromHostFragment();
  const poplar::program::Sequence &cycleCountTensorToHostFragment() const;
  poplar::program::Sequence &rngStateFromHostFragment();
  const poplar::program::Sequence &rngStateFromHostFragment() const;
  poplar::program::Sequence &rngStateToHostFragment();
  const poplar::program::Sequence &rngStateToHostFragment() const;
  poplar::program::Sequence &cycleCountTensorToHostFragment();
  const poplar::program::Sequence &toHostFinalCopyFragment() const;
  poplar::program::Sequence &toHostFinalCopyFragment();
  const poplar::program::Sequence &initFragment() const;
  poplar::program::Sequence &initFragment();
  const poplar::program::Sequence &preForwardFragment() const;
  poplar::program::Sequence &preForwardFragment();
  const poplar::program::Sequence &forwardFragment() const;
  poplar::program::Sequence &forwardFragment();
  const poplar::program::Sequence &backwardFragment() const;
  poplar::program::Sequence &backwardFragment();
  const poplar::program::Sequence &accumulateOuterFragment() const;
  poplar::program::Sequence &accumulateOuterFragment();
  const poplar::program::Sequence &weightsToHostFragment() const;
  poplar::program::Sequence &weightsToHostFragment();
  // If ScheduledPreLoss::Yes, then return forwardFragment(), else return
  // backwardFragment()
  poplar::program::Sequence &forwardOrBackwardFragment(ScheduledPreLoss);

  // A list of programs that can be run by the Poplar engine.
  const std::vector<poplar::program::Program> progs() const;

  poplar::program::Sequence &programFragment(PopPrograms::ProgramFragmentIndex);

  // Sub-graph program fragments, getters and setters
  poplar::program::Sequence &scopeFragment(const Graph &);
  bool containsFragment(const Graph &) const;
  void createFragment(const Graph &);

  poplar::Function &getFragmentFunction(const Graph &called_graph,
                                        poplar::Graph &popgraph);

  // Recompute program fragments, get and (implicitly) create. There is a unique
  // fragment for each recomputed Op
  poplar::program::Sequence &recomputeFragment(OpId);

  bool hasBeenRecomputed(OpId, ExecutionPhase) const;
  void recordRecomputed(OpId, ExecutionPhase);

  enum class PipelineFragmentId {
    ToDeviceStream = 0,
    Restore,
    Forward,
    ToHostStream,
    // IpuCopy fragment has been removed. There is now a single
    // pipelineIpuCopySeq to which copies are added.
  };
  std::string getStrFromPipelineFragmentId(PipelineFragmentId) const;

  // Program fragments specific to pipelined model. Each method to return
  // a pipeline program fragment takes a 'description' string, that describes
  // the code being added to the returned fragment. This description is added
  // to pipelineDescs to build up a full description of the program.
  poplar::program::Sequence &
  pipelineFragment(PipelineStage, PipelineFragmentId, const std::string &desc);

  poplar::program::Sequence &
  pipelineToDeviceStreamFragment(PipelineStage pipelineStage,
                                 const std::string &desc);
  poplar::program::Sequence &pipelineForwardFragment(PipelineStage,
                                                     const std::string &desc);
  poplar::program::Sequence &pipelineRestoreFragment(PipelineStage,
                                                     const std::string &desc);

  // To stream anchors that are computed in the pipelineForwardFragment
  poplar::program::Sequence &
  pipelineToHostStreamFragment(PipelineStage, const std::string &desc);
  poplar::program::Sequence &pipelineIpuCopyFragment(const std::string &desc);

  void addPipelineCycle(
      PipelineCycle pCycle,
      poplar::program::Sequence &sq,
      std::ostringstream &ss,
      std::map<PipelineStage, poplar::Function> &fwdFunctions) const;

  Devicex *dv_p;

private:
  static constexpr int seqs_size = static_cast<int>(ProgramFragmentIndex::N);
  std::array<poplar::program::Sequence, seqs_size> seqs;

  // The sub-graph program fragments will be stored here
  std::unordered_map<std::string, poplar::program::Sequence> scopeSeqs;
  std::unordered_map<std::string, poplar::Function> funcs;

  // The recompute program fragments will be stored here
  std::map<OpId, poplar::program::Sequence> recomputeSeqs;

  // Pipelining fragments for each pipeline stage are stored here
  std::map<PipelineFragmentId,
           std::map<PipelineStage, poplar::program::Sequence>>
      pipelineSeqs;

  // ... and their corresponding descriptions
  std::map<PipelineFragmentId, std::map<PipelineStage, std::string>>
      pipelineDescs;

  // IpuCopy program
  poplar::program::Sequence pipelineIpuCopySeq;
  std::string pipelineIpuCopyDesc;

  poplar::program::Sequence getMainProgramFromPipelineFragments() const;

  std::set<std::pair<OpId, ExecutionPhase>> beenRecomputed;

  poplar::program::Sequence weightsFromHost() const;
  poplar::program::Sequence optimizerFromHost() const;
  poplar::program::Sequence setRandomSeedFromHost() const;
  poplar::program::Sequence rngStateFromHost() const;
  poplar::program::Sequence cycleCountTensorToHost() const;
  poplar::program::Sequence program() const;
  poplar::program::Sequence rngStateToHost() const;
  poplar::program::Sequence weightsToHost() const;
};

} // namespace popx
} // namespace popart

#endif
