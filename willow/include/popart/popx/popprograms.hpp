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
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    SETRANDOMSEEDFROMHOST,
    PROGRAM,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    STREAMWEIGHTSFROMHOST = 0,
    STREAMOPTIMIZERFROMHOST,
    SETRANDOMSEEDFROMHOST,
    INIT,
    PREFORWARD,
    FORWARD,
    BACKWARD,
    VARUPDATEFROMACCUMULATOR,
    WEIGHTSTOHOST,
    TOHOSTFINALCOPY,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &streamWeightsFromHostFragment();
  poplar::program::Sequence &streamOptimizerFromHostFragment();
  poplar::program::Sequence &setRandomSeedFromHostFragment();
  poplar::program::Sequence &toHostFinalCopyFragment();
  poplar::program::Sequence &initFragment();
  poplar::program::Sequence &preForwardFragment();
  poplar::program::Sequence &forwardFragment();
  poplar::program::Sequence &backwardFragment();
  poplar::program::Sequence &accumulateOuterFragment();
  poplar::program::Sequence &weightsToHostFragment();
  // If ScheduledPreLoss::Yes, then return forwardFragment(), else return
  // backwardFragment()
  poplar::program::Sequence &forwardOrBackwardFragment(ScheduledPreLoss);

  // A list of programs that can be run by the Poplar engine.
  std::vector<poplar::program::Program> progs();

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

  bool hasBeenRecomputed(OpId) const;
  void recordRecomputed(OpId);

  enum class PipelineFragmentId {
    ToDeviceStream = 0,
    Forward,
    ToHostStream,
    // IpuCopy fragment has been removed. There is now a Sequence per
    // PipelineCycle in pipelineIpuCopySeqs to which copies are added.
  };
  std::string getStrFromPipelineFragmentId(PipelineFragmentId);

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

  // To stream anchors that are computed in the pipelineForwardFragment
  poplar::program::Sequence &
  pipelineToHostStreamFragment(PipelineStage, const std::string &desc);
  std::vector<poplar::program::Sequence *>
  pipelineIpuCopyFragments(PipelineStage, const std::string &desc);

  void addPipelineCycle(PipelineCycle pCycle,
                        poplar::program::Sequence &sq,
                        std::ostringstream &ss);

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

  // IpuCopy programs
  std::map<PipelineCycle, poplar::program::Sequence> pipelineIpuCopySeqs;
  std::map<PipelineCycle, std::vector<std::string>> pipelineIpuCopySeqDescs;

  // ... and their corresponding descriptions
  std::map<PipelineFragmentId, std::map<PipelineStage, std::string>>
      pipelineDescs;

  poplar::program::Sequence getMainProgramFromPipelineFragments();

  std::set<OpId> beenRecomputed;

  poplar::program::Sequence weightsFromHost();
  poplar::program::Sequence optimizerFromHost();
  poplar::program::Sequence setRandomSeedFromHost();
  poplar::program::Sequence program();
  poplar::program::Sequence weightsToHost();
};

} // namespace popx
} // namespace popart

#endif
