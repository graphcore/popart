#ifndef GUARD_NEURALNET_POPPROGRAMS_HPP
#define GUARD_NEURALNET_POPPROGRAMS_HPP

#include <set>
#include <unordered_map>

#include <poponnx/names.hpp>

namespace poponnx {

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
    PROGRAM,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    STREAMWEIGHTSFROMHOST = 0,
    STREAMOPTIMIZERFROMHOST,
    INIT,
    PREFORWARD,
    FORWARD,
    BACKWARD,
    VARUPDATEFROMACCUMULATOR,
    RESETWEIGHTGRADIENTACCUMULATOR,
    WEIGHTSTOHOST,
    TOHOSTFINALCOPY,
    SETRANDOMSEED,
    SETRANDOMDROPOUTSEED,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &streamWeightsFromHostFragment();
  poplar::program::Sequence &streamOptimizerFromHostFragment();
  poplar::program::Sequence &setRandomSeedFragment();
  poplar::program::Sequence &setRandomDropoutSeedFragment();
  poplar::program::Sequence &toHostFinalCopyFragment();
  poplar::program::Sequence &initFragment();
  poplar::program::Sequence &preForwardFragment();
  poplar::program::Sequence &forwardFragment();
  poplar::program::Sequence &backwardFragment();
  poplar::program::Sequence &varUpdateFromAccumulatorFragment();
  poplar::program::Sequence &resetWeightGradientAccumulatorFragment();
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

  // Recompute program fragments, get and (implicitly) create. There is a unique
  // fragment for each recomputed Op
  poplar::program::Sequence &recomputeFragment(OpId id);

  bool hasBeenRecomputed(OpId) const;
  void recordRecomputed(OpId id);

  enum class PipelineFragmentId {
    ToDeviceStream = 0,
    Forward,
    Backward,
    FwdToHostStream,
    BwdToHostStream,
    IncrStashIndex,
    N // The number of pipeline cycle components
  };
  std::string getStrFromPipelineFragmentId(PipelineFragmentId fragId);

  // Program fragments specific to pipelined model. Each method to return
  // a pipeline program fragment takes a 'description' string, that describes
  // the code being added to the returned fragment. This description is added
  // to pipelineDescs to build up a full description of the program.
  poplar::program::Sequence &pipelineFragment(VGraphId vGraphId,
                                              PipelineFragmentId frag,
                                              const std::string &desc);
  poplar::program::Sequence &
  pipelineToDeviceStreamFragment(VGraphId vGraphId, const std::string &desc);
  poplar::program::Sequence &pipelineForwardFragment(VGraphId vGraphId,
                                                     const std::string &desc);
  poplar::program::Sequence &pipelineBackwardFragment(VGraphId vGraphId,
                                                      const std::string &desc);
  // To stream anchors that are computed in the pipelineForwardFragment
  poplar::program::Sequence &
  pipelineFwdToHostStreamFragment(VGraphId vGraphId, const std::string &desc);
  // To stream anchors that are computed in the pipelineBackwardFragment
  poplar::program::Sequence &
  pipelineBwdToHostStreamFragment(VGraphId vGraphId, const std::string &desc);
  poplar::program::Sequence &pipelineIpuCopyFragment() {
    return pipelineIpuCopySeq;
  }
  poplar::program::Sequence &
  pipelineIncrStashIndexFragment(VGraphId vGraphId, const std::string &desc);
  // If ScheduledPreLoss::Yes, then return pipelineFwdToHostStreamFragment(),
  // else return pipelineBwdToHostStreamFragment()
  poplar::program::Sequence &
  pipelineFwdOrBwdToHostStreamFragment(ScheduledPreLoss,
                                       VGraphId,
                                       const std::string &desc);

  void addPipelineCycle(PipelineCycle pCycle,
                        poplar::program::Sequence &sq,
                        std::ostringstream &ss);

  Devicex *dv_p;

private:
  static constexpr int seqs_size = static_cast<int>(ProgramFragmentIndex::N);
  std::array<poplar::program::Sequence, seqs_size> seqs;

  // The sub-graph program fragments will be stored here
  std::unordered_map<std::string, poplar::program::Sequence> scopeSeqs;

  // The recompute program fragments will be stored here
  std::map<OpId, poplar::program::Sequence> recomputeSeqs;

  // Pipelining fragments for each IPU are stored here
  std::map<PipelineFragmentId, std::map<VGraphId, poplar::program::Sequence>>
      pipelineSeqs;
  // ... and their corresponding descriptions
  std::map<PipelineFragmentId, std::map<VGraphId, std::string>> pipelineDescs;
  poplar::program::Sequence pipelineIpuCopySeq;
  poplar::program::Sequence getMainProgramFromPipelineFragments();

  std::set<OpId> beenRecomputed;

  poplar::program::Sequence weightsFromHost();
  poplar::program::Sequence optimizerFromHost();
  poplar::program::Sequence program();
  poplar::program::Sequence weightsToHost();
};

} // namespace popx
} // namespace poponnx

#endif
