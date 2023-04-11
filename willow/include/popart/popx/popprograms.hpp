// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_POPPROGRAMS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_POPPROGRAMS_HPP_

#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popef/Types.hpp>
#include <poplar/Program.hpp>
#include <popart/graphid.hpp>
#include <popart/names.hpp>
#include <popart/popx/pritask.hpp>

namespace poplar {
class Function;
class FunctionBuffer;
enum class FunctionBufferMappingType;
} // namespace poplar
namespace poplar {
class Graph;
} // namespace poplar

namespace popart {
class Graph;
class PipelineInfo;

enum class ScheduledPreLoss;

namespace popx {

class IrLowering;

/**
 * Class for managing the complete set of \c programs that a \c Devicex can run.
 *
 * A \c program in this context is the instance of the  \c poplar::Program class
 * which represents a control program that executes operations on the graph.
 *
 * The state \c std::vector<poplar::program::Sequence> \c seqs contains all
 *these programs, and is populated during \c IrLowering. The programs are passed
 *to \c poplar::compileGraph to construct the executable (see \c
 *IrLowering::getExecutable()).
 **/
class PopPrograms {

public:
  // We may want to run some programs multiple times without having
  // to communicate with the host to call the 'run'. By supplying a
  // count, we can loop a repeatable program inside a Poplar repeat
  // program
  PopPrograms(IrLowering *ir_lowering_p_);

  enum ProgramIndex {
    WeightsFromHost = 0,
    OptimizerFromHost,
    RandomSeedFromHost,
    RandomSeedToHost,
    RngStateFromHost,
    Program,
    RngStateToHost,
    WeightsToHost,
    CycleCountTensorToHost,
    CustomProgramsStart,
    N // The number of enums
  };

  static const std::unordered_map<popef::ProgramFlow::ProgramIndexType,
                                  std::string>
      commonPrograms;

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    StreamWeightsFromHost = 0,
    StreamOptimizerFromHost,
    RandomSeedFromHost,
    RandomSeedToHost,
    RngStateFromHost,
    Init,
    PreForward,
    Forward,
    Backward,
    VarUpdateFromAccumulator,
    RngStateToHost,
    WeightsToHost,
    ToHostFinalCopy,
    CycleCountTensorToHost,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  const poplar::program::Sequence &streamWeightsFromHostFragment() const;
  poplar::program::Sequence &streamWeightsFromHostFragment();
  const poplar::program::Sequence &streamOptimizerFromHostFragment() const;
  poplar::program::Sequence &streamOptimizerFromHostFragment();
  const poplar::program::Sequence &randomSeedFromHostFragment() const;
  poplar::program::Sequence &randomSeedFromHostFragment();
  const poplar::program::Sequence &randomSeedToHostFragment() const;
  poplar::program::Sequence &randomSeedToHostFragment();
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

  // Sub-graph program fragments, getters and setters for poplar sequences and
  // functions for subgraphs.

  // The number of Poplar sequences associated with a graph.
  int getNumFragments(const Graph &graph) const;
  // Get a vector of all Poplar sequences associated with a graph.
  std::vector<poplar::program::Sequence> &scopeFragments(const Graph &);
  // Get a specific Poplar sequence associated with a graph.
  poplar::program::Sequence &scopeFragment(const Graph &,
                                           SubgraphPartIndex subgraphPart);
  // Determine if any Poplar sequences associated with a graph are allocated.
  bool containsFragments(const Graph &graph) const;
  // Determine whether a specific Poplar sequence associated with a graph has
  // been allocated.
  bool containsFragment(const Graph &graph,
                        SubgraphPartIndex subgraphPart) const;
  // Ensure a specific Poplar sequence is allocated.
  void createFragment(const Graph &graph, SubgraphPartIndex subgraphPart);
  // Wrap all Poplar sequences associated with a graph in to a poplar function
  // that can be called and return them all.
  std::vector<poplar::Function> &
  getFragmentFunctions(const Graph &graph, poplar::Graph &poplarGrpah);
  // Wrap all Poplar sequences associated with a graph in to a poplar function
  // that can be called and return a specific one.
  poplar::Function &getFragmentFunction(const Graph &graph,
                                        SubgraphPartIndex subgraphPart,
                                        poplar::Graph &poplarGraph);

  // Get the program fragment for a recomputed op. createRecomputeFragment must
  // be called first.
  std::vector<poplar::program::Sequence>::iterator recomputeFragment(OpId);
  // Create the program fragment for a recomputed op.
  SequenceMap::SequenceInterval createRecomputeFragment(OpId);

  bool hasBeenRecomputed(OpId, ExecutionPhase) const;
  void recordRecomputed(OpId, ExecutionPhase);

  // Each Pipeline Stage is composed of these fragments. For a given Pipeline
  // Stage, any of these fragments may be empty.
  //
  // Note: the preForwardFragment and IpuCopy fragment do not require a
  // PipelineFragmentId, since they exist as a single fragment idependent of
  // pipeline stage, and are run every Pipeline Cycle.
  enum class PipelineFragmentId { ToDeviceStream = 0, Main, ToHostStream };
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
  poplar::program::Sequence &pipelineMainFragment(PipelineStage,
                                                  const std::string &desc);

  // To stream anchors that are computed in the pipelineMainFragment
  poplar::program::Sequence &
  pipelineToHostStreamFragment(PipelineStage, const std::string &desc);
  poplar::program::Sequence &pipelineIpuCopyFragment(const std::string &desc);

  poplar::program::Sequence &namedBuffersCopyFragment();

  void addPipelineCycle(PipelineInfo pInfo,
                        PipelineCycle pCycle,
                        poplar::program::Sequence &sq,
                        std::ostringstream &ss) const;

  /**
   * Add a vector of pairs {f, buffer} for a given graph id. This is enough for
   * a [Internal|External]CodeCopy op to move code from the buffer in to
   * the function. Note the subgraphpartitioner may have split this into
   * multiple functions, so we require a vector of these for each graph.
   *
   * \param pair The graph id, FunctionBufferMappingType pair to add the
   * functions and buffers for.
   * \param funcVec The vector of functions and buffers.
   */
  void addFunctionBuffers(const GraphId gid,
                          poplar::FunctionBufferMappingType fbmt);

  // Shorthand storage type for storing functionbuffers.
  using FunctionBuffers =
      std::vector<std::pair<const poplar::Function, poplar::FunctionBuffer>>;

  /**
   * Get the Function Buffers for the given GraphId and
   * FunctionBufferMappingType
   *
   * \param gid The GraphId to lookup.
   * \param fbmt The FunctionBufferMappingType to lookup.
   * \returns FunctionBuffers the vector of functions and buffers.
   */
  FunctionBuffers getFunctionBuffer(const GraphId gid,
                                    poplar::FunctionBufferMappingType fbmt) {
    return functionBuffers.at({gid, fbmt});
  }

  /**
   * Returns true if a functionBuffer vector exists for the given graphId and
   * FunctionBufferMappingType.
   *
   * \param gid The graph id to lookup.
   * \param fbmt The FunctionBufferMappingType to lookup.
   * \returns true If pairs exist.
   * \returns false Otherwise.
   */
  bool hasFunctionBuffer(const GraphId gid,
                         poplar::FunctionBufferMappingType fbmt) {
    return functionBuffers.count({gid, fbmt}) > 0;
  }

  /**
   * Add a custom program
   * \param program       Program to add
   * \return              Index of the popart/poplar program
   */
  unsigned addCustomProgram(const poplar::program::Program &program);

  IrLowering *ir_lowering_p;

private:
  std::vector<poplar::program::Sequence> seqs;

  // The sub-graph program fragments will be stored here
  std::unordered_map<std::string, std::vector<poplar::program::Sequence>>
      scopeSeqs;
  std::unordered_map<std::string, std::vector<poplar::Function>> funcs;

  // The recompute program fragments will be stored here. We store the sequences
  // in singleton vectors because grow code requires iterators to vectors.
  std::map<OpId, std::vector<poplar::program::Sequence>> recomputeSeqs;

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

  // Implicit pipeline functions
  poplar::Function zeroPipelineIndexFunction;

  // Functions containing the pipeline stages
  std::map<PipelineStage, poplar::Function> mainPipelineFunctions;

  // Function copying all required tensors between consecutive pipeline stages
  poplar::Function pipelineIpuCopyFunction;

  // Function containing the implicit stream copies from device to host for
  // tensors only streamed once for the whole program run
  // (see AnchorReturnTypeId::Final)
  poplar::Function toHostFinalCopyFunction;

  // Program for selective update of named buffers
  poplar::program::Sequence namedBuffersCopySeq;

  // Custom programs
  std::vector<poplar::program::Program> customPrograms;

  // Map of the {graph id, FunctionBufferMappingType}'s and their associated
  // functions and buffers for use in loading the code for the graph execution
  // on / off chip. Note that there may be multiple functions per graph so a
  // vector of functions / buffers is stored. Furthermore these are indexed by
  // {graphid, FunctionBufferMappingType} in case different
  // FunctionBufferMappingTypes are required for multiple call ops for the same
  // graph.
  std::map<std::pair<const GraphId, poplar::FunctionBufferMappingType>,
           FunctionBuffers>
      functionBuffers;

public:
  /**
   * Turn pipeline sequences into callable pipeline functions
   */
  void createPipelineFunctions();

  /**
   * Return the program based on the pipeline fragments.
   *
   * See docs/notes/transforms/pipelining.md#assemble-from-fragments for
   * detailed explanation.
   *
   * \return The program based on the pipeline fragments
   **/
  poplar::program::Sequence
  getFullProgramFromPipelineFragments(bool fwdOnly) const;

private:
  std::set<std::pair<OpId, ExecutionPhase>> beenRecomputed;

  poplar::program::Sequence weightsFromHost() const;
  poplar::program::Sequence optimizerFromHost() const;
  poplar::program::Sequence randomSeedFromHost() const;
  poplar::program::Sequence randomSeedToHost() const;
  poplar::program::Sequence rngStateFromHost() const;
  poplar::program::Sequence cycleCountTensorToHost() const;
  poplar::program::Sequence program() const;
  poplar::program::Sequence rngStateToHost() const;
  poplar::program::Sequence weightsToHost() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_POPPROGRAMS_HPP_
