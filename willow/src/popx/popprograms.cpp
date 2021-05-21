// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <fstream>
#include <ostream>
#include <random>
#include <set>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include <popart/popx/irlowering.hpp>
#include <popart/popx/popprograms.hpp>

#include <poplar/Program.hpp>

#include <snap/Graph.hpp>

namespace popart {
namespace popx {

std::ostream &operator<<(std::ostream &out,
                         PopPrograms::ProgramFragmentIndex index) {
  switch (index) {
  case PopPrograms::ProgramFragmentIndex::StreamWeightsFromHost: {
    out << "StreamWeightsFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::StreamOptimizerFromHost: {
    out << "StreamOptimizerFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::SetRandomSeedFromHost: {
    out << "SetRandomSeedFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RngStateFromHost: {
    out << "RngStateFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Init: {
    out << "Init";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::PreForward: {
    out << "PreForward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Forward: {
    out << "Forward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Backward: {
    out << "Backward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::VarUpdateFromAccumulator: {
    out << "VarUpdateFromAccumulator";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RngStateToHost: {
    out << "RngStateToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::WeightstoHost: {
    out << "WeightstoHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::ToHostFinalCopy: {
    out << "ToHostFinalCopy";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::CycleCountTensortoHost: {
    out << "CycleCountTensortoHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::N: {
    out << "N";
    break;
  }
  default: {
    throw internal_error("Invalid value for ProgramFragmentIndex");
  }
  };
  return out;
}

PopPrograms::PopPrograms(IrLowering *ir_lowering_p_)
    : ir_lowering_p(ir_lowering_p_) {
  // Populate seqs with Sequences that have names.
  for (int i = 0; i < static_cast<int>(ProgramFragmentIndex::N); ++i) {
    std::stringstream ss;
    ss << static_cast<ProgramFragmentIndex>(i);
    seqs.push_back(poplar::program::Sequence({}, ss.str()));
  }
}

const poplar::program::Sequence &
PopPrograms::streamWeightsFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}
poplar::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}

const poplar::program::Sequence &
PopPrograms::streamOptimizerFromHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}
poplar::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}

const poplar::program::Sequence &
PopPrograms::setRandomSeedFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::SetRandomSeedFromHost));
}
poplar::program::Sequence &PopPrograms::setRandomSeedFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::SetRandomSeedFromHost));
}

const poplar::program::Sequence &PopPrograms::rngStateFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

poplar::program::Sequence &PopPrograms::rngStateFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

const poplar::program::Sequence &PopPrograms::rngStateToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

poplar::program::Sequence &PopPrograms::rngStateToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

const poplar::program::Sequence &
PopPrograms::cycleCountTensorToHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensortoHost));
}
poplar::program::Sequence &PopPrograms::cycleCountTensorToHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensortoHost));
}

const poplar::program::Sequence &PopPrograms::initFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

poplar::program::Sequence &PopPrograms::initFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

const poplar::program::Sequence &PopPrograms::preForwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

poplar::program::Sequence &PopPrograms::preForwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

const poplar::program::Sequence &PopPrograms::forwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

poplar::program::Sequence &PopPrograms::forwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

const poplar::program::Sequence &PopPrograms::backwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

poplar::program::Sequence &PopPrograms::backwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

const poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

const poplar::program::Sequence &PopPrograms::accumulateOuterFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

poplar::program::Sequence &PopPrograms::accumulateOuterFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

const poplar::program::Sequence &PopPrograms::weightsToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightstoHost));
}

poplar::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightstoHost));
}

poplar::program::Sequence PopPrograms::weightsFromHost() const {
  poplar::program::Sequence prog({}, {"weightsFromHost"});
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::optimizerFromHost() const {
  poplar::program::Sequence prog({}, {"optimizerFromHost"});
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::setRandomSeedFromHost() const {
  poplar::program::Sequence prog({}, {"setRandomSeedFromHost"});
  prog.add(setRandomSeedFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::cycleCountTensorToHost() const {
  poplar::program::Sequence prog({}, {"cycleCountTensorToHost"});
  prog.add(cycleCountTensorToHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::rngStateFromHost() const {
  poplar::program::Sequence prog({}, {"rngStateFromHost"});
  prog.add(rngStateFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::rngStateToHost() const {
  poplar::program::Sequence prog({}, {"rngStateToHost"});
  prog.add(rngStateToHostFragment());
  return prog;
}

void PopPrograms::addPipelineCycle(
    PipelineCycle pCycle,
    poplar::program::Sequence &sq,
    std::ostringstream &ss,
    std::map<PipelineStage, poplar::Function> &mainFunctions) const {
  // Inside each pipeline cycle
  //
  // Always do:
  // 1. The pre-forward fragment
  //
  // Then conditionally do:
  // 2. Host->Device copies for each pipeline stage
  // 3. Main fragments for each pipeline stage
  // 4. Device->Host copies for each pipeline stage
  //
  // Then always do:
  // 5. Inter-IPU copies for all pipeline stages

  PipelineInfo pInfo = ir_lowering_p->pipelineInfo();

  // 1.
  sq.add(preForwardFragment());

  // 2.
  if (pipelineSeqs.find(PipelineFragmentId::ToDeviceStream) !=
      pipelineSeqs.end()) {
    for (auto &stage_seq :
         pipelineSeqs.at(PipelineFragmentId::ToDeviceStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToDeviceStream";
        sq.add(stage_seq.second);
      }
    }
  } else {
    if (ir_lowering_p->ir().useSyntheticData() == false) {
      throw error(
          "There are no ToDeviceStream pipeline program fragments. Check that "
          "the stream copies have been added to the correct fragment.");
    }
  }

  // 3.
  for (auto &stage_seq : mainFunctions) {
    auto stage = stage_seq.first;
    if (pInfo.doStage(pCycle, stage)) {
      ss << "\n  ps" << stage << " : Main";
      sq.add(poplar::program::Call(stage_seq.second));
    }
  }

  // 4.
  if (pipelineSeqs.find(PipelineFragmentId::ToHostStream) !=
      pipelineSeqs.end()) {
    for (auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::ToHostStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToHostStream";
        sq.add(stage_seq.second);
      }
    }
  }

  // 5.
  // Note: Always do all the copies. This is ensure that ALL copies are
  // outlined across pipelineCycles AND merged across pipelineStages.
  ss << logging::format("\n  IpuCopies");
  sq.add(pipelineIpuCopySeq);
}

poplar::program::Sequence
PopPrograms::getFullProgramFromPipelineFragments() const {
  // First, some terminology:
  // - Pipeline Stage
  //    - A partition of the graph that can be (in theory) executed in parallel
  //      with any other pipeline stage (although multiple pipeline stages
  //      mapped to a single IPU will in practice run serially).
  //    - Each Pipeline Stage operates on a single and separate microbatch of
  //      data.
  //    - Excluding inter-IPU copies and host IO, operations on a Pipeline Stage
  //      have no dependencies on other Pipeline Stages within a single Pipeline
  //      Cycle.
  //    - Pipeline Stages cannot span IPU boundaries.
  // - Pipeline Cycle
  //    - The time step with which mini-batches move through the pipeline.
  //    - All Pipeline Stages are executed once within one Pipeline Cycle, in
  //      parallel (except for some serialisation if multiple Pipeline Stages
  //      are mapped to a single IPU).
  //
  // The way we assemble the full pipeline program from program fragments is
  // based on two ideas:
  //
  // 1. Constraints are imposed on the order of fragments by Poplar lowering
  //    optimisations to guarantee parallel execution over IPUs.
  //    A poplar::program is constructed serially, like:
  //
  //        poplar::Program::Sequence seq;
  //        seq.add(fragment0);
  //        seq.add(fragment1);
  //        ...
  //        seq.add(fragmentN);
  //
  //    But a successfully pipelined model will have maximally parallelised
  //    execution over IPUs. For fragments 0 to N to be parallelisable, they
  //    must:
  //      - run on different IPUs to one another
  //      - not entail a global exchange
  //
  //    In PopART we enforce this by splitting operations assigned to each
  //    Pipeline Stage into four fragments: ToDeviceStream (D), Main (M),
  //    ToHostStream (H), and IpuCopy (C). The program for a Pipeline Cycle
  //    is assembled by:
  //      - Adding to the program D fragments for all Pipeline Stages
  //        participating in the Pipeline Cycle.
  //      - Followed by M framents for all Pipeline Stages participating in the
  //        Pipeline Cycle.
  //      - Followed by H framents for all Pipeline Stages participating in the
  //        Pipeline Cycle.
  //      - Followed by C framents for all Pipeline Stages (see (2) for an
  //        explaination)
  //    The full pipeline program is then assembled from the programs for each
  //    Pipeline Cycle.
  //
  // 2. Not all Pipeline Stages execute in every Pipeline Cycle. The full
  //    program starts with a 'fill phase' and ends with a 'flush phase',
  //    each consisting of Pipeline Cycles in which some Pipeline Stages do
  //    not participate.
  //    The exception to this is the inter-IPU copy fragment. In order to get
  //    Poplar to run these in parallel over IPUs inside the 'main' Pipeline
  //    Cycle, they must run every Pipeline Cycle for all Pipeline Stages.
  //
  //
  // To illustrate these two ideas, consider the model with three layers in
  // the forward pass:
  //
  // clang-format off
  //
  // StreamFromHost
  //    |
  //    v
  //    A->A' IPU0
  //    |  ^
  //    v  |
  //    B->B' IPU1
  //    |  ^
  //    v  |
  //    C->C' IPU2     (where X' is the grad layer of X)
  //    |
  //    v
  // StreamToHost
  //
  // A simple layer-to-pipeline-stage mapping (either set by the user or
  // inferred automatically based on Virtual Graph mapping) could be:
  //
  // Pipline Stage   Layers
  // 0               {A}
  // 1               {B}
  // 2               {C}
  //
  // After auto-grad is applied, the complete graph will then have the mapping:
  //
  // Pipline Stage   Layers
  // 0               {A}
  // 1               {B}
  // 2               {C, C'}
  // 3               {B'}
  // 4               {A'}
  //
  // Note that in order to satisfy the requirement that 'operations on a
  // Pipeline Stage have no dependencies on other Pipeline Stages', layers
  // that have dependents on other Pipeline Stages on the same IPU are
  // augmented with Stash operations in the IR that copy thier activations to
  // a FILO buffer, or stash. Also, layers that depend on other Pipeline Stages
  // on the same IPU are augmented with Restore operations that restore their
  // inputs from these stashes. The scheduling of these new operations are
  // handled by the IR scheduler.
  //
  // A pipeline with the minimum number of steps for 3 IPUs looks as follows:
  //
  // Pipeline Cycle -->
  //
  //      <-------------- fill --------------> <- main > <-------- flush --------->
  // PS0: D0.M0.| D1.M1.| D2.M2   .| D3.M3   .|D4.M4   .|       |       |    |    |
  // PS1:       |    M0.|    M1   .|    M2   .|   M3   .| M4   .|       |    |    |
  // PS2:       C       C    M0.H0.C    M1.H1.C   M2.H2.C M3.H3.C M4.H4.C    C    C
  // PS3:       |       |          |    M0   .|   M1   .| M2   .| M3   .| M4.|    |
  // PS4:       |       |          |          |   M0   .| M1   .| M2   .| M3.| M4.|
  //
  // Program fragment key:
  //   D - ToDeviceStream, M - Main, H - ToHostStream, C - IpuCopy
  //   D<i> means fragment D executes on mini-batch i, etc.
  //
  // We can see from this diagram how the full pipeline program is assembled -
  // starting in the top-left corner, serialise the 2D 'schedule' by reading
  // down the columns:
  //
  //     poplar::Program::Sequence seq;  // the full pipeline program
  //     seq.add(ps0[D]);
  //     seq.add(ps0[M]);
  //     seq.add(C);
  //     seq.add(ps0[D]);
  //     seq.add(ps0[M]);
  //     seq.add(ps1[M]);
  //     seq.add(C);  // ... etc.
  //
  // We can also look at the full program from the perspective of IPU,
  // cutilization, considering that Pipeline Stages on the same IPU must execute
  // serially:
  //
  //       <-------------- fill ---------------> <--- main ---> <--------------- flush --------------->
  // IPU0: PS0(0), PS0(1), PS0(2), PS0(3)       , PS0(4).PS4(0), PS4(1)        , PS4(2), PS4(3), PS4(4)
  // IPU1:         PS1(0), PS1(1), PS1(2).PS3(0), PS1(3).PS3(1), PS1(4).PS3(2) , PS3(3), PS3(4)
  // IPU2:                 PS2(0), PS2(1)       , PS2(2)       , PS2(3)        , PS2(4),
  //
  // The holes in this second diagram represent idle-time for an IPU. Maximizing
  // utilization is therefore a case of
  //   - Maximizing the proportion of 'main' Pipeline Cycles, achieved by having
  //     as large a 'batches per step' (or gradient accumulation factor, if
  //     gradient accumulation is enabled).
  //   - Optimally balancing the cycles required on each IPU in the main
  //     Pipeline Cycle
  //
  // clang-format on

  // Which parts of the Ir graph are run in each of the pipeline
  // fragments? Print this info here:
  std::ostringstream ss;
  ss << "\n";
  for (auto &fragid_ipudescs : pipelineDescs) {
    PipelineFragmentId fragId = fragid_ipudescs.first;
    std::string fragStr       = getStrFromPipelineFragmentId(fragId);
    ss << "\n" + fragStr + ":";
    for (auto stage_desc : pipelineDescs.at(fragId)) {
      auto vgStr = std::to_string(stage_desc.first);
      auto desc  = stage_desc.second;
      ss << "\n  ps" + vgStr + ":" + desc;
    }
  }

  ss << logging::format("\nIpuCopies: {}", pipelineIpuCopyDesc);

  ss << "\n\n";

  PipelineInfo pInfo = ir_lowering_p->pipelineInfo();

  std::map<PipelineStage, poplar::Function> mainFunctions;

  for (auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::Main)) {
    mainFunctions.insert({stage_seq.first,
                          ir_lowering_p->graph().getPoplarGraph().addFunction(
                              stage_seq.second)});
  }

  poplar::program::Sequence fill({}, {"fill"});
  for (PipelineCycle pCycle = pInfo.fillPhase.start;
       pCycle <= pInfo.fillPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, fill, ss, mainFunctions);
  }

  // All pipeline cycles in the main phase are identical. So we create the
  // program for a single cycle and repeat for mainCycles
  poplar::program::Sequence main({}, {"main"});
  int64_t mainCycles = pInfo.mainPhase.end - pInfo.mainPhase.start + 1;
  ss << "\nPipeline Cycle 'Main', " + std::to_string(mainCycles) + " cycles";
  addPipelineCycle(pInfo.mainPhase.start, main, ss, mainFunctions);

  poplar::program::Sequence flush({}, {"flush"});
  for (PipelineCycle pCycle = pInfo.flushPhase.start;
       pCycle <= pInfo.flushPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, flush, ss, mainFunctions);
  }

  logging::devicex::debug("Pipelining program construction summary:");
  logging::devicex::debug(ss.str());

  poplar::program::Sequence inner({}, {"inner"});

  inner.add(fill);
  // This is the inner main cycles loop, if doing pipelining without gradient
  // accumulation, this the batches per step loop, as batch size = micro_batch
  // size
  inner.add(poplar::program::Repeat(
      static_cast<uint32_t>(mainCycles), main, {"inerLoop"}));
  inner.add(flush);

  poplar::program::Sequence outer({}, {"outer"});

  outer.add(initFragment());

  if (!ir_lowering_p->getOuterLoopFragEmpty()) {

    inner.add(accumulateOuterFragment());
    // If doing gradient accumulation, the inner loop is over mini batches,
    // and this outer loop loops over multiple batches per step.
    auto bps = ir_lowering_p->ir().getDataFlow().batchesPerStep();
    outer.add(poplar::program::Repeat(bps, inner, {"outerloop"}));
  } else {
    // No gradient accumulation, so just add one iteration of the inner program.
    outer.add(inner);
  }

  outer.add(toHostFinalCopyFragment());

  return outer;
}

poplar::program::Sequence PopPrograms::program() const {
  auto instrumentations =
      ir_lowering_p->ir().getSessionOptions().hardwareInstrumentations;

  poplar::program::Sequence outer({}, {"outer"});

  if (ir_lowering_p->ir().getSessionOptions().enableExplicitMainLoops) {
    outer.add(initFragment());
    outer.add(preForwardFragment());
    outer.add(forwardFragment());
    outer.add(backwardFragment());
    outer.add(toHostFinalCopyFragment());
  } else {
    if (ir_lowering_p->ir().getSessionOptions().enablePipelining) {
      outer.add(getFullProgramFromPipelineFragments());
    } else {
      poplar::program::Sequence prog({}, {"program"});
      prog.add(preForwardFragment());
      prog.add(forwardFragment());
      prog.add(backwardFragment());

      outer.add(initFragment());

      auto accumulationFactor = ir_lowering_p->getAccumulationFactor();
      if (!ir_lowering_p->getOuterLoopFragEmpty()) {
        logging::devicex::trace(
            "Adding gradient accumulation repeat loop with {} iterations",
            accumulationFactor);
        prog = {poplar::program::Repeat(
            accumulationFactor, prog, {"accumulationLoop"})};
        prog.add(accumulateOuterFragment());
      }

      if (ir_lowering_p->ir()
              .getSessionOptions()
              .instrumentWithHardwareCycleCounter &&
          instrumentations.find(Instrumentation::Inner) !=
              instrumentations.end()) {
        // Instrument first tile of every IPU for inner program
        auto numIpus = ir_lowering_p->getDeviceInfo()->getNumIpus() /
                       ir_lowering_p->getReplicationFactor();
        for (int64_t i = 0; i < numIpus; ++i) {
          std::stringstream ss;
          // String to identify instrumentation
          ss << "inner_ipu_" << i;
          ir_lowering_p->instrumentWithHardwareCycleCounter(
              prog,
              i * static_cast<int64_t>(
                      ir_lowering_p->getDeviceInfo()->getTilesPerIPU()),
              ss.str());
        }
      }

      auto batchesPerStep = ir_lowering_p->ir().getDataFlow().batchesPerStep();
      // BatchesPerStep loop
      logging::devicex::trace("Adding batches per step loop with {} iterations",
                              batchesPerStep);
      outer.add(
          poplar::program::Repeat(batchesPerStep, prog, {"batchesPerStep"}));
      outer.add(toHostFinalCopyFragment());
    }
  }

  if (ir_lowering_p->ir()
          .getSessionOptions()
          .instrumentWithHardwareCycleCounter &&
      instrumentations.find(Instrumentation::Outer) != instrumentations.end()) {
    ir_lowering_p->instrumentWithHardwareCycleCounter(outer);
  }

  return outer;
}

poplar::program::Sequence PopPrograms::weightsToHost() const {
  poplar::program::Sequence prog({}, {"weightsToHost"});
  prog.add(weightsToHostFragment());
  return prog;
}

const std::vector<poplar::program::Program> PopPrograms::progs() const {
  std::vector<poplar::program::Program> ps(ProgramIndex::N);

  ps[ProgramIndex::WeightsFromHost]        = weightsFromHost();
  ps[ProgramIndex::OptimizerFromHost]      = optimizerFromHost();
  ps[ProgramIndex::SetRandomSeedFromHost]  = setRandomSeedFromHost();
  ps[ProgramIndex::RngStateFromHost]       = rngStateFromHost();
  ps[ProgramIndex::Program]                = program();
  ps[ProgramIndex::RngStateToHost]         = rngStateToHost();
  ps[ProgramIndex::WeightstoHost]          = weightsToHost();
  ps[ProgramIndex::CycleCountTensortoHost] = cycleCountTensorToHost();

  return ps;
}

poplar::program::Sequence &
PopPrograms::programFragment(PopPrograms::ProgramFragmentIndex index) {
  return seqs.at(static_cast<int>(index));
}

int PopPrograms::getNumFragments(const Graph &graph) const {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  if (scopeIt == scopeSeqs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  } else {
    return scopeIt->second.size();
  }
}

std::vector<poplar::program::Sequence> &
PopPrograms::scopeFragments(const Graph &graph) {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  if (scopeIt == scopeSeqs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  } else {
    return scopeIt->second;
  }
}

poplar::program::Sequence &
PopPrograms::scopeFragment(const Graph &graph, SubgraphPartIndex subgraphPart) {
  return scopeSeqs.at(graph.id.str()).at(subgraphPart);
}

bool PopPrograms::containsFragments(const Graph &graph) const {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  return (scopeIt != scopeSeqs.end());
}

bool PopPrograms::containsFragment(const Graph &graph,
                                   SubgraphPartIndex subgraphPart) const {

  auto scopeIt = scopeSeqs.find(graph.id.str());
  return (scopeIt != scopeSeqs.end()) &&
         (subgraphPart < scopeIt->second.size());
}

void PopPrograms::createFragment(const Graph &graph,
                                 SubgraphPartIndex subgraphPart) {

  // We only populate scopeSeqs here, funcs needs to be populated after
  // sequences are grown because sequences are cloned on addFunction.
  auto scopeIt = scopeSeqs.find(graph.id.str());

  // Check if this graph has a vector already.
  if (scopeIt != scopeSeqs.end()) {
    // Vectors exists, check it contains the subgraphPart.
    auto &seqs = scopeIt->second;
    if (seqs.size() < subgraphPart + 1) {
      // Check funcs matches scopeSeqs.
      assert(funcs.size() < subgraphPart + 1);
      // Resize scopeSeqs. The funcs vector will be resized to match below.
      seqs.resize(subgraphPart + 1);
    }
  } else {
    // Vector does not exist, create one that contains the subgraph part.
    std::vector<poplar::program::Sequence> seqs;

    for (size_t part = 0; part <= subgraphPart; ++part) {
      std::stringstream dbgCtx;
      if (graph.id.str() == "") {
        dbgCtx << "main_graph/" << part;
      } else {
        dbgCtx << graph.id.str() << "/" << part;
      }
      seqs.push_back(poplar::program::Sequence({}, dbgCtx.str()));
    }

    scopeSeqs.insert({graph.id.str(), seqs});
  }
}

std::vector<poplar::Function> &
PopPrograms::getFragmentFunctions(const Graph &graph,
                                  snap::Graph &snapGraph) {

  auto seq2func = [&](poplar::program::Sequence &seq) {
    return snapGraph.getPoplarGraph().addFunction(seq);
  };

  auto funcsIt = funcs.find(graph.id.str());
  if (funcsIt == funcs.end()) {
    // Funcs was never populated. Populate it now.
    funcsIt     = funcs.insert({graph.id.str(), {}}).first;
    auto &funcs = funcsIt->second;
    auto &seqs  = scopeSeqs.at(graph.id.str());
    std::transform(
        seqs.begin(), seqs.end(), std::back_inserter(funcs), seq2func);
  }

  if (funcsIt == funcs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  }
  return funcsIt->second;
}

poplar::Function &
PopPrograms::getFragmentFunction(const Graph &graph,
                                 SubgraphPartIndex subgraphPart,
                                 snap::Graph &snapGraph) {

  auto &funcs = getFragmentFunctions(graph, snapGraph);

  if (subgraphPart >= funcs.size()) {
    throw error("There is no scope fragment for {}, part {}",
                graph.getGraphString(),
                subgraphPart);
  } else {
    return funcs[subgraphPart];
  }
}

bool PopPrograms::hasBeenRecomputed(OpId id, ExecutionPhase phase) const {
  auto itHas = (beenRecomputed.find({id, phase}) != beenRecomputed.end());
  return itHas;
}

void PopPrograms::recordRecomputed(OpId id, ExecutionPhase phase) {
  beenRecomputed.insert({id, phase});
}

std::vector<poplar::program::Sequence>::iterator
PopPrograms::recomputeFragment(OpId id) {
  auto found = recomputeSeqs.find(id);
  if (found == recomputeSeqs.end()) {
    throw error("Recompute Fragment for Op {} has not been created.", id);
  }
  return found->second.begin();
}

SequenceMap::SequenceInterval PopPrograms::createRecomputeFragment(OpId id) {
  recomputeSeqs.insert({id, {poplar::program::Sequence{}}});
  return SequenceMap::SequenceInterval(recomputeSeqs[id].begin(),
                                       recomputeSeqs[id].end());
}

poplar::program::Sequence &
PopPrograms::forwardOrBackwardFragment(ScheduledPreLoss preLoss) {
  switch (preLoss) {
  case ScheduledPreLoss::Yes: {
    return forwardFragment();
  }
  case ScheduledPreLoss::No: {
    return backwardFragment();
  }
  case ScheduledPreLoss::Undefined: {
    throw error("There is no fragment for Undefined SchedulePreLoss");
  }
  default:
    throw error("Unknown SchedulePreLoss fragment");
  }
}

poplar::program::Sequence &
PopPrograms::pipelineFragment(PipelineStage pipelineStage,
                              PipelineFragmentId frag,
                              const std::string &desc) {
  auto foundFrag = pipelineSeqs.find(frag);
  if (foundFrag != pipelineSeqs.end()) {
    auto foundPipelineStage = pipelineSeqs.at(frag).find(pipelineStage);
    if (foundPipelineStage != pipelineSeqs.at(frag).end()) {
      pipelineDescs.at(frag).at(pipelineStage).append("\n    " + desc);
      return pipelineSeqs.at(frag).at(pipelineStage);
    } else {
      pipelineDescs.at(frag).insert({pipelineStage, "\n    " + desc});
      pipelineSeqs.at(frag).insert(
          {pipelineStage, poplar::program::Sequence{}});
      return pipelineSeqs.at(frag).at(pipelineStage);
    }
  } else {
    pipelineDescs.insert({frag, {{pipelineStage, "\n    " + desc}}});
    pipelineSeqs.insert({frag, {{pipelineStage, poplar::program::Sequence{}}}});
    return pipelineSeqs.at(frag).at(pipelineStage);
  }
}

poplar::program::Sequence &
PopPrograms::pipelineMainFragment(PipelineStage pipelineStage,
                                  const std::string &desc) {
  return pipelineFragment(pipelineStage, PipelineFragmentId::Main, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineToDeviceStreamFragment(PipelineStage pipelineStage,
                                            const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToDeviceStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineToHostStreamFragment(PipelineStage pipelineStage,
                                          const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToHostStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineIpuCopyFragment(const std::string &desc) {
  pipelineIpuCopyDesc.append("\n    " + desc);
  return pipelineIpuCopySeq;
}

std::string
PopPrograms::getStrFromPipelineFragmentId(PipelineFragmentId fragId) const {
  switch (fragId) {
  case PipelineFragmentId::ToDeviceStream: {
    return "ToDeviceStream";
  }
  case PipelineFragmentId::Main: {
    return "Main";
  }
  case PipelineFragmentId::ToHostStream: {
    return "ToHostStream";
  }
  default:
    throw error("Cannot return string for PipelineFragmentId '{}'",
                static_cast<int>(fragId));
  }
}

} // namespace popx
} // namespace popart
