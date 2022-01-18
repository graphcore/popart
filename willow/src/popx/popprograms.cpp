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
#include <snap/Program.hpp>

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
  case PopPrograms::ProgramFragmentIndex::RandomSeedFromHost: {
    out << "RandomSeedFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RandomSeedToHost: {
    out << "RandomSeedToHost";
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
  case PopPrograms::ProgramFragmentIndex::WeightsToHost: {
    out << "WeightsToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::ToHostFinalCopy: {
    out << "ToHostFinalCopy";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::CycleCountTensorToHost: {
    out << "CycleCountTensorToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::N: {
    out << "N";
    break;
  }
  default: { throw internal_error("Invalid value for ProgramFragmentIndex"); }
  };
  return out;
}

PopPrograms::PopPrograms(IrLowering *ir_lowering_p_)
    : ir_lowering_p(ir_lowering_p_) {}

void PopPrograms::initWithSnapGraph(snap::Graph &g) {
  // Populate seqs with Sequences that have names.
  for (int i = 0; i < static_cast<int>(ProgramFragmentIndex::N); ++i) {
    std::stringstream ss;
    ss << static_cast<ProgramFragmentIndex>(i);
    seqs.push_back(snap::program::Sequence({ss.str()}, g));
  }

  pipelineIpuCopySeq.reset(new snap::program::Sequence(g));
}

const snap::program::Sequence &
PopPrograms::streamWeightsFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}
snap::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}

const snap::program::Sequence &
PopPrograms::streamOptimizerFromHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}
snap::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}

const snap::program::Sequence &PopPrograms::randomSeedFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedFromHost));
}
snap::program::Sequence &PopPrograms::randomSeedFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedFromHost));
}
const snap::program::Sequence &PopPrograms::randomSeedToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedToHost));
}
snap::program::Sequence &PopPrograms::randomSeedToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedToHost));
}

const snap::program::Sequence &PopPrograms::rngStateFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

snap::program::Sequence &PopPrograms::rngStateFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

const snap::program::Sequence &PopPrograms::rngStateToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

snap::program::Sequence &PopPrograms::rngStateToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

const snap::program::Sequence &
PopPrograms::cycleCountTensorToHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensorToHost));
}
snap::program::Sequence &PopPrograms::cycleCountTensorToHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensorToHost));
}

const snap::program::Sequence &PopPrograms::initFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

snap::program::Sequence &PopPrograms::initFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

const snap::program::Sequence &PopPrograms::preForwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

snap::program::Sequence &PopPrograms::preForwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

const snap::program::Sequence &PopPrograms::forwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

snap::program::Sequence &PopPrograms::forwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

const snap::program::Sequence &PopPrograms::backwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

snap::program::Sequence &PopPrograms::backwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

const snap::program::Sequence &PopPrograms::toHostFinalCopyFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

snap::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

const snap::program::Sequence &PopPrograms::accumulateOuterFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

snap::program::Sequence &PopPrograms::accumulateOuterFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

const snap::program::Sequence &PopPrograms::weightsToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightsToHost));
}

snap::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightsToHost));
}

snap::program::Sequence PopPrograms::weightsFromHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"weightsFromHost"},
                               ir_lowering_p->graph());
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::optimizerFromHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"optimizerFromHost"},
                               ir_lowering_p->graph());
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::randomSeedFromHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"randomSeedFromHost"},
                               ir_lowering_p->graph());
  prog.add(randomSeedFromHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::randomSeedToHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"randomSeedToHost"},
                               ir_lowering_p->graph());
  prog.add(randomSeedToHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::cycleCountTensorToHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"cycleCountTensorToHost"},
                               ir_lowering_p->graph());
  prog.add(cycleCountTensorToHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::rngStateFromHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"rngStateFromHost"},
                               ir_lowering_p->graph());
  prog.add(rngStateFromHostFragment());
  return prog;
}

snap::program::Sequence PopPrograms::rngStateToHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"rngStateToHost"},
                               ir_lowering_p->graph());
  prog.add(rngStateToHostFragment());
  return prog;
}

void PopPrograms::addPipelineCycle(
    PipelineCycle pCycle,
    snap::program::Sequence &sq,
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

  PipelineInfo pInfo = ir_lowering_p->ir().pipelineInfo();

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
      sq.add(snap::program::Call(ir_lowering_p->graph(), stage_seq.second));
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
  sq.add(*pipelineIpuCopySeq.get());
}

snap::program::Sequence
PopPrograms::getFullProgramFromPipelineFragments() const {
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

  PipelineInfo pInfo = ir_lowering_p->ir().pipelineInfo();

  std::map<PipelineStage, poplar::Function> mainFunctions;

  for (auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::Main)) {
    const snap::program::Sequence &sequence = stage_seq.second;
    mainFunctions.insert(
        {stage_seq.first, ir_lowering_p->graph().addFunction(sequence)});
  }

  snap::program::Sequence fill(poplar::DebugContext{"fill"},
                               ir_lowering_p->graph());
  for (PipelineCycle pCycle = pInfo.fillPhase.start;
       pCycle <= pInfo.fillPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, fill, ss, mainFunctions);
  }

  // All pipeline cycles in the main phase are identical. So we create the
  // program for a single cycle and repeat for mainCycles
  snap::program::Sequence main(poplar::DebugContext{"main"},
                               ir_lowering_p->graph());
  int64_t mainCycles = pInfo.getMainCycles();
  ss << "\nPipeline Cycle 'Main', " + std::to_string(mainCycles) + " cycles";
  addPipelineCycle(pInfo.mainPhase.start, main, ss, mainFunctions);

  snap::program::Sequence flush(poplar::DebugContext{"flush"},
                                ir_lowering_p->graph());
  for (PipelineCycle pCycle = pInfo.flushPhase.start;
       pCycle <= pInfo.flushPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, flush, ss, mainFunctions);
  }

  logging::devicex::debug("Pipelining program construction summary:");
  logging::devicex::debug(ss.str());

  snap::program::Sequence inner(poplar::DebugContext{"inner"},
                                ir_lowering_p->graph());

  inner.add(fill);
  // This is the inner main cycles loop, if doing pipelining without gradient
  // accumulation, this the batches per step loop, as batch size = micro_batch
  // size
  inner.add(snap::program::Repeat(
      static_cast<uint32_t>(mainCycles), main, {"inerLoop"}));
  inner.add(flush);

  snap::program::Sequence outer(poplar::DebugContext{"outer"},
                                ir_lowering_p->graph());

  outer.add(initFragment());

  if (!ir_lowering_p->getOuterLoopFragEmpty()) {

    inner.add(accumulateOuterFragment());
    // If doing gradient accumulation, the inner loop is over mini batches,
    // and this outer loop loops over multiple batches per step.
    auto bps = ir_lowering_p->ir().getDataFlow().batchesPerStep();
    outer.add(snap::program::Repeat(bps, inner, {"outerloop"}));
  } else {
    // No gradient accumulation, so just add one iteration of the inner program.
    outer.add(inner);
  }

  outer.add(toHostFinalCopyFragment());

  return outer;
}

snap::program::Sequence PopPrograms::program() const {
  const auto &opts      = ir_lowering_p->ir().getSessionOptions();
  auto instrumentations = opts.hardwareInstrumentations;

  snap::program::Sequence outer(poplar::DebugContext{"outer"},
                                ir_lowering_p->graph());

  if (opts.enableExplicitMainLoops) {
    outer.add(initFragment());
    outer.add(preForwardFragment());
    outer.add(forwardFragment());
    outer.add(backwardFragment());
    outer.add(toHostFinalCopyFragment());
  } else {
    if (opts.implicitPipeliningEnabled()) {
      outer.add(getFullProgramFromPipelineFragments());
    } else {
      snap::program::Sequence prog(poplar::DebugContext{"program"},
                                   ir_lowering_p->graph());
      prog.add(preForwardFragment());
      prog.add(forwardFragment());
      prog.add(backwardFragment());

      outer.add(initFragment());

      auto accumulationFactor = ir_lowering_p->getAccumulationFactor();
      if (!ir_lowering_p->getOuterLoopFragEmpty()) {
        logging::devicex::trace(
            "Adding gradient accumulation repeat loop with {} iterations",
            accumulationFactor);
        snap::program::Repeat repeat(
            accumulationFactor, prog, {"accumulationLoop"});
        prog = snap::program::Sequence(ir_lowering_p->graph());
        prog.add(repeat);
        prog.add(accumulateOuterFragment());
      }

      if (opts.instrumentWithHardwareCycleCounter &&
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
          snap::program::Repeat(batchesPerStep, prog, {"batchesPerStep"}));
      outer.add(toHostFinalCopyFragment());
    }
  }

  if (opts.instrumentWithHardwareCycleCounter &&
      instrumentations.find(Instrumentation::Outer) != instrumentations.end()) {
    ir_lowering_p->instrumentWithHardwareCycleCounter(outer);
  }

  return outer;
}

snap::program::Sequence PopPrograms::weightsToHost() const {
  snap::program::Sequence prog(poplar::DebugContext{"weightsToHost"},
                               ir_lowering_p->graph());
  prog.add(weightsToHostFragment());
  return prog;
}

const std::vector<snap::program::Program> PopPrograms::progs() const {
  std::vector<snap::program::Program> ps(ProgramIndex::N);

  ps[ProgramIndex::WeightsFromHost]        = weightsFromHost();
  ps[ProgramIndex::OptimizerFromHost]      = optimizerFromHost();
  ps[ProgramIndex::RandomSeedFromHost]     = randomSeedFromHost();
  ps[ProgramIndex::RandomSeedToHost]       = randomSeedToHost();
  ps[ProgramIndex::RngStateFromHost]       = rngStateFromHost();
  ps[ProgramIndex::Program]                = program();
  ps[ProgramIndex::RngStateToHost]         = rngStateToHost();
  ps[ProgramIndex::WeightsToHost]          = weightsToHost();
  ps[ProgramIndex::CycleCountTensorToHost] = cycleCountTensorToHost();

  return ps;
}

snap::program::Sequence &
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

std::vector<snap::program::Sequence> &
PopPrograms::scopeFragments(const Graph &graph) {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  if (scopeIt == scopeSeqs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  } else {
    return scopeIt->second;
  }
}

snap::program::Sequence &
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
      seqs.resize(subgraphPart + 1, {ir_lowering_p->graph()});
    }
  } else {
    // Vector does not exist, create one that contains the subgraph part.
    std::vector<snap::program::Sequence> seqs;

    for (size_t part = 0; part <= subgraphPart; ++part) {
      std::stringstream dbgCtx;
      if (graph.id.str() == "") {
        dbgCtx << "main_graph/" << part;
      } else {
        dbgCtx << graph.id.str() << "/" << part;
      }
      seqs.push_back(
          snap::program::Sequence(dbgCtx.str(), ir_lowering_p->graph()));
    }

    scopeSeqs.insert({graph.id.str(), seqs});
  }
}

std::vector<poplar::Function> &
PopPrograms::getFragmentFunctions(const Graph &graph, snap::Graph &snapGraph) {

  auto seq2func = [&](snap::program::Sequence &seq) {
    return snapGraph.addFunction(seq);
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

std::vector<snap::program::Sequence>::iterator
PopPrograms::recomputeFragment(OpId id) {
  auto found = recomputeSeqs.find(id);
  if (found == recomputeSeqs.end()) {
    throw error("Recompute Fragment for Op {} has not been created.", id);
  }
  return found->second.begin();
}

SequenceMap::SequenceInterval PopPrograms::createRecomputeFragment(OpId id) {
  recomputeSeqs.insert({id, {snap::program::Sequence{ir_lowering_p->graph()}}});
  return SequenceMap::SequenceInterval(recomputeSeqs[id].begin(),
                                       recomputeSeqs[id].end());
}

snap::program::Sequence &
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

snap::program::Sequence &
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
          {pipelineStage, snap::program::Sequence{ir_lowering_p->graph()}});
      return pipelineSeqs.at(frag).at(pipelineStage);
    }
  } else {
    pipelineDescs.insert({frag, {{pipelineStage, "\n    " + desc}}});
    pipelineSeqs.insert(
        {frag,
         {{pipelineStage, snap::program::Sequence{ir_lowering_p->graph()}}}});
    return pipelineSeqs.at(frag).at(pipelineStage);
  }
}

snap::program::Sequence &
PopPrograms::pipelineMainFragment(PipelineStage pipelineStage,
                                  const std::string &desc) {
  return pipelineFragment(pipelineStage, PipelineFragmentId::Main, desc);
}

snap::program::Sequence &
PopPrograms::pipelineToDeviceStreamFragment(PipelineStage pipelineStage,
                                            const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToDeviceStream, desc);
}

snap::program::Sequence &
PopPrograms::pipelineToHostStreamFragment(PipelineStage pipelineStage,
                                          const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToHostStream, desc);
}

snap::program::Sequence &
PopPrograms::pipelineIpuCopyFragment(const std::string &desc) {
  pipelineIpuCopyDesc.append("\n    " + desc);
  return *pipelineIpuCopySeq.get();
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
