// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <array>
#include <boost/filesystem.hpp>
#include <filereader.hpp>
#include <memory>
#include <numeric>
#include <queue>
#include <schedulegraphgrower.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/scheduler.hpp>

#include <poprithms/logging/logging.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/kahntiebreaker.hpp>
#include <poprithms/schedule/shift/rotationtermination.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/settings.hpp>
#include <poprithms/schedule/shift/solutioncache.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>
#include <poparttracepoint.hpp>

namespace {

using namespace poprithms::schedule;

using shift::AllocAddress;
using shift::AllocWeight;
using shift::OpAddress;
using shift::ScheduleIndex;
using KahnTieBreaker = shift::KahnTieBreaker;

} // namespace

namespace popart {

KahnTieBreaker kahnTieBreakerFromString(const std::string &ktbString) {
  std::string ktbLower = ktbString;
  std::transform(ktbLower.begin(),
                 ktbLower.end(),
                 ktbLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  KahnTieBreaker ktb;

  if (ktbLower == "fifo") {
    ktb = KahnTieBreaker::FIFO;
  } else if (ktbLower == "greedy") {
    ktb = KahnTieBreaker::GREEDY;
  } else if (ktbLower == "random") {
    ktb = KahnTieBreaker::RANDOM;
  } else {
    throw error("Unrecognised KahnTieBreaker, {}. Should be one of \"fifo\", "
                "\"greedy\", and \"random\".",
                ktbString);
  }

  return ktb;
}

void serializePoprithmsGraph(
    const ShiftGraphGrower *grower,
    const std::string &serializedPoprithmsShiftGraphsDir) {

  auto dirName =
      boost::filesystem::canonical(serializedPoprithmsShiftGraphsDir).string();

  if (!boost::filesystem::exists(dirName)) {
    std::ostringstream oss;
    oss << "No directory, `" << dirName << "' exists. "
        << " The SessionOptions directory serializedPoprithmsShiftGraphsDir "
        << "must already exist, PopART will not create it. "
        << "If you do not want to serialize Poprithms Graph, set "
        << " serializePoprithmsShiftGraphs "
        << " to false.";
    throw error(oss.str());
  }
  auto getTargetName = [dirName](int i) {
    return io::appendDirFn(
        dirName, "poprithms_shift_graph_" + std::to_string(i) + ".json");
  };

  // iterate through file names until a non-existant one is found
  int modelNumber{0};
  while (boost::filesystem::exists(getTargetName(modelNumber))) {
    ++modelNumber;
  }
  auto filename = getTargetName(modelNumber);

  std::ofstream ofs;
  ofs.open(filename);
  if (!ofs.is_open()) {
    throw error("Failed to open file {}", filename);
  }
  ofs << grower->getSerializationString();
  ofs.close();

  logging::ir::info("[Scheduler] written {} ", filename);
}

enum class AnnotateForFasterSwapping { No = 0, Yes };

void defaultAnnotate(ShiftGraphGrower *grower,
                     const OpsBeforeKey &gCons,
                     const Graph &pg,
                     const RequireOptimalSchedule requireOptimalSchedule,
                     const bool respectExecutionPhases,
                     AnnotateForFasterSwapping fastSwap) {

  const auto annotateForFasterSwapping =
      (fastSwap == AnnotateForFasterSwapping::Yes);

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "[Scheduler] defaultAnnotate");

  grower->setBasic();
  grower->appendGCons(gCons);
  if (respectExecutionPhases &&
      pg.getIr().getSessionOptions().executionPhaseSettings.phases > 1) {
    grower->annotateExecutionPhase();
  }
  if (pg.getIr().getSessionOptions().enablePipelining &&
      annotateForFasterSwapping) {
    grower->annotatePipelineStages();
  }
  if ((pg.getIr().autoRecomputationEnabled() ||
       pg.getIr().getMainGraph().hasUserRecomputeOps()) &&
      !pg.getIr().getSessionOptions().explicitRecomputation) {
    grower->annotateToLossFromLoss();
  }

  if (annotateForFasterSwapping) {
    grower->annotateAccumulateOuterFragmentOps();
  }
  grower->annotateExecutionContext();

  if (requireOptimalSchedule == RequireOptimalSchedule::Yes) {
    // No need for priorities if we're happy with any schedule.
    grower->annotatePriorities();
  }
}

std::vector<Op *>
Scheduler::getSchedule(const OpsBeforeKey &gCons,
                       const Graph &pg,
                       const RequireOptimalSchedule requireOptimalSchedule,
                       const bool respectExecutionPhases,
                       const double timeLimitScheduler,
                       const int64_t swapLimitScheduler,
                       const std::string &kahnTieBreakerString) {

  const auto sw = pg.getIr().timePartitionLogger().scopedStopwatch(
      "Preparation for scheduling");

  // When the Graph contains no Ops, return an empty schedule (this 0-op edge
  // case unfortunately causes issues later if left unhandled here.)
  if (pg.getOps().empty()) {
    return {};
  }

  using namespace poprithms::schedule;

  const auto rotationTermination =
      shift::RotationTermination(timeLimitScheduler, swapLimitScheduler);

  // Use transitive closure optimizations of
  // 1) the time you might be running the shift step for is large.
  // 2) there are not too many Ops in the Graph.
  const auto useTransitiveClosureOptimizations =
      rotationTermination.longerThan({10.0, 100}) &&
      pg.getOps().size() <=
          pg.getIr().getSessionOptions().transitiveClosureOptimizationThreshold;

  std::ostringstream oss;

  oss << "RotationTermination=("
      << "rotations=" << rotationTermination.maxRotations()
      << ", seconds=" << rotationTermination.maxSeconds() << ")"
      << " and nOps=" << pg.getOps().size() << ". ";
  logging::ir::debug(oss.str());

  const auto transitiveClosureOptimizations =
      useTransitiveClosureOptimizations
          ? shift::TransitiveClosureOptimizations::allOn()
          : shift::TransitiveClosureOptimizations::allOff();

  // The complete set of options for the scheduler:
  poprithms::schedule::shift::Settings settings(
      kahnTieBreakerFromString(kahnTieBreakerString),
      transitiveClosureOptimizations,
      rotationTermination,
      shift::Settings::defaultRotationAlgo(),
      shift::Settings::defaultSeed(),
      shift::DebugMode::Off);

  // The thing for mapping PopART to a poprithms Graph.
  auto grower = std::make_unique<ShiftGraphGrower>(pg);
  defaultAnnotate(grower.get(),
                  gCons,
                  pg,
                  requireOptimalSchedule,
                  respectExecutionPhases,
                  useTransitiveClosureOptimizations
                      ? AnnotateForFasterSwapping::Yes
                      : AnnotateForFasterSwapping::No);

  if (!pg.getIr()
           .getSessionOptions()
           .serializedPoprithmsShiftGraphsDir.empty()) {

    auto scopedStopwatch = pg.getIr().timePartitionLogger().scopedStopwatch(
        "Serializing shift::Graph");
    serializePoprithmsGraph(
        grower.get(),
        pg.getIr().getSessionOptions().serializedPoprithmsShiftGraphsDir);
  }

  // perform the actual actual scheduling:
  grower->initialize(settings, *cacher);
  const std::vector<Op *> finalSchedule = grower->getSchedule();

  return finalSchedule;
}

bool Scheduler::isSchedulable(const OpsBeforeKey &gCons,
                              const Graph &pg,
                              bool respectExecutionPhases) const {
  ShiftGraphGrower grower(pg);
  defaultAnnotate(&grower,
                  gCons,
                  pg,
                  RequireOptimalSchedule::No,
                  respectExecutionPhases,
                  AnnotateForFasterSwapping::No);
  return grower.isSchedulable();
}

Scheduler::Scheduler() : cacher(std::make_unique<shift::SolutionCache>()) {}
Scheduler::~Scheduler() = default;

} // namespace popart
