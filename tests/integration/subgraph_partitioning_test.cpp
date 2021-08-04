// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SubgraphPartitioningTest

#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/call.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/subgraphcopyingstrategy.hpp>
#include <popart/subgraphpartitioner.hpp>

using namespace popart;
using LivenessNode = liveness::LivenessNode;
using OpStatus     = liveness::OpStatus;

// In lieu of writing unit tests for LivenessAnalyzer, SubgraphCopyingStrategy
// and SubgraphPartitioner separately, the tests in this file test their
// combined behaviour. The classes have dependencies as follows:
//
// +-------------------------+           +-------------------------+
// | LivenessAnalyzer        +---------->+ SubgraphCopyingStrategy |
// |                         +<----------+                         |
// +-------------------------+           +-------------------------+
//    ^
//    |
// +-------------------------+
// | SubgraphPartitioner     |
// +-------------------------+
//
// We test two things:
//
//   * LivenessAnalyzer uses SubgraphCopyingStrategy to determine where to put
//     copies between subgraphs in the schedule. We test that this
//     placement is as we expect.
//   * SubgraphPartitioner determines how subgraphs are partitioned based on
//     the LivenessAnalyzer's global schedule. We test that this partitioning
//     is as we expect.

namespace {

// Shorthands.
auto OnEnterAndExit = SubgraphCopyingStrategy::OnEnterAndExit;
auto JustInTime     = SubgraphCopyingStrategy::JustInTime;

// Layout params.
constexpr size_t CALLSTACK_PADDING = 15;
constexpr size_t STATUS_PADDING    = 25;
constexpr size_t DETAILS_PADDING   = 45;

/**
 * A struct that holds our test objects.
 */
struct TestObjects {
  std::unique_ptr<Ir> ir;
  std::unique_ptr<liveness::LivenessAnalyzer> livenessAnalyzer;
  std::unique_ptr<liveness::SubgraphCopyingStrategy> subgraphCopyingStrategy;
  std::unique_ptr<liveness::SubgraphPartitioner> subgraphPartitioner;
};

/**
 * Create a LivenessAnalyzer, SubgraphCopyingStrategy and SubgraphPartitioner,
 * link them up, and call apply.
 */
TestObjects setup(std::function<void(Builder *)> modelBuilder,
                  SubgraphCopyingStrategy copyStrat) {

  // Build model.
  auto builder = Builder::create();
  modelBuilder(builder.get());

  // Create session options.
  SessionOptions opts;
  opts.enableOutlining         = false;
  opts.aliasZeroCopy           = false;
  opts.subgraphCopyingStrategy = copyStrat;

  // Create IR from model.
  auto ir = std::make_unique<Ir>();
  ir->setUserOptions(opts);
  ir->setOnnxModel(io::getModelFromString(builder->getModelProto()));
  ir->registerInputTensors();
  ir->constructForwards();

  // Create the test objects.
  std::unique_ptr<liveness::SubgraphCopyingStrategy> copyingStrat;

  switch (opts.subgraphCopyingStrategy) {
  case SubgraphCopyingStrategy::OnEnterAndExit:
    copyingStrat =
        std::make_unique<liveness::OnEnterAndExitSubgraphCopyingStrategy>();
    break;
  case SubgraphCopyingStrategy::JustInTime:
    copyingStrat =
        std::make_unique<liveness::JustInTimeSubgraphCopyingStrategy>();
    copyingStrat->setIr(ir.get());
    break;
  default:
    assert(false);
  }

  auto livenessAnalyzer = std::make_unique<liveness::LivenessAnalyzer>(
      ir.get(), copyingStrat.get());
  auto subgraphPart = std::make_unique<liveness::SubgraphPartitioner>();

  // Meet dependencies.
  copyingStrat->setIr(ir.get());
  copyingStrat->setLivenessAnalyzer(livenessAnalyzer.get());
  subgraphPart->setIr(ir.get());
  subgraphPart->setLivenessAnalyzer(livenessAnalyzer.get());

  // Call their logic.
  copyingStrat->apply();
  livenessAnalyzer->apply();
  subgraphPart->apply();

  return {std::move(ir),
          std::move(livenessAnalyzer),
          std::move(copyingStrat),
          std::move(subgraphPart)};
}

/**
 * A way to represent an expected liveness node, using strings.
 */
struct ExpectedLivenessNode {
  // Represent callstack by graph names, e.g. "/sg0"
  const char *callstackStr;
  // Represent OpStatus, e.g., "Enter".
  const char *statusStr;
  // Represent node's details.
  const char *detailsStr;
};

using ExpectedLivenessSchedule = std::vector<ExpectedLivenessNode>;

/**
 * For a given liveness node, get a string representation of the call stack.
 */
std::string getCallstackStr(const LivenessNode &node) {
  std::string result = "";
  bool isFirst       = true;
  for (auto op : node.getCallStack()) {
    if (!isFirst)
      result += "|";
    result += op->getGraph().id.str();
    isFirst = false;
  }
  return result;
}

/**
 * For a given liveness node, get a string representation of the status.
 */
std::string getStatusStr(const LivenessNode &node) {
  std::stringstream ss;
  if (node.getStatus() != OpStatus::Normal) {
    ss << node.getStatus();
  } else {
    ss << node.getOp()->opid;
  }
  return ss.str();
}

/**
 * For a given liveness node, get a string representation of the call stack.
 */
std::string getDetailsStr(const LivenessNode &node) {
  std::stringstream ss;
  switch (node.getStatus()) {
  case OpStatus::Enter: {
    break;
  }
  case OpStatus::Exit: {
    break;
  }
  case OpStatus::CopyInput: {
    ss << "'" << node.getTensorIds().second << "'"
       << " := "
       << "'" << node.getTensorIds().first << "'";
    break;
  }
  case OpStatus::CopyOutput: {
    ss << "'" << node.getTensorIds().first << "'"
       << " := "
       << "'" << node.getTensorIds().second << "'";
    break;
  }
  case OpStatus::CopyModified: {
    ss << "'" << node.getTensorIds().first << "'"
       << " := "
       << "'" << node.getTensorIds().second << "'";
    break;
  }
  case OpStatus::CopyLoopCarried: {
    // We don't really care about loop carried, so don't output details.
    break;
  }
  case OpStatus::Normal: {
    bool firstOutput = true;
    ss << "{";
    for (auto output : node.getOp()->output->tensorIdMap()) {
      if (!firstOutput)
        ss << ",";
      ss << "'" << output.second << "'";
      firstOutput = false;
    }
    ss << "} <- {";
    bool firstInput = true;
    for (auto input : node.getOp()->input->tensorIdMap()) {
      if (!firstInput)
        ss << ",";
      ss << "'" << input.second << "'";
      firstInput = false;
    }
    ss << "}";

    break;
  }
  }
  return ss.str();
}

/**
 * Helper function that wraps a string in quotes, pads and aligns left, for
 * ease of reading. Also add a whitespace.
 */
void fieldOut(std::ostream &out, const std::string &str, size_t pad = 15) {
  std::string field = std::string("\"") + str + "\"";
  out << std::setw(pad) << field << " ";
}

/**
 * Stream operator to output global schedule for helpful error messages.
 */
std::ostream &operator<<(std::ostream &out,
                         const liveness::LivenessAnalyzer &liveness) {
  for (size_t l = 0; l < liveness.getOpScheduleSize(); ++l) {
    const auto &node = liveness.getOpScheduleAt(l);
    // Output index.
    out << "    #" << std::setw(4) << l << ": ";
    // Output callstack string.
    fieldOut(out, getCallstackStr(node), CALLSTACK_PADDING);
    // Output status string.
    fieldOut(out, getStatusStr(node), STATUS_PADDING);
    // Output details string.
    fieldOut(out, getDetailsStr(node), DETAILS_PADDING);

    out << std::endl;
  }
  return out;
}

/**
 * Stream operator to output expected schedule for helpful error messages.
 */
std::ostream &operator<<(std::ostream &out,
                         const ExpectedLivenessSchedule &expectedSchedule) {
  for (size_t l = 0; l < expectedSchedule.size(); ++l) {
    const auto &node = expectedSchedule[l];
    // Output index.
    out << "    #" << std::setw(4) << l << ": ";
    // Output callstack string.
    fieldOut(out, node.callstackStr, CALLSTACK_PADDING);
    // Output status string.
    fieldOut(out, node.statusStr, STATUS_PADDING);
    // Output details string.
    fieldOut(out, node.detailsStr, DETAILS_PADDING);
    out << std::endl;
  }
  return out;
}

/**
 * For a given model (defined by a function that takes a builder) and copying
 * strategy, test the produced global schedule against an expectation defined
 * via strings. Print a helpful message when it doesn't hold.
 */
void CHECK_LIVENESS_SCHEDULE(std::function<void(Builder *)> modelBuilder,
                             SubgraphCopyingStrategy copyStrat,
                             const ExpectedLivenessSchedule &expected) {

  // Call the test objects.
  auto testObjs = setup(modelBuilder, copyStrat);
  auto &actual  = *testObjs.livenessAnalyzer;

  std::stringstream ss;
  ss << std::left << std::endl;
  ss << "Expected liveness schedule:" << std::endl;
  ss << expected;
  ss << "Actual liveness schedule:" << std::endl;
  ss << actual;

  BOOST_REQUIRE_MESSAGE(actual.getOpScheduleSize() == expected.size(),
                        ss.str() << "Expected schedule size is "
                                 << expected.size() << ", actual size is "
                                 << actual.getOpScheduleSize());

  for (size_t l = 0; l < actual.getOpScheduleSize(); ++l) {
    auto &actNode = actual.getOpScheduleAt(l);
    auto &expNode = expected.at(l);

    // Check call stack string matches.
    BOOST_REQUIRE_MESSAGE(getCallstackStr(actNode) == expNode.callstackStr,
                          ss.str() << "Expected callstack at position #" << l
                                   << " is '" << expNode.callstackStr
                                   << "', actual is '"
                                   << getCallstackStr(actNode) << "'");

    // Check status string matches.
    BOOST_REQUIRE_MESSAGE(getStatusStr(actNode) == expNode.statusStr,
                          ss.str()
                              << "Expected status at position #" << l << " is '"
                              << expNode.statusStr << "', actual is '"
                              << getStatusStr(actNode) << "'");

    // Check details string matches.
    BOOST_REQUIRE_MESSAGE(getDetailsStr(actNode) == expNode.detailsStr,
                          ss.str() << "Expected details at position #" << l
                                   << " is '" << expNode.detailsStr
                                   << "', actual is '" << getDetailsStr(actNode)
                                   << "'");
  }
}

// Graph identifier.
using GraphId = std::string;
// Represent a subgraph partition as a list of tuples of subgraph part indices
// and strings describing what is lowered in that subgraph part.
using PartitionOpRep  = std::tuple<int, std::string>;
using PartitionOpReps = std::vector<PartitionOpRep>;

using PartitionOpRepsMap = std::map<GraphId, PartitionOpReps>;

/**
 * Get a simple object representation of a subgraph partition for a graph.
 */
PartitionOpReps getPartitionOpReps(const TestObjects &testObjs,
                                   const Graph &graph) {

  PartitionOpReps result;

  const auto &ir          = *testObjs.ir;
  const auto &liveness    = *testObjs.livenessAnalyzer;
  const auto &partitioner = *testObjs.subgraphPartitioner;

  const auto &sched = liveness.getGraphOpSchedule(graph.id.str());
  for (auto &op : sched) {
    std::stringstream ss;
    ss << op->opid;

    if (CallOp *callOp = dynamic_cast<CallOp *>(op)) {
      // For call-ops, get the callop schedule.
      const auto &callOpSchedule = partitioner.getCallOpSchedule(callOp);
      for (const auto &callOpScheduleEntry : callOpSchedule) {
        const auto &callOpPart   = std::get<0>(callOpScheduleEntry);
        const auto &subgraphPart = std::get<1>(callOpScheduleEntry);
        std::stringstream ss;
        ss << callOpPart;
        result.emplace_back(subgraphPart, ss.str());
      }

    } else {
      // For non-CallOp ops, check the op lowers in one part.
      BOOST_CHECK(partitioner.getOpSubgraphPartBegin(op) ==
                  partitioner.getOpSubgraphPartEnd(op) - 1);
      result.emplace_back(partitioner.getOpSubgraphPartBegin(op), ss.str());
    }
  }

  if (!sched.empty()) {
    // Check number of parts matches last ops end (to test getNumSubgraphParts).
    BOOST_CHECK(partitioner.getNumSubgraphParts(graph) ==
                partitioner.getOpSubgraphPartEnd(sched.back()));
  }

  return result;
}

/**
 * Get a simple object representation of subgraph partitions for all graphs.
 */
PartitionOpRepsMap getPartitionOpRepsMap(const TestObjects &testObjs) {
  PartitionOpRepsMap result;
  for (auto graph : testObjs.ir->getAllGraphs()) {
    result[graph->id.str()] = getPartitionOpReps(testObjs, *graph);
  }
  return result;
}

/**
 * Stream operator to output all subgraph schedules for helpful error messages.
 */
std::ostream &operator<<(std::ostream &out, const PartitionOpRepsMap &map) {

  for (const auto &entry : map) {
    out << "    graph '" << entry.first << "':" << std::endl;
    size_t l = 0;
    for (const auto &opRep : entry.second) {
      // Output index.
      out << "      #" << std::setw(4) << l++ << ": ";
      out << std::get<0>(opRep) << " -> " << std::get<1>(opRep) << std::endl;
    }
  }
  return out;
}

/**
 * For a given model (defined by a function that takes a builder) and copying
 * strategy, test the produced subgraph partition matches an expected
 * partitioning by means of a string-based represenation. Print helpful messages
 * when it doesn't hold.
 */
void CHECK_SUBGRAPH_PARTITION(std::function<void(Builder *)> modelBuilder,
                              SubgraphCopyingStrategy copyStrat,
                              const PartitionOpRepsMap &expected) {
  // Call the test objects.
  auto testObjs     = setup(modelBuilder, copyStrat);
  auto &ir          = *testObjs.ir;
  auto &partitioner = *testObjs.subgraphPartitioner;

  auto actual = getPartitionOpRepsMap(testObjs);

  std::stringstream ss;
  ss << std::left << std::endl;
  ss << "Expected subgraph partitions:" << std::endl;
  ss << expected;
  ss << "Actual subgraph partitions:" << std::endl;
  ss << actual;

  BOOST_REQUIRE_MESSAGE(expected.size() == actual.size(),
                        ss.str() << "found " << actual.size() << " graphs ("
                                 << "expected " << expected.size() << ")");

  for (const auto &entry : actual) {
    // Check we have an expectation.
    BOOST_REQUIRE_MESSAGE(expected.find(entry.first) != expected.end(),
                          ss.str() << "found graph '" << entry.first
                                   << "' without expectation");

    // Get the respective op representations.
    const auto &act = entry.second;
    const auto &exp = expected.at(entry.first);

    BOOST_REQUIRE_MESSAGE(act.size() == exp.size(),
                          ss.str() << "For subgraph '" << entry.first << "', "
                                   << "expected subgraph partition size is "
                                   << exp.size() << ", actual size is "
                                   << act.size());

    for (size_t i = 0; i < act.size(); ++i) {
      BOOST_REQUIRE_MESSAGE(std::get<0>(act.at(i)) == std::get<0>(exp.at(i)),
                            ss.str()
                                << "For subgraph '" << entry.first << "', "
                                << "position #" << i << ", expected "
                                << "subgraph part " << std::get<0>(exp.at(i))
                                << ", actual is " << std::get<0>(act.at(i)));

      BOOST_REQUIRE_MESSAGE(std::get<1>(act.at(i)) == std::get<1>(exp.at(i)),
                            ss.str()
                                << "For subgraph '" << entry.first << "', "
                                << "position #" << i << ", expected '"
                                << std::get<1>(exp.at(i)) << "', actual is '"
                                << std::get<1>(act.at(i)) << "'");
    }
  }
}

} // namespace

namespace {

auto model1 = [](Builder *builder) -> void {
  // Create a model that effectively does this:
  //
  // def sg0(a0, x0, y0):
  //   tmp0 = a0 + x0
  //   out0 = tmp0 + y0
  //   return out0
  //
  //  def main(a, x, y):
  //   out = sg0(a, x, y)
  //   return out

  // sg0.
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg0");
  auto a0   = sg0Builder->addUntypedInputTensor("a0");
  auto x0   = sg0Builder->addUntypedInputTensor("x0");
  auto y0   = sg0Builder->addUntypedInputTensor("y0");
  auto tmp0 = sg0Builder->aiOnnxOpset9().add({a0, x0}, "tmp0");
  auto out0 = sg0Builder->aiOnnxOpset9().add({tmp0, y0}, "out0");
  sg0Builder->addOutputTensor(out0);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a = builder->addInputTensor(inputInfo, "a");
  auto x = builder->addInputTensor(inputInfo, "x");
  auto y = builder->addInputTensor(inputInfo, "y");
  auto out =
      builder->aiGraphcoreOpset1().call({a, x, y}, 1, *sg0Builder, "tmp")[0];
  builder->addOutputTensor(out);
};

} // namespace

BOOST_AUTO_TEST_CASE(DelayInput_Model1_OnEnterAndExit) {

  // Check that model1, when using the OnEnterAndExit strategy,
  // basically follows the copy inputs, do stuff, then copy outputs for each
  // call. This means subgraphs need not to be broken in multiple parts
  // because no parent copies happen in the middle of a subgraph.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model1, OnEnterAndExit, {
    {"",     "Enter",         ""},
    {"",     "CopyInput",     "'sg0/a0' := 'a'"},
    {"",     "CopyInput",     "'sg0/x0' := 'x'"},
    {"",     "CopyInput",     "'sg0/y0' := 'y'"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0'} <- {'sg0/a0','sg0/x0'}"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0/1'} <- {'sg0/Add:0','sg0/y0'}"},
    {"",     "CopyOutput",    "'Call:0' := 'sg0/Add:0/1'"},
    {"",     "Exit",          ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model1, OnEnterAndExit, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" }
      }
    },
    {"sg0",
      {
        {0, "ai.onnx.Add:7" },
        {0, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

BOOST_AUTO_TEST_CASE(DelayInput_Model1_JustInTime) {

  // Check that model1, when using the 'just-in-time' subgraph copying strategy,
  // delays an input copy for tensor 'sg0/y0'. To facilitate this, sg0 needs
  // to be partitioned into two parts, with adds being in one part each.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model1, JustInTime, {
    {"",     "Enter",         ""},
    {"",     "CopyInput",     "'sg0/a0' := 'a'"},
    {"",     "CopyInput",     "'sg0/x0' := 'x'"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0'} <- {'sg0/a0','sg0/x0'}"},
    {"",     "CopyInput",     "'sg0/y0' := 'y'"}, // <- moved here
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0/1'} <- {'sg0/Add:0','sg0/y0'}"},
    {"",     "CopyOutput",    "'Call:0' := 'sg0/Add:0/1'"},
    {"",     "Exit",          ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model1, JustInTime, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CallSubgraphPart(0)" }, // call to part 0 of sg0.
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(1)" }, // call to part 1 of sg0.
        {0, "CopyOutput@0" }
      }
    },
    {"sg0",
      {
        {0, "ai.onnx.Add:7" },
        {1, "ai.onnx.Add:7" }  // <- sg0 has two parts.
      }
    }
  });
  // clang-format on
}

namespace {

auto model2 = [](Builder *builder) -> void {
  // Create a model that effectively does this:
  //
  // def sg0(a0, x0, y0):
  //   tmp0 = a0 + x0    <- tmp0 and a0 are the same variable, so this updates
  //                        a0 and hence a in the main graph.
  //   out0 = tmp0 + y0
  //   return out0
  //
  //  def main(a, x, y):
  //   out = sg0(a, x, y) <- a is 'passed by reference'
  //   return out

  // sg0.
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg0");
  auto a0                       = sg0Builder->addUntypedInputTensor("a0");
  auto x0                       = sg0Builder->addUntypedInputTensor("x0");
  auto y0                       = sg0Builder->addUntypedInputTensor("y0");
  std::vector<TensorId> tensors = {a0, x0};
  // Inplace add.
  auto tmp0 = sg0Builder->customOp(
      Onnx::CustomOperators::AddLhsInplace, 1, {a0, x0}, 1, {}, "tmp0")[0];
  auto out0 = sg0Builder->aiOnnxOpset9().add({tmp0, y0}, "out0");
  sg0Builder->addOutputTensor(out0);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a = builder->addInputTensor(inputInfo, "a");
  auto x = builder->addInputTensor(inputInfo, "x");
  auto y = builder->addInputTensor(inputInfo, "y");

  // Instead of the call below, add a call which has the attributes necessary
  // to mimick inputIndex 0 being modified in the call using customOp.
  // auto tmp = builder->aiGraphcoreOpset1().call(
  //    {a, x, y}, 1, *sg0Builder, "tmp")[0];

  auto sg0ModelProto = io::getModelFromString(sg0Builder->getModelProto());
  auto calleeProto   = sg0ModelProto.graph();
  auto out           = builder->customOp(
      Onnx::AiGraphcore::OpSet1::Call,
      1,
      {a, x, y},
      1,
      {{"callee", calleeProto}, {"modifiedInputs", std::vector<int64_t>({0})}},
      "tmp")[0];

  builder->addOutputTensor(out);
};

} // namespace

BOOST_AUTO_TEST_CASE(CopyModified_Model2_OnEnterAndExit) {

  // Check that model2 (which is just like model1 except that the call
  // thinks that the first input is modified in the call, and hence introduces
  // a CopyModified) with the simple subgraph copying strategy inserts the
  // CopyModified at the end, not requiring any partitioning of subgraphs.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model2, OnEnterAndExit, {
    {"",     "Enter",                        ""},
    {"",     "CopyInput",                    "'sg0/a0' := 'a'"},
    {"",     "CopyInput",                    "'sg0/x0' := 'x'"},
    {"",     "CopyInput",                    "'sg0/y0' := 'y'"},
    {"|sg0", "ai.graphcore.AddLhsInplace:1", "{'sg0/AddLhsInplace:0'} <- {'sg0/a0','sg0/x0'}"},
    {"|sg0", "ai.onnx.Add:7",                "{'sg0/Add:0'} <- {'sg0/AddLhsInplace:0','sg0/y0'}"},
    {"",     "CopyOutput",                   "'Call:0' := 'sg0/Add:0'"},
    {"",     "CopyModified",                 "'a' := 'sg0/a0'"},
    {"",     "Exit",                         ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model2, OnEnterAndExit, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" },
        {0, "CopyModified@0" },
      }
    },
    {"sg0",
      {
        {0, "ai.graphcore.AddLhsInplace:1" },
        {0, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

BOOST_AUTO_TEST_CASE(CopyModified_Model2_JustInTime) {

  // Check that model2 (which is just like model1 except that the call
  // thinks that the first input is modified in the call, and hence introduces
  // a CopyModified) with the just-in-time subgraph copying strategy inserts the
  // CopyModified once the value of 'sg0/a0' is final.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model2, JustInTime, {
    {"",     "Enter",                        ""},
    {"",     "CopyInput",                    "'sg0/a0' := 'a'"},
    {"",     "CopyInput",                    "'sg0/x0' := 'x'"},
    {"|sg0", "ai.graphcore.AddLhsInplace:1", "{'sg0/AddLhsInplace:0'} <- {'sg0/a0','sg0/x0'}"},
    {"",     "CopyModified",                 "'a' := 'sg0/a0'"}, // <- moved here
    {"",     "CopyInput",                    "'sg0/y0' := 'y'"}, // <- moved here
    {"|sg0", "ai.onnx.Add:7",                "{'sg0/Add:0'} <- {'sg0/AddLhsInplace:0','sg0/y0'}"},
    {"",     "CopyOutput",                   "'Call:0' := 'sg0/Add:0'"},
    {"",     "Exit",                         ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model2, JustInTime, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CallSubgraphPart(0)" }, // call to part 0 of sg0.
        {0, "CopyModified@0" },      // <- early modified return
        {0, "CopyInput@2" },         // <- late input copy
        {0, "CallSubgraphPart(1)" }, // call to part 1 of sg0.
        {0, "CopyOutput@0" }
      }
    },
    {"sg0",
      {
        {0, "ai.graphcore.AddLhsInplace:1" },
        // <-- main graph does copies here.
        {1, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

namespace {

auto model3 = [](Builder *builder) -> void {
  // Create a model that effectively does this:
  //
  // def sg0(a0):
  //   out0_0 = a0 + a0
  //   out0_1 = out0_0 + a0
  //   return out0_0, out0_1
  //
  //  def main(a):
  //   out_0, out_1 = sg0(a)
  //   return out_0, out_1

  // sg0.
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg0");
  auto a0     = sg0Builder->addUntypedInputTensor("a0");
  auto out0_0 = sg0Builder->aiOnnxOpset9().add({a0, a0}, "out0_0");
  auto out0_1 = sg0Builder->aiOnnxOpset9().add({out0_0, a0}, "out0_1");
  sg0Builder->addOutputTensor(out0_0);
  sg0Builder->addOutputTensor(out0_1);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a     = builder->addInputTensor(inputInfo, "a");
  auto call  = builder->aiGraphcoreOpset1().call({a}, 2, *sg0Builder, "call");
  auto out_0 = call[0];
  auto out_1 = call[1];
  builder->addOutputTensor(out_0);
  builder->addOutputTensor(out_1);
};

} // namespace

BOOST_AUTO_TEST_CASE(EarlyOutput_Model3_OnEnterAndExit) {

  // Check that model3, when using the 'simple' subgraph copying strategy,
  // basically follows the copy inputs, do stuff, then copy outputs for each
  // call. This means subgraphs need not to be broken in multiple parts
  // because no parent copies happen in the middle of a subgraph.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model3, OnEnterAndExit, {
    {"",     "Enter",         ""},
    {"",     "CopyInput",     "'sg0/a0' := 'a'"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0'} <- {'sg0/a0','sg0/a0'}"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0/1'} <- {'sg0/Add:0','sg0/a0'}"},
    {"",     "CopyOutput",    "'Call:0' := 'sg0/Add:0'"},
    {"",     "CopyOutput",    "'Call:1' := 'sg0/Add:0/1'"},
    {"",     "Exit",          ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model3, OnEnterAndExit, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" },
        {0, "CopyOutput@1" }
      }
    },
    {"sg0",
      {
        {0, "ai.onnx.Add:7" },
        {0, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

BOOST_AUTO_TEST_CASE(EarlyOutput_Model3_JustInTime) {

  // Check that model3, when using the 'just-in-time' subgraph copying strategy,
  // one output is copied early. To facilitate this, sg0 needs is partitioned
  // into two parts, with adds being in one part each.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model3, JustInTime, {
    {"",     "Enter",         ""},
    {"",     "CopyInput",     "'sg0/a0' := 'a'"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0'} <- {'sg0/a0','sg0/a0'}"},
    {"",     "CopyOutput",    "'Call:0' := 'sg0/Add:0'"},
    {"|sg0", "ai.onnx.Add:7", "{'sg0/Add:0/1'} <- {'sg0/Add:0','sg0/a0'}"},
    {"",     "CopyOutput",    "'Call:1' := 'sg0/Add:0/1'"},
    {"",     "Exit",          ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model3, JustInTime, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CallSubgraphPart(0)" }, // <- Call to first add.
        {0, "CopyOutput@0" },        // <- Early output.
        {0, "CallSubgraphPart(1)" }, // <- Call to second add.
        {0, "CopyOutput@1" }
      }
    },
    {"sg0",
      {
        {0, "ai.onnx.Add:7" },
        // <-- main graph does copies here.
        {1, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

namespace {

auto model4 = [](Builder *builder) -> void {
  // Create a model with a nested call:
  //
  // def sg1(a1, b1, c1):
  //   out1_0 = a1 + b1
  //   out1_1 = out1_0 + c1
  //   return out1_0, out1_1
  //
  // def sg0(a0, b0, c0):
  //   tmp0_0 = a0 + a0
  //   out0_0, tmp0_1 = sg1(tmp0_0, b0, c0)
  //   tmp0_2, out0_1 = sg1(a0, tmp0_1, c0)
  //   return out0_0, out0_1
  //
  //  def main(a):
  //   out_0, out_1 = sg0(a, a, a)
  //   return out_0, out_1

  // sg1.
  auto sg1Builder = &(builder->createSubgraphBuilder());
  sg1Builder->setGraphName("sg1");
  auto a1     = sg1Builder->addUntypedInputTensor("a1");
  auto b1     = sg1Builder->addUntypedInputTensor("b1");
  auto c1     = sg1Builder->addUntypedInputTensor("c1");
  auto out1_0 = sg1Builder->aiOnnxOpset9().add({a1, b1}, "out1_0");
  auto out1_1 = sg1Builder->aiOnnxOpset9().add({out1_0, c1}, "c1");
  sg1Builder->addOutputTensor(out1_0);
  sg1Builder->addOutputTensor(out1_1);

  // sg0.
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg0");
  auto a0      = sg0Builder->addUntypedInputTensor("a0");
  auto b0      = sg0Builder->addUntypedInputTensor("b0");
  auto c0      = sg0Builder->addUntypedInputTensor("c0");
  auto tmp0_0  = sg0Builder->aiOnnxOpset9().add({a0, a0}, "tmp0_0");
  auto call0_0 = sg0Builder->aiGraphcoreOpset1().call(
      {tmp0_0, b0, b0}, 2, *sg1Builder, "call0_0");
  auto out0_0  = call0_0[0];
  auto tmp0_1  = call0_0[1];
  auto call0_1 = sg0Builder->aiGraphcoreOpset1().call(
      {a0, tmp0_1, c0}, 2, *sg1Builder, "call0_1");
  // auto tmp0_2 = call0_1[0]; <- unused.
  auto out0_1 = call0_1[1];
  sg0Builder->addOutputTensor(out0_0);
  sg0Builder->addOutputTensor(out0_1);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a = builder->addInputTensor(inputInfo, "a");
  auto call =
      builder->aiGraphcoreOpset1().call({a, a, a}, 2, *sg0Builder, "call");
  auto out_0 = call[0];
  auto out_1 = call[1];
  builder->addOutputTensor(out_0);
  builder->addOutputTensor(out_1);
};

} // namespace

BOOST_AUTO_TEST_CASE(Nested_Model4_OnEnterAndExit) {

  // Check that model4, a model with nested CallOps, when using the 'on enter
  // and exit' subgraph copying strategy, no subgraphs are partitioned still.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model4, OnEnterAndExit, {
    {"",         "Enter",                   ""},
    {"",         "CopyInput",               "'sg0/a0' := 'a'"},
    {"",         "CopyInput",               "'sg0/b0' := 'a'"},
    {"",         "CopyInput",               "'sg0/c0' := 'a'"},
    {"|sg0",     "ai.onnx.Add:7",           "{'sg0/Add:0'} <- {'sg0/a0','sg0/a0'}"},
    {"|sg0",     "Enter",                   ""},
    {"|sg0",     "CopyInput",               "'sg1/a1' := 'sg0/Add:0'"},
    {"|sg0",     "CopyInput",               "'sg1/b1' := 'sg0/b0'"},
    {"|sg0",     "CopyInput",               "'sg1/c1' := 'sg0/b0'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a1','sg1/b1'}"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/c1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:0' := 'sg1/Add:0'"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:1' := 'sg1/Add:0/1'"},
    {"|sg0",     "Exit",                    ""},
    {"|sg0",     "Enter",                   ""},
    {"|sg0",     "CopyInput",               "'sg1/a1' := 'sg0/a0'"},
    {"|sg0",     "CopyInput",               "'sg1/b1' := 'sg0/Call:1'"},
    {"|sg0",     "CopyInput",               "'sg1/c1' := 'sg0/c0'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a1','sg1/b1'}"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/c1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:0/1' := 'sg1/Add:0'"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:1/1' := 'sg1/Add:0/1'"},
    {"|sg0",     "Exit",                    ""},
    {"",         "CopyOutput",              "'Call:0/2' := 'sg0/Call:0'"},
    {"",         "CopyOutput",              "'Call:1/2' := 'sg0/Call:1/1'"},
    {"",         "Exit",                    ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model4, OnEnterAndExit, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" },
        {0, "CopyOutput@1" }
      }
    },
    {"sg0",
      {
        {0, "ai.onnx.Add:7" },
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" },
        {0, "CopyOutput@1" },
        {0, "CopyInput@0" },
        {0, "CopyInput@1" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyOutput@0" },
        {0, "CopyOutput@1" }
      }
    },
    {"sg1",
      {
        {0, "ai.onnx.Add:7" },
        {0, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

BOOST_AUTO_TEST_CASE(Nested_Model4_JustInTime) {

  // Check that model4, when using the 'just-in-time' subgraph copying strategy,
  // one output is copied early. To facilitate this, sg0 as well as sg1 need
  // to be partitioned.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model4, JustInTime, {
    {"",         "Enter",                   ""},
    {"",         "CopyInput",               "'sg0/a0' := 'a'"},
    {"|sg0",     "ai.onnx.Add:7",           "{'sg0/Add:0'} <- {'sg0/a0','sg0/a0'}"},
    {"|sg0",     "Enter",                   ""},
    {"",         "CopyInput",               "'sg0/b0' := 'a'"},
    {"|sg0",     "CopyInput",               "'sg1/a1' := 'sg0/Add:0'"},
    {"|sg0",     "CopyInput",               "'sg1/b1' := 'sg0/b0'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a1','sg1/b1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:0' := 'sg1/Add:0'"},
    {"",         "CopyOutput",              "'Call:0/2' := 'sg0/Call:0'"},
    {"|sg0",     "CopyInput",               "'sg1/c1' := 'sg0/b0'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/c1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:1' := 'sg1/Add:0/1'"},
    {"|sg0",     "Exit",                    ""},
    {"|sg0",     "Enter",                   ""},
    {"|sg0",     "CopyInput",               "'sg1/a1' := 'sg0/a0'"},
    {"|sg0",     "CopyInput",               "'sg1/b1' := 'sg0/Call:1'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a1','sg1/b1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:0/1' := 'sg1/Add:0'"},
    {"",         "CopyInput",               "'sg0/c0' := 'a'"},
    {"|sg0",     "CopyInput",               "'sg1/c1' := 'sg0/c0'"},
    {"|sg0|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/c1'}"},
    {"|sg0",     "CopyOutput",              "'sg0/Call:1/1' := 'sg1/Add:0/1'"},
    {"",         "CopyOutput",              "'Call:1/2' := 'sg0/Call:1/1'"},
    {"|sg0",     "Exit",                    ""},
    {"",         "Exit",                    ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model4, JustInTime, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CallSubgraphPart(0)" },
        {0, "CopyInput@1" },
        {0, "CallSubgraphPart(1)" },
        {0, "CopyOutput@0" },
        {0, "CallSubgraphPart(2)" },
        {0, "CopyInput@2" },
        {0, "CallSubgraphPart(3)" },
        {0, "CopyOutput@1" }
      }
    },
    {"sg0", // <-- Split over 4 parts (see calls to parts in main graph).
      {
        {0, "ai.onnx.Add:7" },
        // <-- main graph does copies here.
        {1, "CopyInput@0" },
        {1, "CopyInput@1" },
        {1, "CallSubgraphPart(0)" },
        {1, "CopyOutput@0" },
        // <-- main graph does copies here.
        {2, "CopyInput@2" },
        {2, "CallSubgraphPart(1)" },
        {2, "CopyOutput@1" },
        {2, "CopyInput@0" },
        {2, "CopyInput@1" },
        {2, "CallSubgraphPart(0)" },
        {2, "CopyOutput@0" },
        // <-- main graph does copies here.
        {3, "CopyInput@2" },
        {3, "CallSubgraphPart(1)" },
        {3, "CopyOutput@1" }
      }
    },
    {"sg1", // <-- Split over 2 parts (see calls to parts in "sg0").
      {
        {0, "ai.onnx.Add:7" },
        // <-- sg does copies here.
        {1, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}

namespace {

auto model5 = [](Builder *builder) -> void {
  // Create a model with a pass-through subgraph:
  //
  // def sg0(a0):
  //   return a0
  //
  //  def main(a):
  //   out = sg0()
  //   return out

  // sg0.
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg0");
  auto a0 = sg0Builder->addUntypedInputTensor("a0");
  sg0Builder->addOutputTensor(a0);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a   = builder->addInputTensor(inputInfo, "a");
  auto out = builder->aiGraphcoreOpset1().call({a}, 1, *sg0Builder, "call")[0];
  builder->addOutputTensor(out);
};

} // namespace

BOOST_AUTO_TEST_CASE(EdgeCase_Passthrough_Model5_JustInTime) {

  // Check that model5 which has an input that isn't consumed by an op and an
  // output that isn't produced by an op, we still see the inputs/outputs.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model5, JustInTime, {
    {"",         "Enter",                   ""},
    {"",         "CopyInput",               "'sg0/a0' := 'a'"},
    {"",         "CopyOutput",              "'Call:0' := 'sg0/a0'"},
    {"",         "Exit",                    ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model5, JustInTime, {
    {"",
      {
        {0, "CopyInput@0" },
        {0, "CopyOutput@0" }
      }
    },
    {"sg0",
      {
      }
    }
  });
  // clang-format on
}

namespace {

auto model6 = [](Builder *builder) -> void {
  // Stuff needed for loops.
  int const_int_data[1]   = {10};
  bool const_bool_data[1] = {true};
  TensorInfo const_int32_shape{"INT32", std::vector<int64_t>{}};
  TensorInfo const_bool_shape{"BOOL", std::vector<int64_t>{}};
  TensorInfo const_int64_shape{"INT64", std::vector<int64_t>{}};
  popart::ConstVoidData const_int_cvdata{const_int_data, const_int32_shape};
  popart::ConstVoidData const_bool_cvdata{const_bool_data, const_bool_shape};
  auto M    = builder->aiOnnxOpset11().constant(const_int_cvdata);
  auto cond = builder->aiOnnxOpset11().constant(const_bool_cvdata);

  // def sg0(a0, b0):
  //   out0 = a0 + a0
  //   out1 = out0 + b0 <-- opportunity to copy b0 late
  //   return out0, out1
  auto sg0Builder = &(builder->createSubgraphBuilder());
  sg0Builder->setGraphName("sg1");
  auto iter      = sg0Builder->addInputTensor(const_int64_shape);
  auto keepgoing = sg0Builder->addInputTensor(const_bool_shape);
  auto a0        = sg0Builder->addUntypedInputTensor("a0");
  auto b0        = sg0Builder->addUntypedInputTensor("b0");
  auto out0      = sg0Builder->aiOnnxOpset9().add({a0, a0}, "out0");
  auto out1      = sg0Builder->aiOnnxOpset9().add({out0, a0}, "out1");
  sg0Builder->addOutputTensor(keepgoing);
  sg0Builder->addOutputTensor(out0);
  sg0Builder->addOutputTensor(out1);

  // Main graph.
  Shape inputShape = {1};
  TensorInfo inputInfo{"INT32", inputShape};
  auto a = builder->addInputTensor(inputInfo, "a");
  auto loop =
      builder->aiOnnxOpset11().loop({M, cond, a, a}, 2, *sg0Builder, "call");
  builder->addOutputTensor(loop[0]);
};

} // namespace

BOOST_AUTO_TEST_CASE(EdgeCase_Loop_Model6_JustInTime) {

  // Check that model6 which has loops, doesn't allow delaying inputs passed
  // a loop's subgraph or make outputs earlier before a loops exit. Input and
  // output copies should be just like with the 'on enter and exit' strategy
  // for subgraphs like this.

  // clang-format off
  CHECK_LIVENESS_SCHEDULE(model6, JustInTime, {
    {"",     "Enter",                   ""},
    {"",     "CopyInput",               "'sg1/input/1' := 'Constant:0/1'"},
    {"",     "CopyInput",               "'sg1/Add:0' := 'a'"},
    {"",     "CopyInput",               "'sg1/Add:0/1' := 'a'"},
    {"",     "CopyLoopCarried",         ""},
    {"",     "CopyLoopCarried",         ""},
    {"",     "CopyLoopCarried",         ""},
    {"|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a0','sg1/a0'}"},
    {"|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/a0'}"},
    {"",     "CopyLoopCarried",         ""},
    {"",     "CopyLoopCarried",         ""},
    {"",     "CopyLoopCarried",         ""},
    {"|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0'} <- {'sg1/a0','sg1/a0'}"},
    {"|sg1", "ai.onnx.Add:7",           "{'sg1/Add:0/1'} <- {'sg1/Add:0','sg1/a0'}"},
    {"",     "CopyOutput",              "'Loop:0' := 'sg1/Add:0'"},
    {"",     "CopyOutput",              "'Loop:1' := 'sg1/Add:0/1'"},
    {"",     "Exit",                    ""}
  });
  // clang-format on

  // clang-format off
  CHECK_SUBGRAPH_PARTITION(model6, JustInTime, {
    {"",
      {
        {0, "ai.onnx.Loop:11" }, // <- No call schedule for loops.
      }
    },
    {"sg1",
      {
        {0, "ai.onnx.Add:7" },
        {0, "ai.onnx.Add:7" }
      }
    }
  });
  // clang-format on
}
