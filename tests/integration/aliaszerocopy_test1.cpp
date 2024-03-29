// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AliasZeroCopyTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <popart/aliaszerocopy.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/add.hpp>
#include <popart/op/loop.hpp>
#include <popart/subgraphcopyingstrategy.hpp>
#include <popart/util.hpp>

#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"

namespace popart {
class Tensor;
} // namespace popart

using namespace popart;
using namespace liveness;

// Test if the liveness analyzer and alias zero copy treat LoopOps correctly
BOOST_AUTO_TEST_CASE(AliasZeroCopyLoopTest0) {

  // Construct ir, graph, subgraph
  Ir ir;

  auto &graph      = ir.getMainGraph();
  auto subgraph_id = ir.createUniqueSubgraphId({"loop"});
  auto &subgraph   = ir.createGraph(subgraph_id);

  // Add mandatory loop iterator tensor to subgraph (is not an output)
  TensorId loopItScopedId = addScope(subgraph, reservedLoopIteratorPrefix());
  subgraph.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

  // Add mandatory loop condition tensor to subgraph (is also an output)
  TensorId loopCondScopedId = addScope(subgraph, reservedLoopCondPrefix());
  subgraph.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
  subgraph.markAsOutput(loopCondScopedId);

  TensorId taId = "A";
  TensorId tbId = "B";

  TensorInfo infoA(DataType::FLOAT, Shape{});
  TensorInfo infoB(DataType::FLOAT, Shape{});

  graph.addInput(taId, infoA);
  graph.addInput(tbId, infoB);

  Op::Settings gsettings(graph, "main");
  Op::Settings sgsettings(subgraph, "loop");

  auto loopOpUp =
      std::make_unique<LoopOp>(Onnx::Operators::Loop_11, gsettings, subgraph);
  LoopOp *loopOp = loopOpUp.get();
  graph.moveIntoGraph(std::move(loopOpUp));
  loopOp->setTripCountValue(4);

  TensorId staId = addScope(subgraph, taId);
  TensorId stbId = addScope(subgraph, tbId);

  TensorId tcId = "C";
  TensorId tdId = "D";

  TensorId stdId = addScope(subgraph, tdId);

  std::unique_ptr<AddOp> addOpUp =
      std::make_unique<AddOp>(Onnx::Operators::Add_7, sgsettings);
  Op *addOp = addOpUp.get();
  subgraph.moveIntoGraph(std::move(addOpUp));

  loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + 0, taId, staId, false);
  loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + 1, tbId, stbId, false);

  addOp->connectInTensor(AddOp::getArg0InIndex(), stbId);
  addOp->connectInTensor(AddOp::getArg1InIndex(), stbId);
  addOp->createAndConnectOutTensor(AddOp::getOutIndex(), stdId);
  addOp->setup();

  loopOp->addLoopOutput(
      LoopOp::getFirstOutputOutIndex() + 0, tcId, staId, false);
  loopOp->addLoopOutput(
      LoopOp::getFirstOutputOutIndex() + 1, tdId, stdId, false);
  loopOp->setup();

  graph.markAsOutput(tdId);
  OnEnterAndExitSubgraphCopyingStrategy strat;
  LivenessAnalyzer analyzer(&ir, &strat);
  strat.setLivenessAnalyzer(&analyzer);
  analyzer.apply();
  AliasZeroCopy zeroCopy(&ir, &analyzer);
  zeroCopy.apply();

  // Check analyzer

  // Ground truth
  std::vector<LivenessNode> reference;
  reference.emplace_back(OpStatus::Enter, 0, 0, 0); // LoopOp
  // see loop.hpp, 0/1 skipped
  reference.emplace_back(OpStatus::CopyInput, 2, 0, 0);       // taId -> staId
  reference.emplace_back(OpStatus::CopyInput, 3, 0, 0);       // tbId -> stbId
  reference.emplace_back(OpStatus::CopyLoopCarried, 0, 0, 0); // loop cond
  reference.emplace_back(OpStatus::CopyLoopCarried, 1, 0, 0); // staId -> staId
  reference.emplace_back(OpStatus::CopyLoopCarried, 2, 0, 0); // stdId -> stbId
  reference.emplace_back(OpStatus::Normal, 0, 0, 0); // AddOp (iteration 0)
  reference.emplace_back(OpStatus::CopyLoopCarried, 0, 0, 0); // loop cond
  reference.emplace_back(OpStatus::CopyLoopCarried, 1, 0, 0); // staId -> staId
  reference.emplace_back(OpStatus::CopyLoopCarried, 2, 0, 0); // stdId -> stbId
  reference.emplace_back(OpStatus::Normal, 0, 0, 0);     // AddOp (iteration 1)
  reference.emplace_back(OpStatus::CopyOutput, 0, 0, 0); // staId -> tcId
  reference.emplace_back(OpStatus::CopyOutput, 1, 0, 0); // stdId -> tdId
  reference.emplace_back(OpStatus::Exit, 0, 0, 0);       // LoopOp

  BOOST_CHECK_EQUAL(analyzer.getOpScheduleSize(), reference.size());
  for (int64_t i = 0; i < analyzer.getOpScheduleSize(); ++i) {
    auto &node    = analyzer.getOpScheduleAt(i);
    auto &refNode = reference.at(i);
    logging::trace("Node: {}", node);
    BOOST_CHECK_EQUAL(node.getStatus(), refNode.getStatus());
    BOOST_CHECK_EQUAL(node.getIndex(), refNode.getIndex());
  }

  // Check alias zero copy
  // Expected output:
  // 11111121112111
  // ______________ (A)
  // **___*___*____ (B)
  // ______________ (C)
  // ************** (D)
  // ______________ (loop_subgraph(0)/A)
  // **___*___*____ (loop_subgraph(0)/B)
  // __***_***_**__ (loop_subgraph(0)/D)
  //
  // B -> loop_subgraph(0)/B alias
  // C, loop_subgraph(0)/A, A are never read
  // loop_subgraph(0)/D is loop_subgraph(0)/B + loop_subgraph(0)/B
  // D is an anchored tensor and conservatively kept always live

  zeroCopy.printLivenessIntervals(
      {
          ir.getTensor(taId),
          ir.getTensor(tbId),
          ir.getTensor(tcId),
          ir.getTensor(tdId),
          ir.getTensor(staId),
          ir.getTensor(stbId),
          ir.getTensor(stdId),
      },
      ProducerInterval::Enforce);

  Intervals refFull;
  refFull.insert(0, 14);

  Intervals refEmpty;

  Intervals refStbId;
  refStbId.insert(0, 2);
  refStbId.insert(5, 6);
  refStbId.insert(9, 10);

  Intervals refStdId;
  refStdId.insert(2, 5);
  refStdId.insert(6, 9);
  refStdId.insert(10, 12);

  BOOST_CHECK_EQUAL(zeroCopy.getCandidateLivenessIntervals(ir.getTensor(taId)),
                    refEmpty);
  BOOST_CHECK(
      zeroCopy.getCandidateLivenessIntervals(ir.getTensor(tcId)).empty());
  BOOST_CHECK_EQUAL(zeroCopy.getCandidateLivenessIntervals(ir.getTensor(staId)),
                    refEmpty);
  BOOST_CHECK_EQUAL(zeroCopy.getCandidateLivenessIntervals(ir.getTensor(stbId)),
                    refStbId);
  BOOST_CHECK_EQUAL(zeroCopy.getCandidateLivenessIntervals(ir.getTensor(stdId)),
                    refStdId);

  // Check that the right loop parts got disabled
  BOOST_CHECK(!zeroCopy.copyInputRequired(loopOp, 2));
  BOOST_CHECK(zeroCopy.copyInputRequired(loopOp, 3));
  BOOST_CHECK(!zeroCopy.copyLoopCarriedRequired(loopOp, 1));
  BOOST_CHECK(zeroCopy.copyLoopCarriedRequired(loopOp, 2));
  BOOST_CHECK(!zeroCopy.copyOutputRequired(loopOp, 0));
  BOOST_CHECK(zeroCopy.copyOutputRequired(loopOp, 1));
}
