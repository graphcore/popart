// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LoopAliasTest0

#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/scale.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

/**
 * Test loop correctness with aliasing tensors (regression test: see T54054).
 *
 * The loop computes (pseudocode):
 *
 * A = {0, ..., 8}
 * B = 0
 * for (int i = 0; i < 5; ++i) {
 *   A *= 2                         // A aliases (inplace update)
 *   B = ConcatInplace(A, A, 1)     // B aliases A, and contains internal
 *                                  // aliasing
 * }
 *
 * where A and B are loop carried with both internal and external aliasing.
 *
 * Potential points of failure can be:
 * - Copying the initial inputs to the body outputs due to internal aliasing on
 *   the body outputs.
 * - Copying the body outputs to the body inputs due to external aliasing
 *   between body inputs and body outputs (because A and B alias each other, and
 *   A and B also alias between body input and output).
 * - Copying from the loop body to the LoopOp output (because the output tensors
 *   can be aliased with AliasZeroCopy).
 */
BOOST_AUTO_TEST_CASE(LoopAliasTest0) {
  auto ir = std::make_unique<Ir>();

  SessionOptions options;
  options.aliasZeroCopy = true;
  ir->setUserOptions(options);

  Graph &g  = ir->getMainGraph();
  Graph &sg = ir->createGraph(GraphId{"loopBody"});

  const TensorInfo info_A{DataType::FLOAT, Shape{2, 4}};
  const TensorInfo info_B{DataType::FLOAT, Shape{2, 8}};

  std::vector<float> data_A{0, 1, 2, 3, 4, 5, 6, 7, 8};

  g.getTensors().addVarInit("A", info_A, data_A.data());

  Op::Settings gSettings(g, "op", {});
  Op::Settings sgSettings(sg, "op", {});

  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), "B"}},
                              Onnx::CustomOperators::Init_1,
                              info_B,
                              TensorType::ActGrad,
                              InitType::Zero,
                              gSettings);

  // Add mandatory loop iterator tensor to subgraph (is not an output)
  TensorId loopItScopedId = addScope(sg, reservedLoopIteratorPrefix());
  sg.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

  // Add mandatory loop condition tensor to subgraph (is also an output)
  TensorId loopCondScopedId = addScope(sg, reservedLoopCondPrefix());
  sg.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
  sg.markAsOutput(loopCondScopedId);

  sg.addInput(addScope(sg, "A"), info_A);
  sg.addInput(addScope(sg, "B"), info_B);

  sg.createConnectedOp<ScaleInplaceOp>(
      {{0, addScope(sg, "A")}},
      {{ConcatInplaceOp::getOutIndex(), addScope(sg, "As")}},
      Onnx::CustomOperators::ScaleInplace,
      2.0,
      sgSettings.copy("ScaleInplaceOp"));

  // Causes internal aliasing for the loop body output tensor
  sg.createConnectedOp<ConcatInplaceOp>(
      {{0, addScope(sg, "As")}, {1, addScope(sg, "A")}},
      {{ConcatInplaceOp::getOutIndex(), addScope(sg, "C")}},
      1,
      sgSettings.copy("ConcatInplaceOp"));

  // As -> A
  sg.markAsOutput(addScope(sg, "As"));

  // C -> B
  sg.markAsOutput(addScope(sg, "C"));

  LoopOp *loopOp =
      g.createConnectedOp<LoopOp>({{LoopOp::getFirstInputInIndex() + 0, "A"},
                                   {LoopOp::getFirstInputInIndex() + 1, "B"}},
                                  {{LoopOp::getFirstOutputOutIndex() + 0, "D"},
                                   {LoopOp::getFirstOutputOutIndex() + 1, "C"}},
                                  Onnx::Operators::Loop_11,
                                  gSettings,
                                  sg);
  loopOp->setTripCountValue(5);
  loopOp->setup();

  ir->setDataFlow(DataFlow{
      1, {{"C", AnchorReturnType("All")}, {"D", AnchorReturnType("All")}}});
  ir->updateVertices();

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));
  session->prepareDevice();

  std::vector<float> outC(info_B.nelms());
  std::vector<float> outD(info_A.nelms());

  NDArrayWrapper<float> outWrapperC(outC.data(), info_B);
  NDArrayWrapper<float> outWrapperD(outD.data(), info_A);

  std::map<TensorId, IArray &> anchors = {{"C", outWrapperC},
                                          {"D", outWrapperD}};

  StepIO stepio({}, anchors);
  session->weightsFromHost();
  session->run(stepio);

  logging::trace("C: {}", outC);
  logging::trace("D: {}", outD);

  std::vector<float> refC{
      0, 32, 64, 96, 0, 32, 64, 96, 128, 160, 192, 224, 128, 160, 192, 224};
  std::vector<float> refD{0, 32, 64, 96, 128, 160, 192, 224};

  BOOST_REQUIRE_EQUAL(outC, refC);
  BOOST_REQUIRE_EQUAL(outD, refD);
}
