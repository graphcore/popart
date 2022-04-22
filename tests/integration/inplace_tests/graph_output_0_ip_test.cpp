// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphOutput0InplaceTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(InplaceGraphOutputTest1) {

  //
  //
  //  in -- scale  --  scale  --  scale  --  scale
  //          \                                \
  //     graph output                     graph output
  //
  // The second scale consumes a graph output, so should not be inplaced

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  // can inplace
  auto sc0 = aiGraphcore.scale({superIn0}, 1.5, "sc0");

  // cannot inplace, as consumes graph output
  auto sc1 = aiGraphcore.scale({sc0}, 2.0, "sc1");

  // can inplace
  auto sc2 = aiGraphcore.scale({sc1}, 2.5, "sc2");

  // can inplace
  auto sc3 = aiGraphcore.scale({sc2}, 3.0, "sc3");

  builder->addOutputTensor(sc0);
  builder->addOutputTensor(sc3);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{acts[1], AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 4) {
      BOOST_CHECK(sched[0]->opid == Onnx::CustomOperators::ScaleInplace);
      BOOST_CHECK(sched[1]->opid == Onnx::AiGraphcore::OpSet1::Scale);
      BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
      BOOST_CHECK(sched[3]->opid == Onnx::CustomOperators::ScaleInplace);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}

BOOST_AUTO_TEST_CASE(InplaceGraphOutputTest2) {

  //
  //  in -- transpose -- transpose --scale
  //           \                       \
  //          out                      out
  //
  //  priorities:
  //          300           200        100
  //
  //  Expect scale (last candidate for inplacing) to not
  //  be inplaced, as if it were the first anchor would be corrupted.
  //

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0 = aiOnnx.transpose({superIn0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 300}});
  auto t1 = aiOnnx.transpose({t0});
  builder->setInplacePreferences(t1, {{"TransposeInplace", 200}});
  auto sc0 = aiGraphcore.scale({t1}, 1.5, "sc0");
  builder->setInplacePreferences(sc0, {{"ScaleInplace", 100}});

  builder->addOutputTensor(t0);
  builder->addOutputTensor(sc0);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{acts[1], AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 3) {
      BOOST_CHECK(sched[0]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[2]->opid == Onnx::AiGraphcore::OpSet1::Scale);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}

BOOST_AUTO_TEST_CASE(InplaceSubGraphOutputTest1) {

  //
  //  in -- transpose0 -- transpose1 -- out
  //           \
  //          scale0 -- scale1 -- out
  //
  //
  //  Expect transpose0 not to
  //  be inplaced as inplacing all the four ops will corrupt the output of
  //  transpose1 when 'call'ed as a subgraph, even if the topolofical constraint
  //  is in place as the output is copied out of the subgraph after all the
  //  nodes of the subgraph is executed. so either transpose0 or scale0 must not
  //  be inplaced, depending on the order in which the nodes are considered for
  //  inplacing.
  //

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0 = aiOnnx.transpose({superIn0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 10}});
  auto t1 = aiOnnx.transpose({t0});
  builder->setInplacePreferences(t1, {{"TransposeInplace", 100}});
  auto sc0 = aiGraphcore.scale({superIn0}, 1.5, "sc0");
  builder->setInplacePreferences(sc0, {{"ScaleInplace", 50}});
  auto sc1 = aiGraphcore.scale({sc0}, 2, "sc1");
  builder->setInplacePreferences(sc1, {{"ScaleInplace", 100}});

  builder->addOutputTensor(t1);
  builder->addOutputTensor(sc1);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  superBuilder->addOutputTensor(acts[0]);
  superBuilder->addOutputTensor(acts[1]);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{{acts[0], AnchorReturnType("ALL")},
                             {acts[1], AnchorReturnType("ALL")}}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 4) {

      BOOST_CHECK(sched[0]->opid == Onnx::AiOnnx::OpSet6::Transpose);

      BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
      BOOST_CHECK(sched[3]->opid == Onnx::CustomOperators::ScaleInplace);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}

BOOST_AUTO_TEST_CASE(InplaceSubGraphOutputTest2) {

  //
  //  in -- transpose0 -- transpose1 -- out
  //           \
  //          scale0 -- scale1 -- out
  //
  //
  //  Expect scale0 to not
  //  be inplaced as inplacing all the four ops will corrupt the output of
  //  transpose1 when 'call'ed as a subgraph, even if the topolofical constraint
  //  is in place as the output is copied out of the subgraph after all the
  //  nodes of the subgraph is executed. so either transpose0 or scale0 must not
  //  be inplaced, depending on the order in which the nodes are considered for
  //  inplacing.
  //

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0 = aiOnnx.transpose({superIn0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 50}});
  auto t1 = aiOnnx.transpose({t0});
  builder->setInplacePreferences(t1, {{"TransposeInplace", 100}});
  auto sc0 = aiGraphcore.scale({superIn0}, 1.5, "sc0");
  builder->setInplacePreferences(sc0, {{"ScaleInplace", 10}});
  auto sc1 = aiGraphcore.scale({sc0}, 2, "sc1");
  builder->setInplacePreferences(sc1, {{"ScaleInplace", 100}});

  builder->addOutputTensor(t1);
  builder->addOutputTensor(sc1);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  superBuilder->addOutputTensor(acts[0]);
  superBuilder->addOutputTensor(acts[1]);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{{acts[0], AnchorReturnType("ALL")},
                             {acts[1], AnchorReturnType("ALL")}}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 4) {

      BOOST_CHECK(sched[0]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[2]->opid == Onnx::AiGraphcore::OpSet1::Scale);
      BOOST_CHECK(sched[3]->opid == Onnx::CustomOperators::ScaleInplace);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}

BOOST_AUTO_TEST_CASE(InplaceSubGraphOutputTest3) {

  //
  //  in -- transpose0 -- transpose1 -- out
  //           \
  //          scale0 -- scale1 -- out
  //
  //
  //  Expect transpose0 to not
  //  be inplaced as inplacing all the four ops will corrupt the output of
  //  transpose1 when 'call'ed as a subgraph, even if the topolofical constraint
  //  is in place as the output is copied out of the subgraph after all the
  //  nodes of the subgraph is executed. so either transpose0 or scale0 must not
  //  be inplaced, depending on the order in which the nodes are considered for
  //  inplacing.
  //

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0 = aiOnnx.transpose({superIn0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 80}});
  auto t1 = aiOnnx.transpose({t0});
  builder->setInplacePreferences(t1, {{"TransposeInplace", 10}});
  auto sc0 = aiGraphcore.scale({superIn0}, 1.5, "sc0");
  builder->setInplacePreferences(sc0, {{"ScaleInplace", 100}});
  auto sc1 = aiGraphcore.scale({sc0}, 2, "sc1");
  builder->setInplacePreferences(sc1, {{"ScaleInplace", 50}});

  builder->addOutputTensor(t1);
  builder->addOutputTensor(sc1);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  superBuilder->addOutputTensor(acts[0]);
  superBuilder->addOutputTensor(acts[1]);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{{acts[0], AnchorReturnType("ALL")},
                             {acts[1], AnchorReturnType("ALL")}}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 4) {

      BOOST_CHECK(sched[0]->opid == Onnx::AiOnnx::OpSet6::Transpose);
      BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
      BOOST_CHECK(sched[3]->opid == Onnx::CustomOperators::ScaleInplace);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}

BOOST_AUTO_TEST_CASE(InplaceSubGraphOutputTest4) {

  //
  //  in -- transpose0 -- transpose1 -- out
  //           \
  //          scale0 -- scale1 -- out
  //
  //
  //  Expect transpose0 to not
  //  be inplaced as inplacing all the four ops will corrupt the output of
  //  transpose1 when 'call'ed as a subgraph, even if the topolofical constraint
  //  is in place as the output is copied out of the subgraph after all the
  //  nodes of the subgraph is executed. so either transpose0 or scale0 must not
  //  be inplaced, depending on the order in which the nodes are considered for
  //  inplacing.
  //

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto superIn0 = superBuilder->addInputTensor(shape0);

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0 = aiOnnx.transpose({superIn0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 80}});
  auto t1 = aiOnnx.transpose({t0});
  builder->setInplacePreferences(t1, {{"TransposeInplace", 50}});
  auto sc0 = aiGraphcore.scale({superIn0}, 1.5, "sc0");
  builder->setInplacePreferences(sc0, {{"ScaleInplace", 100}});
  auto sc1 = aiGraphcore.scale({sc0}, 2, "sc1");
  builder->setInplacePreferences(sc1, {{"ScaleInplace", 10}});

  builder->addOutputTensor(t1);
  builder->addOutputTensor(sc1);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  superBuilder->addOutputTensor(acts[0]);
  superBuilder->addOutputTensor(acts[1]);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{{acts[0], AnchorReturnType("ALL")},
                             {acts[1], AnchorReturnType("ALL")}}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched      = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);
    if (gSched.size() == 1) {
      // TODO (T18153)
      // Should be Call, is ScaleInplace
      //  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Call);
    } else if (gSched.size() == 4) {

      BOOST_CHECK(sched[0]->opid == Onnx::AiOnnx::OpSet6::Transpose);
      BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
      BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
      BOOST_CHECK(sched[3]->opid == Onnx::CustomOperators::ScaleInplace);
    } else {
      throw error("Expected main schedule of size 1, call of size 4");
    }
  }
}
