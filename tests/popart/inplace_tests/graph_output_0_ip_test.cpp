// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphOutput0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

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

  builder->addInputTensorFromHigherScope(superIn0);

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
  auto dataFlow = DataFlow(1, {{acts[1], AnchorReturnType("ALL")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  auto sched      = ir.getOpSchedule({});
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({});
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

  builder->addInputTensorFromHigherScope(superIn0);

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
  auto dataFlow = DataFlow(1, {{acts[1], AnchorReturnType("ALL")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  auto sched      = ir.getOpSchedule({});
  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({});
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
