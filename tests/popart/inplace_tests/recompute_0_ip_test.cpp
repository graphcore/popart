// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Recompute0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(InplaceRecomputeTest) {

  //
  //
  //  in -- scale  --  scale  --  scale  --  scale
  //          R         R           Ch         R
  // Expect:
  //         !in        in          in        !in
  //
  // first run
  //   (1)   1.5        3.0         7.5       22.5
  //
  // second run
  //   (1)   1.5        3.0        (7.5)      22.5
  //

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto in0 = builder->addInputTensor(shape0);

  // not inplacable as consumes an input Tensor
  auto sc0 = aiGraphcore.scale({in0}, 1.5);
  builder->recomputeOutputInBackwardPass(sc0, RecomputeType::Recompute);

  // consumes a recompute, so can be inplace
  auto sc1 = aiGraphcore.scale({sc0}, 2.0);
  builder->recomputeOutputInBackwardPass(sc1, RecomputeType::Recompute);

  // anchor consuming a recompute, can be inplace
  auto sc2 = aiGraphcore.scale({sc1}, 2.5);
  builder->recomputeOutputInBackwardPass(sc2, RecomputeType::Checkpoint);

  // recompute consuming an anchor, can be inplace
  auto sc3 = aiGraphcore.scale({sc2}, 3.0);
  builder->recomputeOutputInBackwardPass(sc3, RecomputeType::Recompute);

  builder->addOutputTensor(sc2);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{sc3, AnchorReturnType("All")}});
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

  auto sched = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  BOOST_CHECK(sched.size() == 4);

  BOOST_CHECK(sched[0]->opid == Onnx::AiGraphcore::OpSet1::Scale);
  BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::ScaleInplace);
  BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
  BOOST_CHECK(sched[3]->opid == Onnx::AiGraphcore::OpSet1::Scale);
}
