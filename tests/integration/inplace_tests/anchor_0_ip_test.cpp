// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Anchor0InplaceTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(InplaceAnchortest1) {

  // in0 -> scale(1.5) -> scale(2.0) -> scale(2.5)
  //              \                         \
  //              anchor                   anchor
  //
  // test that the first scale is in-place, the second is not (consumes anchor)
  // and the third one is
  //

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};

  auto in0 = builder->addInputTensor(shape0);
  auto sc0 = aiGraphcore.scale({in0}, 1.5);
  auto sc1 = aiGraphcore.scale({sc0}, 2.0);
  auto sc2 = aiGraphcore.scale({sc1}, 2.5);
  builder->addOutputTensor(sc2);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(
      1, {{sc0, AnchorReturnType("All")}, {sc2, AnchorReturnType("All")}});
  auto device = createTestDevice(TEST_TARGET);

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
  BOOST_CHECK(sched.size() == 3);
  BOOST_CHECK(sched[0]->opid == Onnx::CustomOperators::ScaleInplace);
  BOOST_CHECK(sched[1]->opid == Onnx::AiGraphcore::OpSet1::Scale);
  BOOST_CHECK(sched[2]->opid == Onnx::CustomOperators::ScaleInplace);
}

BOOST_AUTO_TEST_CASE(InplaceAnchorTest2) {
  //                     anchor
  //                      /
  // [in0] -> reduce -> [A] transpose -> [B] -> scale -> [C] -> reduce -> out
  //
  // inplace priorities:       100                10
  // (these priorities ensure that transpose is considered for inplacing before
  // scale)
  //
  // We expect transpose to be inplaced and then scale to not be inplaced, as it
  // would modify the anchor A.
  //

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{2, 2, 2, 2}};

  auto in0 = builder->addInputTensor(shape0);
  auto A   = aiOnnx.reducesum({in0}, std::vector<int64_t>{{0, 1}});
  auto B   = aiOnnx.transpose({A});
  builder->setInplacePreferences(B, {{"TransposeInplace", 100}});

  auto C = aiGraphcore.scale({B}, 2.0);
  builder->setInplacePreferences(C, {{"ScaleInplace", 10}});

  auto out = aiOnnx.reducesum({C}, std::vector<int64_t>{{0, 1}});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(
      1, {{A, AnchorReturnType("All")}, {out, AnchorReturnType("All")}});
  auto device = createTestDevice(TEST_TARGET);

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
  for (auto op : sched) {
    std::cout << op->opid << std::endl;
  }
  BOOST_CHECK(sched[1]->opid == Onnx::CustomOperators::TransposeInplace);
  BOOST_CHECK(sched[2]->opid == Onnx::AiGraphcore::OpSet1::Scale);
}

BOOST_AUTO_TEST_CASE(InplaceAnchorTest3) {

  //                     anchor
  //                      /                                    anchor
  //                     /                                       /
  // [in0] -> scale -> [s0] -> transpose -> concat -> scale -> [s1]
  //                        -> identity  ->
  //
  // priorities:
  //          300                400         100       200
  //                             500
  //
  // We expect all but the concat to be inplace
  //

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{2, 2, 2, 2}};

  auto in0 = builder->addInputTensor(shape0);
  auto s0  = aiGraphcore.scale({in0}, 2.0);
  builder->setInplacePreferences(s0, {{"ScaleInplace", 300}});
  auto t0 = aiOnnx.transpose({s0});
  builder->setInplacePreferences(t0, {{"TransposeInplace", 400}});
  auto i0 = aiOnnx.identity({s0});
  builder->setInplacePreferences(i0, {{"IdentityInplace", 500}});
  auto c0 = aiOnnx.concat({t0, i0}, 0);
  builder->setInplacePreferences(c0, {{"ConcatInplace", 100}});
  auto s1 = aiGraphcore.scale({c0}, 3.0);
  builder->setInplacePreferences(s1, {{"ScaleInplace", 200}});

  builder->addOutputTensor(s1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(
      1, {{s0, AnchorReturnType("All")}, {s1, AnchorReturnType("All")}});
  auto device = createTestDevice(TEST_TARGET);

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
  BOOST_CHECK(sched.size() == 5);

  if (logging::shouldLog(logging::Module::popart, logging::Level::Debug)) {
    for (auto op : sched) {
      std::cout << op->opid.type << std::endl;
    }
  }

  BOOST_CHECK(sched[0]->opid == Onnx::CustomOperators::ScaleInplace);
  for (auto id : {sched[1]->opid, sched[2]->opid}) {
    BOOST_CHECK(id == Onnx::CustomOperators::IdentityInplace ||
                id == Onnx::CustomOperators::TransposeInplace);
  }
  BOOST_CHECK(sched[3]->opid.type == "Concat");
  BOOST_CHECK(sched[3]->opid != Onnx::CustomOperators::ConcatInplace);
  BOOST_CHECK(sched[4]->opid == Onnx::CustomOperators::ScaleInplace);
}

//
