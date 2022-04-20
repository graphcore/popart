// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Subsample0InplaceTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
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
#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_subsample0) {

  //       input of shape {2,3,4}
  //        |               \
  //       relu            scale-0.2
  //        |                   \
  //      subsample {2,1,2}    subsample {2,1,2}
  //            \                 /
  //             \               /
  //              \             /
  //               \           /
  //                \         /
  //                 \       /
  //                  \     /
  //                   \   /
  //                    add -> output of shape {1,3,2}

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo info0{"FLOAT", std::vector<int64_t>{2, 3, 4}};
  auto in0 = builder->addInputTensor(info0);

  auto rel0       = aiOnnx.relu({in0});
  auto subsample0 = builder->aiGraphcoreOpset1().subsample({rel0}, {2, 1, 2});

  float scaleFactor0 = 0.2;
  auto s0            = builder->aiGraphcoreOpset1().scale({in0}, scaleFactor0);
  auto subsample1    = builder->aiGraphcoreOpset1().subsample({s0}, {2, 1, 2});

  auto addOut = aiOnnx.add({subsample0, subsample1});
  auto dotOut = aiOnnx.sigmoid({addOut});
  builder->addOutputTensor(dotOut);

  builder->setInplacePreferences(s0, {{"ScaleInplace", 300.}});
  builder->setInplacePreferences(rel0, {{"ReluInplace", 900.}});
  builder->setInplacePreferences(subsample0, {{"SubsampleInplace", 800.0}});
  builder->setInplacePreferences(subsample1, {{"SubsampleInplace", 10.0}});
  builder->setInplacePreferences(addOut, {{"AddLhsInplace", 1000.0}});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{dotOut, AnchorReturnType("All")}});
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

  auto opsOfTypeSubsample = ir.opsOfType(Onnx::AiGraphcore::OpSet1::Subsample);
  BOOST_CHECK(opsOfTypeSubsample.size() == 0);

  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 0);

  auto opsOfTypeScale = ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale);
  BOOST_CHECK(opsOfTypeScale.size() == 1);

  auto opsOfTypeAdd = ir.opsOfType(Onnx::AiOnnx::OpSet9::Add);
  BOOST_CHECK(opsOfTypeAdd.size() == 0);
}
