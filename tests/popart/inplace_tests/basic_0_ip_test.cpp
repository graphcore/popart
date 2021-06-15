// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Parallel0InplaceTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

// confirm that when the Pattern is not enabled, nothing is made inplace
BOOST_AUTO_TEST_CASE(Inplace_basic0) {

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0 = builder->addInputTensor(shape);
  auto h0  = aiOnnx.relu({in0});
  builder->setInplacePreferences(h0, {{"ReluInplace", 1e8}});
  auto h1  = aiOnnx.relu({h0});
  auto out = aiOnnx.relu({h1});
  auto l1  = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(false)});

  // Check the ir
  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 3);
  auto opsOfTypeReluInplace = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  BOOST_CHECK(opsOfTypeReluInplace.size() == 0);
}

// confirm that properties are transferred to the inplace Op
BOOST_AUTO_TEST_CASE(Inplace_basic1) {

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0 = builder->addInputTensor(shape);

  std::string relu0name = "allymacky";
  auto h0               = aiOnnx.relu({in0}, relu0name);
  int64_t relu0vGraph   = 16;
  builder->setInplacePreferences(h0, {{"ReluInplace", 1e8}});
  builder->virtualGraph(h0, relu0vGraph);

  std::string relu1name = "zipperzoo";
  auto h1               = aiOnnx.relu({h0}, relu1name);
  int64_t relu1vGraph   = 127;
  builder->setInplacePreferences(h1, {{"ReluInplace", 1e7}});
  builder->virtualGraph(h1, relu1vGraph);

  auto out = aiOnnx.relu({h1});
  builder->virtualGraph(out, 4);
  auto l1 = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);
  builder->virtualGraph(l1, 4);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              opts,
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  // Check the ir
  // first check that all 3 relus have been inplaced
  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 0);
  auto opsOfTypeReluInplace = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  BOOST_CHECK(opsOfTypeReluInplace.size() == 3);

  auto inplaceOps = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  // we have already confirmed that there are 2 inplace ops:
  if (inplaceOps.size() == 2) {
    if (inplaceOps[0]->output->id(0) == h1) {
      std::swap(inplaceOps[0], inplaceOps[1]);
    }

    // confirm the output names are h0 and h1
    BOOST_CHECK(inplaceOps[0]->output->id(0) == h0);
    BOOST_CHECK(inplaceOps[1]->output->id(0) == h1);

    // confirm the virtualGraph numbers are relu0vGraph and relu1vGraph
    BOOST_CHECK(inplaceOps[0]->settings.vgraphId == relu0vGraph);
    BOOST_CHECK(inplaceOps[1]->settings.vgraphId == relu1vGraph);

    // confirm the names contain relu0name and relu1name in them
    BOOST_CHECK(inplaceOps[0]->settings.name.find(relu0name) !=
                std::string::npos);
    BOOST_CHECK(inplaceOps[1]->settings.name.find(relu1name) !=
                std::string::npos);

    // as for the recompute flag, i suspect that
    // this doesn't make sense for inplace ops
  }
}
