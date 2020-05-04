// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Sequeeze0InplaceTest

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

BOOST_AUTO_TEST_CASE(Inplace_sequeeze0) {

  // in0 (1, 3, 6)    |- [Sequeeze [0] -|
  //                                    |-- [MatMul] ---
  // in1 (1, 6, 4)    |- [Sequeeze [0] -|
  //
  // We expect both Squeeze ops to be inplaced when enableInPlace is true

  auto test = [](bool enable_inplace) {
    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 3, 6}};
    TensorInfo shape1{"FLOAT", std::vector<int64_t>{1, 6, 4}};

    auto in0 = builder->addInputTensor(shape0);
    auto in1 = builder->addInputTensor(shape1);

    // the API required (name) (ends) (starts)
    auto sl0 = aiOnnx.squeeze({in0}, {0});
    auto sl1 = aiOnnx.squeeze({in1}, {0});
    auto out = aiOnnx.matmul({sl0, sl1});
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

    auto device = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare(
        {modelProto,
         InputShapeInfo(),
         dataFlow,
         {},
         nullptr,
         *device,
         {},
         Patterns(PatternsLevel::NoPatterns).enableInPlace(enable_inplace)});

    // Check the ir
    // All the Relus have been optimised out if enable_inplace
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Squeeze).size() ==
                (enable_inplace ? 0 : 2));
    // and have been replaced with ReluInplace if enable_inplace
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SqueezeInplace).size() ==
                (enable_inplace ? 2 : 0));
  };

  // test with inplacing enabled,
  test(true);
  // test with inplacing disabled.
  test(false);
}
