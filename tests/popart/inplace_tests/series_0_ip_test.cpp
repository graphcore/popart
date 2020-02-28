#define BOOST_TEST_MODULE Series0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

// where we test that a series of Relus and Exps is converted
// into a series of InplaceRelus and InplaceExps by the Inplace pattern
BOOST_AUTO_TEST_CASE(Inplace_series0) {

  // Consider the SERIES of Relu Ops:
  //
  // (in0) -> [Relu] -> (h0)
  //       -> [Relu] -> (h1)
  //       -> [Exp]  -> (h2)
  //       -> [Exp]  -> (h3)
  //       -> [Relu] -> (preId)
  //       -> [Identity] -> (out),
  //
  // with (out) as an anchor tensor. This should become,
  //
  // (in0) -> [ReluInplace] -> (h0)
  //       -> [ReluInplace] -> (h1)
  //       -> [ExpInplace]  -> (h2)
  //       -> [ExpInplace]  -> (h3)
  //       -> [ReluInplace] -> (preId)
  //       -> [Identity] -> (out).

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0   = builder->addInputTensor(shape);
  auto h0    = aiOnnx.relu({in0});
  auto h1    = aiOnnx.relu({h0});
  auto h2    = aiOnnx.exp({h1});
  auto h3    = aiOnnx.exp({h2});
  auto preId = aiOnnx.relu({h3});
  auto out   = aiOnnx.identity({preId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the (training) IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)};
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *device,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // Check the ir
  // All the Relus and Exps have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).size() == 0);

  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 3);
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ExpInplace).size() == 2);
}

BOOST_AUTO_TEST_CASE(Inplace_series_changedPreferences) {

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorInfo shape{"FLOAT", std::vector<int64_t>{5, 5}};
  auto in0 = builder->addInputTensor(shape);
  std::vector<TensorId> chainedIds{in0};

  // The number of Relus in series:
  int N = 10;

  for (int i = 0; i < N; ++i) {
    // priorities are lowest in the middle of the series of
    // relus, so the middle relus will be attempted last
    float priority = 100.0f + std::abs(i - N / 2 + 0.5);
    chainedIds.push_back(aiOnnx.relu({chainedIds.back()}));
    builder->setInplacePreferences(chainedIds.back(),
                                   {{"ReluInplace", priority}});
  }

  auto out = chainedIds.back();
  builder->addOutputTensor(out);
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(
      1, {{out, AnchorReturnType("ALL")}, {in0, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)};
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *device,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // All the Relus have been optimised out,
  // except the two which are anchors.
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == N - 2);
}
