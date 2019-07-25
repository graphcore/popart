#define BOOST_TEST_MODULE Parallel0InplaceTest

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

using namespace popart;

// where we test that with Relus is parallel, exactly 1 of
// them is converted into an InplaceRelu by the Inplace pattern.
BOOST_AUTO_TEST_CASE(Inplace_parallel0) {

  // clang-format off
  //
  // Consider the Relu Ops in PARALLEL
  //
  //           | -- [Relu] -- (h0) -- |
  //           |                      | --- [Add] -- (h3) -|
  // (in0) >---| -- [Relu] -- (h1) -- |                    |
  //           |                                           | -> [Add] -- (out)
  //           | -- [Relu] -- (h2) ----------------------- |
  //
  // We can make the first relu in-place, but then we stall as
  // an in-place modifying op must run after all other consumers of its
  // input (and therefore there can only be one in-place consumer here)
  // So, we expect:
  //
  //         | - [ReluInplace] - (h0) -|
  //         |                         | --- [Add] -- (h3) -|
  // (in0) > | ------ [Relu] -- (h1) --|                    |
  //         |                                              | -> [Add] -- (out)
  //         | ------ [Relu] -- (h2) ---------------------- |
  //
  // We guarantee that it is indeed the Relu to h0 which is inplace 
  // by setting it to have a very high priority
  //
  // clang-format on

  // Build an onnx model (for training)
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0 = builder->addInputTensor(shape);
  auto h0  = aiOnnx.relu({in0});
  builder->setInplacePreferences(h0, {{"ReluInplace", 1e8}});
  auto h1 = aiOnnx.relu({in0});
  auto h2 = aiOnnx.relu({in0});
  // TODO T6707 : ensure that this (and other) tests are still valid
  // when inplace for Add is implemented, or default priorities change
  auto h3  = aiOnnx.add({h0, h1});
  auto out = aiOnnx.add({h2, h3});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // Check the ir
  // Just the one Relu has been inplaced
  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 3 - 1);
  auto opsOfTypeReluInplace = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  BOOST_CHECK(opsOfTypeReluInplace.size() == 1);
  // and that the output of the inplace relu is h0
  BOOST_CHECK(opsOfTypeReluInplace.back()->output->id(0) == h0);
}

BOOST_AUTO_TEST_CASE(Inplace_parallel1) {

  //     |- Exp --|
  // in -|        |- Matmul
  //     |- Relu -|
  //
  // where Exp is higher priority for inplacing

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{4, 4, 4}};
  auto in0  = builder->addInputTensor(shape);
  auto left = aiOnnx.exp({in0});
  builder->setInplacePreferences(left, {{"ExpInplace", 1e8}});
  auto right = aiOnnx.relu({in0});
  builder->setInplacePreferences(right, {{"ReluInplace", 1e-8}});
  auto out = aiOnnx.matmul({left, right});

  builder->addOutputTensor(out);
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {
                               {out, AnchorReturnType("ALL")},
                           });

  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *cpuDevice,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // Check the ir
  // Only the Exp has been inplaced
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 0);

  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Exp).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ExpInplace).size() == 1);
}

// next test:
//
// r0 = relu(i0)
// r1 = relu(i1)
// r2 = relu(i2)
// r3 = relu(i3)
// catadd = cat(i0, i1, i2) + cat(i1, i2, i3)
// radcat = cat(r0, r1, r2) + cat(r1, r2, r3)
// out = reducesum(catadd) + radcat
