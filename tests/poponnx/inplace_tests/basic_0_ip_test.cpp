#define BOOST_TEST_MODULE Parallel0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

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
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(false)});

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
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out, "l1LossVal", 0.1)};
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
  // first check that 2 relus have been inplaced (the final relu creates
  // an anchor so can't be)
  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 1);
  auto opsOfTypeReluInplace = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  BOOST_CHECK(opsOfTypeReluInplace.size() == 2);

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
