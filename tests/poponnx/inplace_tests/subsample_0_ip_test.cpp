#define BOOST_TEST_MODULE Subsample0InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

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
  auto s0            = aiOnnx.scale({in0}, scaleFactor0);
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
  auto dataFlow  = DataFlow(1, {{dotOut, AnchorReturnType("ALL")}});
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

  auto opsOfTypeSubsample = ir.opsOfType(Onnx::AiGraphcore::OpSet1::Subsample);
  BOOST_CHECK(opsOfTypeSubsample.size() == 0);

  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 0);

  auto opsOfTypeScale = ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale);
  BOOST_CHECK(opsOfTypeScale.size() == 1);

  auto opsOfTypeAdd = ir.opsOfType(Onnx::AiOnnx::OpSet9::Add);
  BOOST_CHECK(opsOfTypeAdd.size() == 0);
}
