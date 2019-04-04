#define BOOST_TEST_MODULE Scale0InplaceTest

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
#include <poponnx/topocons.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(Inplace_Scale0) {

  //                 |-- slice [0,3 ], [0,3] - scale 1.1 -|
  //                 |-- slice [3,6 ], [3,6] - scale 2.2 -| - matmul -|
  //                 |
  // in0 (10 x 10) --|
  //

  auto test = [](bool branch77High) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();
    TensorInfo shape0{"FLOAT", std::vector<int64_t>{6, 6}};
    auto in0 = builder->addInputTensor(shape0);

    auto sl11 = aiOnnx.slice({in0}, {6, 3}, {3, 0}, {0, 1});
    auto sl22 = aiOnnx.slice({in0}, {6, 6}, {3, 3}, {0, 1});

    builder->setInplacePreferences(sl11, {{"SliceInplace", 1000.0}});
    builder->setInplacePreferences(sl22, {{"SliceInplace", 1000.0}});

    float factor11 = 1.1 * (2 * branch77High - 1);
    float factor22 = 2.2 * (2 * branch77High - 1);

    auto sc11 = aiOnnx.scale({sl11}, factor11);
    auto sc22 = aiOnnx.scale({sl22}, factor22);

    builder->setInplacePreferences(sc11, {{"ScaleInplace", 10.0 + factor11}});
    builder->setInplacePreferences(sc22, {{"ScaleInplace", 10.0 + factor22}});

    auto out = aiOnnx.matmul({sc22, sc11});

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

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

    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul).size() == 1);
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SliceInplace).size() == 2);
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ScaleInplace).size() == 2);
  };

  test(true);
  test(false);
}
