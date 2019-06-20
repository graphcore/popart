#define BOOST_TEST_MODULE SplitToSlicePatternTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>

BOOST_AUTO_TEST_CASE(SplitToSliceTest0) {

  using namespace poponnx;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{6}};

  auto input1 = builder->addInputTensor(shape1);
  auto ident0 = aiOnnx.identity({input1});

  auto outs = aiOnnx.split({ident0}, 3, 0, {1, 2, 3});

  for (auto out : outs) {
    builder->addOutputTensor(out);
  }

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{outs.at(0), AnchorReturnType("ALL")},
                            {outs.at(1), AnchorReturnType("ALL")},
                            {outs.at(2), AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.enableVirtualGraphs = true;

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "2"}};
  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              userOptions,
              Patterns({PreAliasPatternType::SPLITOP})});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Slice_1).size() == 3);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Split_2).size() == 0);
}
