#define BOOST_TEST_MODULE ConstExprUnsqueezeTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(ConstExprTest_Unsqueeze0) {
  // clang-format off
  //
  // {(c0) -> [Unsqueeze] -> (h0)
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // where c0 is a constant, should become
  //
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // clang-format on

  std::vector<float> raw_const_data(2 * 3 * 4, 0.0);
  ConstVoidData const_data = {raw_const_data.data(), {"FLOAT", Shape{2, 3, 4}}};

  TensorInfo in_info{"FLOAT", Shape{2, 3, 1, 4, 1}};

  auto builder          = Builder::create();
  auto aiOnnx           = builder->aiOnnxOpset9();
  auto const_node       = aiOnnx.constant(const_data, "const_data");
  auto unsqueeze_output = aiOnnx.unsqueeze({const_node}, {2, 4});
  auto in_id            = builder->addInputTensor(in_info);
  auto out_id           = aiOnnx.add({unsqueeze_output, in_id});
  builder->addOutputTensor(out_id);

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto data_flow = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out_id, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              losses,
              &optimizer,
              *cpuDevice,
              {}, // no SessionOptions
              Patterns({})});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the Concat Op is not present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Unsqueeze).size() == 0);
  // 3) that the shape of the output tensor is as specified.
  Shape ref_shape{2, 3, 1, 4, 1};
  BOOST_CHECK(ir.getTensors().get(out_id)->info.shape() == ref_shape);
}
