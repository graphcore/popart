#define BOOST_TEST_MODULE ConstExprConcatTest

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

BOOST_AUTO_TEST_CASE(ConstExprTest_Concat0) {
  // clang-format off
  //
  // {(c0, c1) -> [Concat] -> (h0)
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // where c0 and c1 are constants, should become
  //
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // clang-format on

  std::vector<float> raw_const_data_0(2 * 2 * 3, 0.0);
  std::vector<float> raw_const_data_1(2 * 2 * 3, 0.0);
  ConstVoidData const_data_0 = {raw_const_data_0.data(),
                                {"FLOAT", Shape{2, 2, 3}}};
  ConstVoidData const_data_1 = {raw_const_data_1.data(),
                                {"FLOAT", Shape{2, 2, 3}}};

  TensorInfo in_info{"FLOAT", Shape{1, 1, 1}};

  auto builder      = Builder::create();
  auto aiOnnx     = builder->aiOnnxOpset9();
  auto const_node_0 = aiOnnx.constant(const_data_0, "const_data_0");
  auto const_node_1 = aiOnnx.constant(const_data_1, "const_data_1");
  auto concat_node  = aiOnnx.concat({const_node_0, const_node_1}, 1);
  auto in_id        = builder->addInputTensor(in_info);
  auto out_id       = aiOnnx.add({concat_node, in_id});
  builder->addOutputTensor(out_id);

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto data_flow = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out_id, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              losses,
              &optimizer,
              {}, // no SessionOptions
              Patterns({})});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the Concat Op is not present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 0);
  // 3) that the shape of the output tensor is as specified.
  Shape ref_shape{2, 4, 3};
  BOOST_CHECK(ir.getTensors().get(out_id)->info.shape() == ref_shape);
}
