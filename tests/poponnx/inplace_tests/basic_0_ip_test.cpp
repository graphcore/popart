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

BOOST_AUTO_TEST_CASE(Inplace_basic0) {

  // clang-format off
  //
  // confirm that when the Pattern is not enabled, nothing is made inplace
  //
  // clang-format on

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

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {},
              Patterns(PatternsLevel::NONE).enableInPlace(false)});

  // Check the ir
  auto opsOfTypeRelu = ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu);
  BOOST_CHECK(opsOfTypeRelu.size() == 3);
  auto opsOfTypeReluInplace = ir.opsOfType(Onnx::CustomOperators::ReluInplace);
  BOOST_CHECK(opsOfTypeReluInplace.size() == 0);
}
