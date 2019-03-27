#define BOOST_TEST_MODULE Concat0InplaceTest

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

BOOST_AUTO_TEST_CASE(Inplace_concat0) {

  //  in0 -|
  //       |- [Add] - inAdd - [Relu] -- radd -|
  //       |                                  |- [Concat] - cou -|
  //       |- [Sub] - inSub - [Relu] -- rsub -|                  |
  //  in1 -|                                                     |- [Add] - out
  //   |                                                         |
  //   |--------------------- [Relu] -- rin1 --------------------|
  //
  //
  //   We expect all 3 [Relu] ops and the 1 [Concat] op to be inplaced

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{3, 1, 2}};
  TensorInfo shape1{"FLOAT", std::vector<int64_t>{3, 1, 1}};
  auto in0   = builder->addInputTensor(shape0);
  auto in1   = builder->addInputTensor(shape1);
  auto inAdd = aiOnnx.add({in0, in1});
  auto inSub = aiOnnx.sub({in0, in1});
  auto radd  = aiOnnx.relu({inAdd});
  auto rsub  = aiOnnx.relu({inSub});
  auto cou   = aiOnnx.concat({radd, rsub}, 1);
  auto rin1  = aiOnnx.relu({in1});
  auto out   = aiOnnx.add({rin1, cou});
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
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 3);

  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() == 1);
}
