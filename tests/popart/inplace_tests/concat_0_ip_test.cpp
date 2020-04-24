// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Concat0InplaceTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>
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
  std::vector<std::shared_ptr<Loss>> losses{
      std::make_shared<L1Loss>(out, "l1LossVal", 0.1, ReductionType::SUM)};
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
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 3);

  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() == 1);
}
