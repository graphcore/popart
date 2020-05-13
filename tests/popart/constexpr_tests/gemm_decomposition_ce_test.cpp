// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprAddTest

#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(ConstExprTest_Gemm_Decomposition0) {
  // {(A) (B) (C) (alpha) (beta)} -> [Gemm] -> (*)
  //
  // after GemmDecompositionPattern becomes
  //
  // {(alpha) (A)} -> [Scale] -> (sA)
  // {(beta) (C)} -> [Scale] -> (sC)
  // {(sA) (B)} -> [MatMul] -> (sAB)
  // {(sAB) (sC) -> [Add] -> (*)
  //
  // const folding should compute (s_C) leaving
  //
  // {(alpha) (A)} -> [Scale] -> (sA)
  // {(sA) (B)} -> [MatMul] -> (sAB)
  // {(sAB) (sC) -> [Add] -> (*)

  Shape in_out_shape = {2, 2};
  TensorInfo tinfo{"FLOAT", in_out_shape};

  std::vector<float> data{1., 2., 3., 4.};

  ConstVoidData a_data = {data.data(), tinfo};
  ConstVoidData c_data = {data.data(), tinfo};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  // The two fixed-point tensors which are Constants
  auto a_id    = aiOnnx.constant(a_data, "aData");
  auto c_id    = aiOnnx.constant(c_data, "cData");
  auto b_id    = builder->addInputTensor(tinfo);
  auto out_id  = aiOnnx.gemm({a_id, b_id, c_id}, 1., 1., 0, 0);
  auto loss_id = builder->aiGraphcoreOpset1().l1loss({out_id}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding out_id as an anchor
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<std::shared_ptr<Loss>> losses{
      std::make_shared<IdentityLoss>(loss_id, "l1LossVal", ReductionType::Sum)};
  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns({PreAliasPatternType::PostNRepl,
                        PreAliasPatternType::GemmDecomposition})});

  // Check the ir
  // 1) there should only be 1 scale op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale).size() == 1);
  // 2) there should only be 1 mul op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul).size() == 1);
  // 3) there should only be 1 add op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 4) there should only be 3 ops total
  BOOST_CHECK(ir.getMainGraphOps().size() == 3);
}
