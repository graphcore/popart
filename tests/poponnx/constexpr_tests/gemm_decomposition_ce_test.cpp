#define BOOST_TEST_MODULE ConstExprAddTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/half.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

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
  auto a_id   = aiOnnx.constant(a_data, "aData");
  auto c_id   = aiOnnx.constant(c_data, "cData");
  auto b_id   = builder->addInputTensor(tinfo);
  auto out_id = aiOnnx.gemm({a_id, b_id, c_id}, 1., 1., 0, 0);
  builder->addOutputTensor(out_id);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding out_id as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(out_id, "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {}, // no SessionOptions
              Patterns({PreAliasPatternType::POSTNREPL,
                        PreAliasPatternType::GEMMDECOMPOSITION})});

  // Check the ir
  // 1) there should only be 1 scale op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale).size() == 1);
  // 2) there should only be 1 mul op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul).size() == 1);
  // 3) there should only be 1 add op,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 4) there should only be 3 ops total
  BOOST_CHECK(ir.getMainGraphOps().size() == 3);
}
