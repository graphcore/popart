#define BOOST_TEST_MODULE InplaceTest

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

// where we test that a series of Relus is converted
// into InplaceRelus by the Inplace0 pattern
BOOST_AUTO_TEST_CASE(Inplace0_series) {

  // Consider the SERIES of Relu Ops:
  //
  // (in0) -> [Relu] -> (h0)
  //       -> [Relu] -> (h1)
  //       -> [Relu] -> (preId)
  //       -> [Identity] -> (out),
  //
  // with (out) as an anchor tensor. This should become,
  //
  //
  // (in0) -> [ReluInplace] -> (h0)
  //       -> [ReluInplace] -> (h1)
  //       -> [ReluInplace] -> (preId)
  //       -> [Identity] -> (out).

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0   = builder->addInputTensor(shape);
  auto h0    = aiOnnx.relu({in0});
  auto h1    = aiOnnx.relu({h0});
  auto preId = aiOnnx.relu({h1});
  auto out   = aiOnnx.identity({preId});
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
              Patterns({PatternType::INPLACE0})});

  // Check the ir
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 3);
}

// where we test that with Relus is parallel, exactly 1 of
// them is converted into an InplaceRelu by the Inplace0 pattern.
BOOST_AUTO_TEST_CASE(Inplace0_parallel) {

  // Consider the Relu Ops in PARALLEL
  //
  //           | -- [Relu] -- (h0) -- |
  //           |                      | --- [Add] -- (h3) -|
  // (in0) >---| -- [Relu] -- (h1) -- |                    |
  //           |                                           | -> [Add] -- (out)
  //           | -- [Relu] -- (h2) ----------------------- |
  //
  // We can make the first relu in-place, but then we stall as
  // an in-place modifying op must run after all other consumers of its
  // input (and therefore there can only be one in-place consumer here)
  // So, we expect:
  //
  //           | -- [ReluInplace] --- |
  //           |                      | --- [Add] -- (h3) -|
  // (in0) >---| -- [Relu] -- (h1) -- |                    |
  //           |                                           | -> [Add] -- (out)
  //           | -- [Relu] -- (h2) ----------------------- |
  //
  //

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0 = builder->addInputTensor(shape);
  auto h0  = aiOnnx.relu({in0});
  auto h1  = aiOnnx.relu({in0});
  auto h2  = aiOnnx.relu({in0});
  auto h3  = aiOnnx.add({h0, h1});
  auto out = aiOnnx.add({h2, h3});
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
              Patterns({PatternType::INPLACE0})});

  // Check the ir
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 3 - 1);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 1);
}

BOOST_AUTO_TEST_CASE(Inplace_test0) {

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
  //
  //
  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx = builder->aiOnnxOpset9();

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

  Ir ir;
  ir.prepare(
      {modelProto,
       InputShapeInfo(),
       dataFlow,
       losses,
       &optimizer,
       {},
       Patterns({PatternType::INPLACEALL})}); // PatternType::INPLACE0,
                                              // PatternType::INPLACEALL})});

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

BOOST_AUTO_TEST_CASE(Inplace_test1) {

  // in0 -|
  //      |
  // in1 -|- [Concat] ----|
  //      |               |
  // in2 -|--|            |- [Concat] - [Relu] - o1 -|
  //         |- [Concat] -|                          |
  // in3 ----|                                       |- [Concat] - o2 - [Tanh] -
  // out
  //                                                 |
  // in4 ---------- [Relu] -- c3 --------------------|

  // We expect all 2 [Relu] ops and all 4 [Concat] op to be inplaced

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx = builder->aiOnnxOpset9();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{3}};
  auto in0 = builder->addInputTensor(shape0);
  auto in1 = builder->addInputTensor(shape0);
  auto in2 = builder->addInputTensor(shape0);
  auto in3 = builder->addInputTensor(shape0);

  auto x1 = aiOnnx.concat({in0, in1, in2}, 0);
  auto x2 = aiOnnx.concat({in2, in3}, 0);
  auto x3 = aiOnnx.concat({x1, x2}, 0);
  auto o1 = aiOnnx.relu({x3});

  auto in4 = builder->addInputTensor(shape0);
  auto c3  = aiOnnx.relu({in4});
  auto o2  = aiOnnx.concat({o1, c3}, 0);
  auto out = aiOnnx.tanh({o2});

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
              Patterns({PatternType::INPLACEALL})});

  // Check the ir
  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ReluInplace).size() == 2);

  // All the Relus have been optimised out,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 0);
  // and have been replaced with ReluInplace.
  BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() == 4);
}

// next test:
//
// r0 = relu(i0)
// r1 = relu(i1)
// r2 = relu(i2)
// r3 = relu(i3)
// catadd = cat(i0, i1, i2) + cat(i1, i2, i3)
// radcat = cat(r0, r1, r2) + cat(r1, r2, r3)
// out = reducesum(catadd) + radcat
// I MUST add reducesum, and replace tanh with it above
