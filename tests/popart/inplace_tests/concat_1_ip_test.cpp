// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Concat1InplaceTest

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

BOOST_AUTO_TEST_CASE(Inplace_replConcat) {

  //            |----|
  //  in0 ... --|----|-- concat -- relu -- ...
  //            |----|
  //
  //  We expect all concat and relu to be inplaced at the popart level
  //  see T7100

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 3}};
  auto in0     = builder->addInputTensor(shape0);
  auto r0      = aiOnnx.sigmoid({in0});
  auto concat0 = aiOnnx.concat({r0, r0, r0}, 0);
  auto preOut  = aiOnnx.relu({concat0});
  auto out     = aiOnnx.sigmoid({preOut});
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
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
}
