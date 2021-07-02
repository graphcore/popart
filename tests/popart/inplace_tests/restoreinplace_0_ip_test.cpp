// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RestoreInplace0InplaceTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/optimizer.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(test0) {
  //  input                                                           \
  //    |                                                              |
  //   Relu                                                            |
  //    |                                                              |
  //   t0  -------------------                                         |
  //    | \                    \                                       |
  //    |  Stash - stash_t0 - RestoreInplace - t0_alias - ...          |
  //    |                                                              | Ipu0
  // Reshape0                                                          |
  //    |                                                              |
  //   t1                                                              |
  //    |                                                              |
  // Reshape1                                                          |
  //    |                                                              |
  //   t2                                                              |
  //    |                                                             /
  // Identity                                                         \
  //    |                                                              | Ipu1
  //  output                                                          /
  //
  // In this pipelined model, if all operations on the path from t0 to the
  // IpuCopy from Ipu0 to Ipu1 are inplaced, then we have a problem. t2 (which,
  // in this case, would be an alias of t0) would contain 'old' restored data
  // when it is copied. Test that Reshape0 is not inplaced, but Reshape1 is

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  auto input = builder->addInputTensor("FLOAT", {8, 2});
  auto t0    = aiOnnx.relu({input});

  Shape t_shapeSize = {2};
  Shape t1_shape    = {4, 4};
  auto t1_shape_t = aiOnnx.constant({t1_shape.data(), {"INT64", t_shapeSize}});
  auto t1         = aiOnnx.reshape({t0, t1_shape_t});

  Shape t2_shape  = {16, 1};
  auto t2_shape_t = aiOnnx.constant({t2_shape.data(), {"INT64", t_shapeSize}});
  auto t2         = aiOnnx.reshape({t1, t2_shape_t});

  auto output = aiGraphcore.l1loss({t2}, 0.1);

  auto opts             = SessionOptions();
  opts.virtualGraphMode = VirtualGraphMode::Manual;
  opts.enablePipelining = true;
  builder->virtualGraph(t0, 0);
  builder->virtualGraph(t1, 0);
  builder->virtualGraph(t2, 0);
  builder->virtualGraph(output, 1);

  // Create the IR
  std::vector<TensorId> anchorIds = {reservedGradientPrefix() + input};
  auto device                     = createTestDevice(TEST_TARGET, 2);
  auto optimizer                  = ConstSGD(0.01);
  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              DataFlow(3, anchorIds),
              output,
              &optimizer,
              *device,
              opts,
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto sched = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  BOOST_CHECK_EQUAL(sched[1]->opid, Onnx::CustomOperators::Stash);
  BOOST_CHECK_EQUAL(sched[2]->opid, Onnx::CustomOperators::ReshapeInplace);
  BOOST_CHECK_EQUAL(sched[3]->opid, Onnx::AiOnnx::OpSet9::Reshape);
  BOOST_CHECK_EQUAL(ir.opsOfType(Onnx::CustomOperators::Restore).size(), 0);
  BOOST_CHECK_EQUAL(ir.opsOfType(Onnx::CustomOperators::RestoreInplace).size(),
                    1);
}
