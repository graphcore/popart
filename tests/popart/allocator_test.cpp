// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AllocatorTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>

using namespace popart;

// model:
//
//  i1 -- Slice -- |
//                 |-- Conv --> o
//  i2 -- Slice -- |
//
BOOST_AUTO_TEST_CASE(allocator_conv_control) {
  /* In this test we check that a convolution does allocate an input
   * tensor when 'unwinding' through 'canunwind' ops with special cases
   * (in this case Slice, where only part of the region can be allocated by
   * conv)
   *
   * We observe that the input poplar tensors are not created linearly
   */

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  Shape dShape = {2, 2, 4, 4};
  TensorInfo data_input_shape{"FLOAT", dShape};
  Shape fShape = {3, 2, 3, 3};
  TensorInfo filt_input_shape{"FLOAT", fShape};

  auto i1 = builder->addInputTensor(data_input_shape);
  auto s1 = aiOnnx.slice({i1}, {1}, {0}, {0}); // shape = [1, 2, 4, 4]

  auto i2 = builder->addInputTensor(filt_input_shape);
  auto s2 = aiOnnx.slice({i2}, {2}, {0}, {0}); // shape = [2, 2, 3, 3]

  auto o = aiOnnx.conv({s1, s2}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  builder->addOutputTensor(o);

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              DataFlow(1, {{o, AnchorReturnType("All")}}),
              {},         // no losses
              {},         // no optimizer
              *cpuDevice, // no deviceInfo
              {},         // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  std::unique_ptr<popx::Devicex> device;
  device.reset(new popx::Devicex(ir, cpuDevice));
  device->prepare();

  BOOST_CHECK(device->getLinearlyCreatedInputTensors().count(i1) == 0);
  BOOST_CHECK(device->getLinearlyCreatedInputTensors().count(i2) == 0);
}

// model:
//
//  i1 --|-- Reshape --|-- Transpose --|
//                                     |-- Conv --> o
//  i2 --|-- Reshape --|-- Transpose --|
//
BOOST_AUTO_TEST_CASE(allocator_single_input_viewchanging_conv) {
  /* In this test we check that a convolution can allocate an input
   * tensor by 'unwinding' through ops (transpose and reshape) that
   * have non-identity unwind functions
   *
   * We observe that the input poplar tensors are not created linearly
   */

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  // data
  Shape dShape = {1, 32};
  TensorInfo data_input_shape{"FLOAT", dShape};
  auto i1 = builder->addInputTensor(data_input_shape);
  auto r1 = builder->reshape_const(aiOnnx, {i1}, {4, 4, 2, 1}, "r1");
  auto t1 = aiOnnx.transpose({r1}, {3, 2, 1, 0}, "t1"); // shape = [1, 2, 4, 4]

  // weights
  Shape fShape = {36};
  TensorInfo filt_input_shape{"FLOAT", fShape};
  auto i2 = builder->addInputTensor(filt_input_shape);
  auto r2 = builder->reshape_const(aiOnnx, {i2}, {3, 3, 2, 2}, "r2");
  auto t2 = aiOnnx.transpose({r2}, {3, 2, 1, 0}, "t2"); // shape = [2, 2, 3, 3]

  // conv
  auto o = aiOnnx.conv({t1, t2}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  builder->addOutputTensor(o);

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              DataFlow(1, {{o, AnchorReturnType("All")}}),
              {},         // no losses
              {},         // no optimizer
              *cpuDevice, // noDeviceInfo
              {},         // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  std::unique_ptr<popx::Devicex> device;
  device.reset(new popx::Devicex(ir, cpuDevice));
  device->prepare();

  BOOST_CHECK(device->getEfficientlyCreatedInputTensors().count(i1) == 1);
  BOOST_CHECK(device->getEfficientlyCreatedInputTensors().count(i2) == 1);
}
