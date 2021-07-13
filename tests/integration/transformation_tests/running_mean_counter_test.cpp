// Copyright(c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RunningMeanCounter

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <vector>
#include <popart/adam.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(Default) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2}};

  float dummy[4];
  ConstVoidData t1Data = {dummy, shape};
  auto weight0         = builder->addInitializedInputTensor(t1Data);
  auto weight1         = builder->addInitializedInputTensor(t1Data);
  auto input           = builder->addInputTensor(shape);

  auto x = aiOnnx.matmul({input, weight0});

  auto o = aiOnnx.matmul({x, weight1});

  auto loss = builder->aiGraphcoreOpset1().l1loss({o}, 0.1);

  // Create the IR
  auto dataFlow = DataFlow(1, {{o, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.enableOutlining                         = false;
  userOptions.enableGradientAccumulation              = true;
  userOptions.accumulationFactor                      = 2;
  userOptions.accumulationAndReplicationReductionType = ReductionType::Mean;
  userOptions.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;

  auto device = createTestDevice(TEST_TARGET, 1);
  auto optim  = Adam({},
                    AdamMode::Adam,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT);

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              dataFlow,
              loss,
              &optim,
              *device,
              userOptions,
              Patterns()});

  BOOST_CHECK(ir.getMainGraphTensors().contains(reservedCounterPrefix()));
  unsigned counterAccumulateOps = 0;
  for (auto op : ir.getAllOps()) {
    if (op->isConvertibleTo<AccumulateOp>()) {
      if (op->inId(AccumulateOp::getVarToUpdateInIndex())
              .find(reservedCounterPrefix()) != std::string::npos) {
        counterAccumulateOps++;
      }
    }
  }
  BOOST_CHECK(counterAccumulateOps == 1);
}

BOOST_AUTO_TEST_CASE(VirtualGraph) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2}};

  float dummy[4];
  ConstVoidData t1Data = {dummy, shape};
  auto weight0         = builder->addInitializedInputTensor(t1Data);
  auto weight1         = builder->addInitializedInputTensor(t1Data);
  auto input           = builder->addInputTensor(shape);

  auto x = aiOnnx.matmul({input, weight0});
  builder->virtualGraph({x}, 0);

  auto o = aiOnnx.matmul({x, weight1});
  builder->virtualGraph({o}, 1);

  auto loss = builder->aiGraphcoreOpset1().l1loss({o}, 0.1);
  builder->virtualGraph({loss}, 1);

  // Create the IR
  auto dataFlow = DataFlow(1, {{o, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode           = VirtualGraphMode::Manual;
  userOptions.enableOutlining            = false;
  userOptions.enableGradientAccumulation = true;
  userOptions.accumulationFactor         = 2;
  userOptions.accumulationAndReplicationReductionType = ReductionType::Mean;
  userOptions.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;

  auto device = createTestDevice(TEST_TARGET, 2);
  auto optim  = Adam({},
                    AdamMode::Adam,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT);

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              dataFlow,
              loss,
              &optim,
              *device,
              userOptions,
              Patterns()});

  BOOST_CHECK(ir.getMainGraphTensors().contains(
      std::string(reservedCounterPrefix()) + "_VGraph0"));
  BOOST_CHECK(ir.getMainGraphTensors().contains(
      std::string(reservedCounterPrefix()) + "_VGraph1"));
  unsigned counterAccumulateOps = 0;
  for (auto op : ir.getAllOps()) {
    if (op->isConvertibleTo<AccumulateOp>()) {
      if (op->inId(AccumulateOp::getVarToUpdateInIndex())
              .find(reservedCounterPrefix()) != std::string::npos) {
        counterAccumulateOps++;
      }
    }
  }
  BOOST_CHECK(counterAccumulateOps == 2);
}

BOOST_AUTO_TEST_CASE(Pipelining) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2, 2}};

  float dummy[4];
  ConstVoidData t1Data = {dummy, shape};
  auto weight0         = builder->addInitializedInputTensor(t1Data);
  auto weight1         = builder->addInitializedInputTensor(t1Data);
  auto weight2         = builder->addInitializedInputTensor(t1Data);
  auto input           = builder->addInputTensor(shape);

  auto x = aiOnnx.matmul({input, weight0});
  builder->virtualGraph({x}, 0);
  builder->pipelineStage({x}, 0);

  x = aiOnnx.matmul({x, weight1});
  builder->virtualGraph({x}, 1);
  builder->pipelineStage({x}, 1);

  auto o = aiOnnx.matmul({x, weight2});
  builder->virtualGraph({o}, 0);
  builder->pipelineStage({o}, 2);

  auto loss = builder->aiGraphcoreOpset1().l1loss({o}, 0.1);
  builder->virtualGraph({loss}, 0);
  builder->pipelineStage({loss}, 2);

  // Create the IR
  auto dataFlow = DataFlow(1, {{o, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.enablePipelining           = true;
  userOptions.virtualGraphMode           = VirtualGraphMode::Manual;
  userOptions.enableOutlining            = false;
  userOptions.enableGradientAccumulation = true;
  userOptions.accumulationFactor         = 5;
  userOptions.accumulationAndReplicationReductionType = ReductionType::Mean;
  userOptions.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;

  auto device = createTestDevice(TEST_TARGET, 2);
  auto optim  = Adam({},
                    AdamMode::Adam,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT);

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              dataFlow,
              loss,
              &optim,
              *device,
              userOptions,
              Patterns()});

  BOOST_CHECK(ir.getMainGraphTensors().contains(
      std::string(reservedCounterPrefix()) + "_VGraph0_PStage2"));
  BOOST_CHECK(ir.getMainGraphTensors().contains(
      std::string(reservedCounterPrefix()) + "_VGraph1_PStage3"));
  BOOST_CHECK(ir.getMainGraphTensors().contains(
      std::string(reservedCounterPrefix()) + "_VGraph0_PStage4"));
  unsigned counterAccumulateOps = 0;
  for (auto op : ir.getAllOps()) {
    if (op->isConvertibleTo<AccumulateOp>()) {
      if (op->inId(AccumulateOp::getVarToUpdateInIndex())
              .find(reservedCounterPrefix()) != std::string::npos) {
        counterAccumulateOps++;
      }
    }
  }
  BOOST_CHECK(counterAccumulateOps == 3);
}