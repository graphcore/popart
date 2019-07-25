#define BOOST_TEST_MODULE ReplicateAndShardIrTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>

// In this test: with 8 IPUs, replication over 4 IPUs, we test that the
// auto-sharder uses 2 IPUs per replica

BOOST_AUTO_TEST_CASE(SplitToSliceTest0) {

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{6}};

  auto input1 = builder->addInputTensor(shape1);

  auto act = aiOnnx.relu({input1});
  for (int i = 0; i < 6; ++i) {
    act = aiGraphcore.scale({act}, 0.5);
    act = aiOnnx.sigmoid({act});
    act = aiOnnx.relu({act});
  }

  builder->addOutputTensor(act);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto dataFlow = DataFlow(1, {{act, AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.enableVirtualGraphs    = true;
  userOptions.enableReplicatedGraphs = true;
  userOptions.autoVirtualGraph       = true;
  userOptions.replicatedGraphCount   = 4;
  std::map<std::string, std::string> deviceOpts{{"numIPUs", "8"}};

  auto optimizer = ConstSGD(0.01);

  auto loss = std::unique_ptr<Loss>(
      new L1Loss(act, "l1LossVal", 0.1, ReductionType::SUM));

  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss.get()},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::DEFAULT)});

  std::set<int> vGraphs;
  for (auto &id_op : ir.getMainGraphOps()) {
    if (id_op.second->hasVirtualGraphId()) {
      vGraphs.emplace(id_op.second->getVirtualGraphId());
    }
  }

  std::set<int> expected;
  expected.emplace(0);
  expected.emplace(1);
  BOOST_CHECK(vGraphs == expected);
}
