// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineRecomputeIrTest1

#include <memory>

#include "pipeline_recompute_string.hpp"
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(PipelineRecomputeIrTest1) {

  bool withLogging = true;

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  TensorInfo info{"FLOAT", std::vector<int64_t>{4, 4}};
  std::vector<float> wVals(4 * 6, 1.0f);
  ConstVoidData wData = {wVals.data(), info};

  auto input1 = builder->addInputTensor(info);
  auto w1     = builder->addInitializedInputTensor(wData);

  auto act = aiOnnx.add({input1, w1});
  builder->virtualGraph(act, 0);

  auto getPipe = [&aiOnnx, &aiGraphcore, &builder](TensorId act,
                                                   VGraphId vgid) {
    // >-----
    //      |
    //  Sigmoid (to stash)
    //      |
    //   -------
    //   |  |  |
    //   |  |  | (everything in here can be recomputed)
    //   |  |  |
    //   -------
    //      |
    //      |
    //      ------>

    act = aiOnnx.sigmoid({act});
    builder->virtualGraph(act, vgid);

    auto act0 = aiOnnx.sin({act});
    builder->virtualGraph(act0, vgid);

    auto act1 = aiOnnx.cos({act});
    builder->virtualGraph(act1, vgid);

    auto act2 = aiOnnx.exp({act});
    builder->virtualGraph(act2, vgid);

    for (int i = 0; i < 2; ++i) {
      auto act3 = aiOnnx.matmul({act0, act1});
      builder->virtualGraph(act3, vgid);

      auto act4 = aiOnnx.matmul({act1, act2});
      builder->virtualGraph(act4, vgid);

      auto act5 = aiOnnx.matmul({act2, act0});
      builder->virtualGraph(act5, vgid);

      act0 = aiOnnx.sin({act3});
      builder->virtualGraph(act0, vgid);

      act1 = aiOnnx.cos({act4});
      builder->virtualGraph(act1, vgid);

      act2 = aiOnnx.exp({act5});
      builder->virtualGraph(act2, vgid);
    }
    act = aiOnnx.sum({act0, act1, act2});
    builder->virtualGraph(act, vgid);
    act = aiGraphcore.scale({act}, 1.3);
    builder->virtualGraph(act, vgid);

    return act;
  };

  // to stash on IPU 0 : only the output of sigmoid
  // (sigmoid grad takes in output output of sigmoid)
  act = getPipe(act, 0);

  // to stash on IPU 1 : only the output of first sigmoid
  act = getPipe(act, 1);

  // to stash on IPU 2 : only the output of first sigmoid
  act = getPipe(act, 2);

  // to stash on IPU 3 : none, as it it the final IPU.
  act = getPipe(act, 3);

  act = builder->aiGraphcoreOpset1().l1loss({act}, 0.1);
  builder->virtualGraph(act, 3);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto dataFlow   = DataFlow(100, {{act, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode  = VirtualGraphMode::Manual;
  userOptions.enablePipelining  = true;
  userOptions.autoRecomputation = RecomputationType::Standard;

  constexpr int64_t nIpus{4};

  auto optimizer = ConstSGD(0.01);

  auto loss1 =
      std::make_shared<IdentityLoss>(act, "l1LossVal_1", ReductionType::Mean);

  loss1->virtualGraph(3);

  auto device = createTestDevice(TEST_TARGET, nIpus);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss1},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::Default)});

  auto sched = ir.getMainGraph().getOpSchedule({});

  std::vector<int64_t> stashIpus;
  for (auto op : sched) {
    if (dynamic_cast<StashOp *>(op)) {
      std::stringstream ss;
      dynamic_cast<StashOp *>(op)->append(ss);
      std::cout << ss.str() << std::endl;
      stashIpus.push_back(op->getVirtualGraphId());
    }

    // Backwards pass Ops must not be Recompute
    if (op->fromLoss == PathFromLoss::Yes) {
      BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
    }
  }

  // unique stashes on all but last IPU.
  std::cout << "number of stashes: " << stashIpus.size()
            << " number of IPUs: " << nIpus << std::endl;
  BOOST_CHECK(stashIpus.size() == nIpus - 1);
  for (int64_t ipu = 0; ipu < nIpus - 1; ++ipu) {
    BOOST_CHECK(std::find(stashIpus.begin(), stashIpus.end(), ipu) !=
                stashIpus.end());
  }

  if (withLogging) {
    std::array<std::stringstream, nIpus> sss;
    pipeline_recompute_util::fillLogStreams(sss, sched);
    for (int64_t ipu = 0; ipu < nIpus; ++ipu) {
      std::cout << "On IPU " << ipu << std::endl;
      std::cout << sss[ipu].str() << "\n\n" << std::endl;
    }
  }
}
