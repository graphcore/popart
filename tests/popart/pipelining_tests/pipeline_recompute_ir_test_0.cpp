// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineRecomputeIrTest0

#include "pipeline_recompute_string.hpp"
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(PipelineNoMultiSourceTest0) {

  bool withLogging = true;

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  TensorInfo info{"FLOAT", std::vector<int64_t>{4, 6}};
  std::vector<float> wVals(4 * 6, 1.0f);
  ConstVoidData wData = {wVals.data(), info};

  auto input1 = builder->addInputTensor(info);
  auto w1     = builder->addInitializedInputTensor(wData);

  constexpr int64_t nIpus{3};

  auto act = aiOnnx.add({input1, w1});
  builder->virtualGraph(act, 0);

  for (int vgid = 0; vgid < nIpus; ++vgid) {
    for (int i = 0; i < 2; ++i) {
      act = aiOnnx.sigmoid({act});
      builder->virtualGraph(act, vgid);
    }
    act = aiGraphcore.scale({act}, 1.55);
    builder->virtualGraph(act, vgid);
  }
  builder->addOutputTensor(act);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto dataFlow   = DataFlow(100, {{act, AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode  = VirtualGraphMode::Manual;
  userOptions.enablePipelining  = true;
  userOptions.autoRecomputation = RecomputationType::Standard;

  auto optimizer = ConstSGD(0.01);

  auto loss1 = std::unique_ptr<Loss>(
      new L1Loss(act, "l1LossVal_1", 0.1, ReductionType::MEAN));
  loss1->virtualGraph(2);

  auto device = createTestDevice(TEST_TARGET, nIpus);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss1.get()},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::DEFAULT)});

  auto sched = ir.getOpSchedule({});

  std::vector<int64_t> stashIpus;
  for (auto op : sched) {
    if (dynamic_cast<StashOp *>(op)) {
      stashIpus.push_back(op->getVirtualGraphId());
    }

    // Backwards pass Ops must not be RECOMPUTE
    if (op->fromLoss == PathFromLoss::Yes) {
      BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
    }
  }

  // 2 stashes, one on all but last IPU.
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
