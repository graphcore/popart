#define BOOST_TEST_MODULE PipelineTopoConTest0

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>

// We check that the topological constraints on Stash and Restore are satisfied
BOOST_AUTO_TEST_CASE(PipelineTopoConTest0) {

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};

  auto input1 = builder->addInputTensor(info);

  std::vector<float> w0Vals(2 * 2, 1.0f);
  ConstVoidData w0Data = {w0Vals.data(), info};
  auto w0              = builder->addInitializedInputTensor(w0Data);
  auto act0            = aiOnnx.add({w0, input1}, "act0");
  act0                 = aiOnnx.relu({act0});

  std::vector<float> w1Vals(2 * 2, 1.0f);
  ConstVoidData w1Data = {w1Vals.data(), info};
  auto w1              = builder->addInitializedInputTensor(w1Data);
  auto act1            = aiOnnx.add({w1, act0}, "act1");
  act1                 = aiOnnx.relu({act1});

  act1 = aiOnnx.sin({act1});
  act1 = aiOnnx.cos({act1});
  act1 = aiOnnx.mul({act0, act1});
  act1 = aiOnnx.sigmoid({act1});
  act1 = aiOnnx.sin({act1});
  act1 = aiOnnx.sigmoid({act1});

  std::vector<float> w2Vals(2 * 2, 1.0f);
  ConstVoidData w2Data = {w2Vals.data(), info};
  auto w2              = builder->addInitializedInputTensor(w2Data);
  auto act2            = aiOnnx.add({w2, act1}, "act2");
  act2                 = aiOnnx.relu({act2});

  std::vector<float> w3Vals(2 * 2, 1.0f);
  ConstVoidData w3Data = {w3Vals.data(), info};
  auto w3              = builder->addInitializedInputTensor(w3Data);
  auto act3            = aiOnnx.add({w3, act2}, "act3");
  act3                 = aiOnnx.relu({act3});

  std::vector<float> w4Vals(2 * 2, 1.0f);
  ConstVoidData w4Data = {w4Vals.data(), info};
  auto w4              = builder->addInitializedInputTensor(w4Data);
  auto act4            = aiOnnx.add({w4, act3}, "act4");
  act4                 = aiOnnx.relu({act4});

  std::vector<float> w5Vals(2 * 2, 1.0f);
  ConstVoidData w5Data = {w5Vals.data(), info};
  auto w5              = builder->addInitializedInputTensor(w5Data);
  auto act5            = aiOnnx.add({w5, act4}, "act5");
  act5                 = aiOnnx.relu({act5});

  builder->addOutputTensor(act5);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto dataFlow = DataFlow(10, {{act0, AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.enableVirtualGraphs = true;
  userOptions.autoVirtualGraph    = true;
  userOptions.enablePipelining    = true;

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "3"}};

  auto optimizer = ConstSGD(0.01);

  auto loss = std::unique_ptr<Loss>(
      new L1Loss(act5, "l1LossVal", 0.1, ReductionType::SUM));

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

  //
  // Testing starts now
  //

  auto opSchedule = ir.getOpSchedule({});
  for (auto op : opSchedule) {
    std::cout << op->str() << std::endl;
  }

  std::map<int64_t, std::vector<Op *>> schedsPerIPU;
  for (auto op : opSchedule) {
    int64_t id;
    if (op->hasVirtualGraphId()) {
      id = op->getVirtualGraphId();
    } else if (dynamic_cast<IpuCopyOp *>(op)) {
      id = dynamic_cast<IpuCopyOp *>(op)->getSourceIpu();
    } else {
      throw error("failed to determine IpuId for Op {}", op->str());
    }
    auto found = schedsPerIPU.find(id);
    if (found == schedsPerIPU.end()) {
      schedsPerIPU.insert({id, {op}});
    } else {
      found->second.push_back(op);
    }
  }

  for (auto ipu_ops : schedsPerIPU) {
    std::stringstream ss;
    ss << ipu_ops.first << "On IPU : "
       << "\n";
    for (auto op : ipu_ops.second) {
      op->append(ss);
    }
    std::cout << ss.str();
  }

  std::map<Op *, int> schedIndex;
  for (int i = 0; i < opSchedule.size(); ++i) {
    schedIndex.insert({opSchedule[i], i});
  }

  for (auto op : opSchedule) {
    auto opIndex = schedIndex.at(op);

    // If it's a Stash, we check that
    // 1) Stash appears before the corresponding Restore (we
    // confirm there is 1 corresponding Restore)
    // 2) Stash has ScheduledPreLoss::Yes.
    // 3) Consumers of stashed Tensor with PathFromLoss::Yes appear after Stash
    auto stashOp = dynamic_cast<StashOp *>(op);
    if (stashOp) {
      // 2)
      BOOST_CHECK(stashOp->scheduledPreLoss == ScheduledPreLoss::Yes);
      int nRestores = 0;
      for (auto consumer : stashOp->input->tensor(0)->consumers.getOps()) {
        // 3)
        if (consumer->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(schedIndex.at(consumer) > opIndex);
        }
        // 1)
        if (dynamic_cast<RestoreOp *>(consumer)) {
          ++nRestores;
          BOOST_CHECK(schedIndex.at(consumer) > opIndex);
        }
      }
      // 1)
      BOOST_CHECK(nRestores == 1);
    }

    // if it's Restore, we check
    // 1) Restore happens after it's unique corresponding Stash
    // 2) Restore has ScheduledPreLoss::No
    // 3) Consumers of restored Tensor with PathFromLoss::Yes appear after
    // Restore 4) Consumers of restored Tensor with PathFromLoss::No appear
    // before Restore
    auto restoreOp = dynamic_cast<RestoreOp *>(op);
    if (restoreOp) {
      // 2)
      BOOST_CHECK(restoreOp->scheduledPreLoss == ScheduledPreLoss::No);
      int nStash   = 0;
      auto inIndex = restoreOp->getActToRestoreInIndex();
      auto act     = restoreOp->input->tensor(inIndex);
      for (auto consumer : act->consumers.getOps()) {
        // 4)
        if (consumer->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(schedIndex.at(consumer) < opIndex);
        }
        // 3)
        if (consumer->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(schedIndex.at(consumer) > opIndex);
        }
        if (dynamic_cast<StashOp *>(consumer)) {
          // 1)
          ++nStash;
          BOOST_CHECK(schedIndex.at(consumer) < opIndex);
        }
      }
      // 1)
      BOOST_CHECK(nStash == 1);
    }
  }
}
