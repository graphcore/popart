#define BOOST_TEST_MODULE PipelineTopoConTest0

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

void prepareIr1(popart::Ir &ir) {

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo info{"FLOAT", std::vector<int64_t>{2, 2}};

  TensorInfo info_l{"INT32", std::vector<int64_t>{2}};

  auto input1 = builder->addInputTensor(info);
  auto input2 = builder->addInputTensor(info);
  auto input3 = builder->addInputTensor(info);

  auto l0 = builder->addInputTensor(info_l);
  auto l1 = builder->addInputTensor(info_l);

  std::vector<float> wVals(2 * 2, 1.0f);
  ConstVoidData wData = {wVals.data(), info};

  auto w0   = builder->addInitializedInputTensor(wData);
  auto act0 = aiOnnx.add({w0, input1}, "act0");
  act0      = aiOnnx.relu({act0});

  auto w1   = builder->addInitializedInputTensor(wData);
  auto act1 = aiOnnx.add({w1, act0}, "act1");
  act1      = aiOnnx.relu({act1});

  act1 = aiOnnx.sin({act1});
  act1 = aiOnnx.cos({act1});
  act1 = aiOnnx.mul({act0, act1});
  act1 = aiOnnx.sigmoid({act1});
  act1 = aiOnnx.sin({act1});
  act1 = aiOnnx.sigmoid({act1});

  auto w2   = builder->addInitializedInputTensor(wData);
  auto act2 = aiOnnx.add({w2, act1}, "act2");
  act2      = aiOnnx.relu({act2});

  auto w3   = builder->addInitializedInputTensor(wData);
  auto act3 = aiOnnx.add({w2, act2}, "act3");
  act3      = aiOnnx.relu({act3});

  auto act4 = aiOnnx.add({act2, act3});

  act4 = aiOnnx.sigmoid({act4});
  act4 = aiOnnx.sin({act4});
  act4 = aiOnnx.sigmoid({act4});

  auto w5   = builder->addInitializedInputTensor(wData);
  auto act5 = aiOnnx.mul({act4, w5});
  auto act6 = aiOnnx.mul({act5, act3});

  act6 = aiOnnx.mul({act6, input2});

  // TODO : IpuCopyOp copies Tensors from multiple sources (T10373) are required
  // to handle this extension:
  //  act6 = aiOnnx.mul({act6, input1});
  //  act6 = aiOnnx.mul({act6, input3});

  auto act7 = aiOnnx.sin({act6});
  act7      = aiOnnx.sigmoid({act6});
  act7      = aiOnnx.add({act6, act7});

  auto act8 = aiOnnx.mul({act7, act7});
  auto w6   = builder->addInitializedInputTensor(wData);
  act8      = aiOnnx.mul({act8, w6});
  act8      = aiOnnx.add({act8, input3});

  auto act9 = aiOnnx.relu({act8});
  act9      = aiOnnx.softmax({act9});

  builder->addOutputTensor(act6);
  builder->addOutputTensor(act3);
  builder->addOutputTensor(act8);
  builder->addOutputTensor(act9);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto dataFlow = DataFlow(10, {{act4, AnchorReturnType("ALL")}});

  SessionOptions userOptions;
  userOptions.enableVirtualGraphs = true;
  userOptions.autoVirtualGraph    = true;
  userOptions.enablePipelining    = true;

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "3"}};

  auto optimizer = ConstSGD(0.01);

  auto loss1 = std::unique_ptr<Loss>(
      new L1Loss(act6, "l1LossVal_1", 0.1, ReductionType::MEAN));

  auto loss2 = std::unique_ptr<Loss>(
      new L1Loss(act8, "l1LossVal_2", 0.2, ReductionType::SUM));

  // TODO : the inclusion of this Loss on act3 causes a Stash Tensor which is
  // scheduled-pre-loss. This should be fixed T10375
  //   auto loss3 = std::unique_ptr<Loss>(
  //       new NllLoss(act3, l0, "nllLossVal_1", ReductionType::MEAN));

  auto loss4 = std::unique_ptr<Loss>(
      new NllLoss(act9, l1, "nllLossVal_2", ReductionType::SUM));

  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss1.get(), loss2.get(), loss4.get()},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::DEFAULT)});
}

void prepareIr0(popart::Ir &ir) {

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

  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss.get()},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::DEFAULT)});
}

// We check that the topological constraints on Stash and Restore are satisfied
BOOST_AUTO_TEST_CASE(PipelineTopoConTest0) {

  using namespace popart;

  auto test = [](Ir &ir) {
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
      // 3) Consumers of stashed Tensor with PathFromLoss::Yes appear after
      // Stash
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

      // if it's a Restore Op, we don't check any ordering conditions in the Ir,
      // because the backend is expected to control the scheduling of Restore
      // Ops, which happens later
      auto restoreOp = dynamic_cast<RestoreOp *>(op);
      if (restoreOp) {
        int nStash   = 0;
        auto inIndex = restoreOp->getActToRestoreInIndex();
        auto act     = restoreOp->input->tensor(inIndex);
        for (auto consumer : act->consumers.getOps()) {
          if (dynamic_cast<StashOp *>(consumer)) {
            ++nStash;
            BOOST_CHECK(schedIndex.at(consumer) < opIndex);
          }
        }
        BOOST_CHECK(nStash == 1);
      }
    }
  };
  Ir ir0;
  prepareIr0(ir0);
  test(ir0);

  Ir ir1;
  prepareIr1(ir1);
  test(ir1);
}
