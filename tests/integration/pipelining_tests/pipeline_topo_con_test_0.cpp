// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineTopoConTest0

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

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

  auto w3      = builder->addInitializedInputTensor(wData);
  auto act3    = aiOnnx.add({w3, act2}, "act3");
  act3         = aiOnnx.relu({act3});
  auto act3nll = aiGraphcore.nllloss({act3, l0}, ReductionType::Mean);

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
  auto act6l1 = aiGraphcore.l1loss({act6}, 0.1, ReductionType::Mean);

  auto act7 = aiOnnx.sin({act6});
  act7      = aiOnnx.sigmoid({act6});
  act7      = aiOnnx.add({act6, act7});

  auto act8   = aiOnnx.mul({act7, act7});
  auto w6     = builder->addInitializedInputTensor(wData);
  act8        = aiOnnx.mul({act8, w6});
  act8        = aiOnnx.add({act8, input3});
  auto act8l1 = aiGraphcore.l1loss({act8}, 0.2, ReductionType::Sum);

  auto act9    = aiOnnx.relu({act8});
  act9         = aiOnnx.softmax({act9});
  auto act9nll = aiGraphcore.nllloss({act9, l1}, ReductionType::Sum);

  auto finalLoss = aiOnnx.sum({act6l1, act8l1, act3nll, act9nll});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto dataFlow = DataFlow(10, {{act4, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Auto;
  userOptions.enablePipelining = true;

  auto optimizer = ConstSGD(0.01);

  auto device = createTestDevice(TEST_TARGET, 3);

  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              finalLoss,
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::Default)});
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

  auto l1 = builder->aiGraphcoreOpset1().l1loss({act5}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  auto dataFlow = DataFlow(10, {{act0, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Auto;
  userOptions.enablePipelining = true;

  auto optimizer = ConstSGD(0.01);

  auto device = createTestDevice(TEST_TARGET, 3);

  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              l1,
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::Default)});
}

// We check that the topological constraints on Stash and Restore are satisfied
BOOST_AUTO_TEST_CASE(PipelineTopoConTest0) {

  using namespace popart;

  auto test = [](Ir &ir) {
    auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
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

    std::map<Op *, int, POpCmp> schedIndex;
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
        int nRestores = 0;
        for (auto consumer : stashOp->output->tensor(0)->consumers.getOps()) {
          // 1)
          if (dynamic_cast<RestoreOp *>(consumer)) {
            ++nRestores;
            BOOST_CHECK(schedIndex.at(consumer) > opIndex);
          }
        }
        // 1)
        BOOST_CHECK(nRestores == 1);

        // 2)
        BOOST_CHECK(stashOp->scheduledPreLoss == ScheduledPreLoss::Yes);
        for (auto consumer : stashOp->input->tensor(0)->consumers.getOps()) {
          // 3)
          if (consumer->fromLoss == PathFromLoss::Yes) {
            BOOST_CHECK(schedIndex.at(consumer) > opIndex);
          }
        }
      }

      // if it's a Restore Op, we don't check any ordering conditions in the Ir,
      // because the backend is expected to control the scheduling of Restore
      // Ops, which happens later
      auto restoreOp = dynamic_cast<RestoreOp *>(op);
      if (restoreOp) {
        int nStash   = 0;
        auto stash   = restoreOp->input->tensor(restoreOp->getStashInIndex());
        auto stashOp = stash->getProducer();
        auto act     = stashOp->input->tensor(StashOp::getInIndex());
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
