// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PingPongShardingTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/pingpong.hpp>

using namespace popart;

// Model: 2x2 PingPong, repeated N times:
// _____________________________________________________________________________
// phase 0:            IPU 0            |                       IPU 2
// in0 ---- Slice/Slice -----------------------------.
//            |                         |            |
// w0 ----- MatMul                      |          MatMul ----- w1
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//______________________________________X__(inter-phase cross-IPU copy)_________
// phase 1:            IPU 1           /|\                      IPU 3
//            .-----------------------' | '----------.
//            |                         |            |
// w2 ----- MatMul                      |          MatMul ----- w3
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//                                      X  (intra-phase cross-IPU copy)
//                                     /|\
//            .-----------------------' | '----------.
//            |                         |            |
// w4 ----- MatMul                      |          MatMul ----- w5
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//______________________________________X_______________________________________
// phase 2:            IPU 0           /|\                      IPU 2
// ......                               |
// ......                               |
//______________________________________X__(inter-phase cross-IPU copy)_________
// phase N*2-1:        IPU 1           /|\                      IPU 3
//            .-----------------------' | '----------.
//            |                         |            |
// w2 ----- MatMul                      |          MatMul ----- w3
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//                                      X  (intra-phase cross-IPU copy)
//                                     /|\
//            .-----------------------' | '----------.
//            |                         |            |
// w4 ----- MatMul                      |          MatMul ----- w5
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------------------ Sum ----- L1Loss
//______________________________________|_______________________________________

BOOST_AUTO_TEST_CASE(Test2x2PingPong) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 5;
  int size          = 100;

  // Weights are [size, size]
  // Input and Acts are [1, size] or [1, 2 * size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{1, 2 * size}};
    auto input = builder.addInputTensor(inInfo);

    std::vector<std::string> insl0(2);
    std::vector<std::string> insl1(2);
    insl0[0] = aiOnnx.slice({input}, {size}, {0}, {1}, "CHECKOP_SL0");
    insl0[1] = aiOnnx.slice({input}, {2 * size}, {size}, {1}, "CHECKOP_SL1");

    builder.pingPongPhase(insl0.at(0), 0);
    builder.pingPongPhase(insl0.at(1), 0);
    builder.virtualGraph(insl0.at(0), 0);
    builder.virtualGraph(insl0.at(1), 0);

    // 2N phases
    for (int n = 0; n < 2 * N; ++n) {
      // Switch between 1 and 2 MatMul blocks per phase
      for (int j = 0; j < 1 + n % 2; ++j) {
        for (int ipu = 0; ipu < 2; ++ipu) {
          auto w        = builder.addInitializedInputTensor(wCVData);
          VGraphId vgid = (n % 2 + 2 * ipu);
          auto out      = aiOnnx.matmul(
              {insl0[ipu], w},
              logging::format("CHECKOP_MM: [{} {}]", n, vgid % 2));
          builder.pingPongPhase(out, n);
          builder.virtualGraph(out, vgid);
          out = aiOnnx.relu(
              {out}, logging::format("CHECKOP_RELU: [{} {}]", n, vgid % 2));
          builder.pingPongPhase(out, n);
          builder.virtualGraph(out, vgid);

          // Cross over between IPUs (intra- or inter-phase)
          insl1[(ipu + 1) % 2] = out;
        }
        insl0 = insl1;
      }
    }
    auto sum = aiOnnx.sum(insl0);
    builder.pingPongPhase(sum, N * 2 - 1);
    builder.virtualGraph(sum, 3);

    auto loss = new IdentityLoss(sum, "l1LossVal", ReductionType::Mean);
    loss->virtualGraph(3);

    // To make introspecting the IR easy
    runner.opts.enableOutlining  = false;
    runner.opts.pingPongPhases   = N * 2;
    runner.opts.virtualGraphMode = VirtualGraphMode::PingPong;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.losses.push_back(loss);

    return sum;
  });

  // Testing that the schedule makes sense for 2x2 PingPong execution:
  // 1.) VGIDs and PingPongPhases of the MatMul and ReLU stay consistent
  // 2.) Inter- and intra-phase IpuCopyOps are placed correctly
  // 3.) Initial SliceOps is placed correctly
  // 4.) Final loss op is placed correctly
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule = ir.getOpSchedule({});
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {}", op->debugName());

      // 1.)
      if (op->getName().find("CHECKOP_MM") != std::string::npos) {
        PingPongPhase n = op->getPingPongPhase();
        VGraphId vgid   = op->getVirtualGraphId();
        if (op->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_MM: [{} {}]", n, vgid % 2)) !=
                      std::string::npos);
        }
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_MM: [{} {}]", N * 4 - n - 2, vgid % 2)) !=
                      std::string::npos);
        }
      }
      if (op->getName().find("CHECKOP_RELU") != std::string::npos) {
        PingPongPhase n = op->getPingPongPhase();
        VGraphId vgid   = op->getVirtualGraphId();
        if (op->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_RELU: [{} {}]", n, vgid % 2)) !=
                      std::string::npos);
        }
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_RELU: [{} {}]", N * 4 - n - 2, vgid % 2)) !=
                      std::string::npos);
        }
      }

      // 2.)
      if (op->isIpuCopyOp()) {
        // IpuCopyOps should not have the VGID set
        BOOST_CHECK(!op->hasVirtualGraphId());
        IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
        if (copy && copy->getSourceIpu() % 2 != copy->getDestIpu() % 2) {
          // Inter-phase
          // See pingpong.cpp ipuCopyPriority
          BOOST_CHECK(copy->settings.schedulePriority == -9998.0);
        } else {
          // Intra-phase
          BOOST_CHECK(copy->settings.schedulePriority == 0.0);
        }
      }

      // 3.)
      if (op->getName().find("CHECKOP_SL0") != std::string::npos ||
          op->getName().find("CHECKOP_SL1") != std::string::npos) {
        BOOST_CHECK(op->getVirtualGraphId() == 0);
        BOOST_CHECK(op->getPingPongPhase() == 0);
      }

      // 4.)
      if (op->isLossOp()) {
        BOOST_CHECK(op->getVirtualGraphId() == 3);
        BOOST_CHECK(op->getPingPongPhase() == N * 2 - 1);
      }
    }
  });
}
