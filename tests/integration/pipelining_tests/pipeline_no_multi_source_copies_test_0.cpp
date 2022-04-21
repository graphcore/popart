// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineNoMultiSourceTest0

#include <memory>

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
#include <popart/optimizer.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

// We confirm that a model which would normally have an IpuCopyOp
// with multiple sources, can be pipelined as multi-source IpuCopyOps
// are not created when pipelining is enabled.

BOOST_AUTO_TEST_CASE(PipelineNoMultiSourceTest0) {

  using namespace popart;

  auto test = [](bool withPipelining) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    TensorInfo info{"FLOAT", std::vector<int64_t>{4, 6}};
    std::vector<float> wVals(4 * 6, 1.0f);
    ConstVoidData wData = {wVals.data(), info};

    auto input1 = builder->addInputTensor(info);
    auto w1     = builder->addInitializedInputTensor(wData);

    auto input5 = builder->addInputTensor(info);
    auto w5     = builder->addInitializedInputTensor(wData);

    auto input2 = builder->addInputTensor(info);
    auto w2     = builder->addInitializedInputTensor(wData);

    auto input4 = builder->addInputTensor(info);
    auto w4     = builder->addInitializedInputTensor(wData);

    auto input3 = builder->addInputTensor(info);
    auto w3     = builder->addInitializedInputTensor(wData);

    auto getPipe = [&aiOnnx, &builder, &aiGraphcore](TensorId id1,
                                                     TensorId id2,
                                                     const std::string &num,
                                                     int vgid) {
      auto act = aiOnnx.add({id1, id2}, "add1-" + num);
      builder->virtualGraph(act, vgid);
      act = aiOnnx.sigmoid({act}, "sigmoid-" + num);
      builder->virtualGraph(act, vgid);
      act = aiOnnx.cos({act}, "cos-" + num);
      builder->virtualGraph(act, vgid);
      act = aiOnnx.mul({act, id1}, "mul-" + num);
      builder->virtualGraph(act, vgid);

      act = aiOnnx.add({id2, act}, "add2-" + num);
      builder->virtualGraph(act, vgid);

      act = aiOnnx.relu({act}, "relu-" + num);
      builder->virtualGraph(act, vgid);

      act = aiOnnx.sigmoid({act}, "sigmoid-" + num);
      builder->virtualGraph(act, vgid);

      act = aiGraphcore.scale({act}, 5, "scale-" + num);
      builder->virtualGraph(act, vgid);

      return act;
    };

    auto act1 = getPipe(input1, w1, "1", 0);
    auto act2 = getPipe(input2, w2, "2", 1);
    auto act3 = getPipe(input3, w3, "3", 2);
    auto act4 = getPipe(input4, w4, "4", 3);
    auto act5 = getPipe(input5, w5, "5", 4);

    auto act = aiOnnx.add({act1, act2}, "add1-final");
    builder->virtualGraph(act, 5);
    act = aiOnnx.sub({act, act3}, "sub-final");
    builder->virtualGraph(act, 5);
    act = aiOnnx.mul({act, act4}, "mul-final");
    builder->virtualGraph(act, 5);
    act = aiOnnx.add({act, act5}, "add2-final");
    builder->virtualGraph(act, 5);
    act = aiOnnx.sigmoid({act}, "sigmoid-final");
    builder->virtualGraph(act, 5);
    act = aiOnnx.relu({act}, "relu-final");
    builder->virtualGraph(act, 5);
    act = builder->aiGraphcoreOpset1().l1loss({act}, 0.1);
    builder->virtualGraph(act, 5);

    // in1, w1 in2,w2  in3, w3  in4, w4   in5, w5
    // |       |       |        |         |
    // |P      |       |        |         |
    // |I      |       |        |         |
    // |P      |       |        |         |
    // |E      |       |        |         |
    // |1      |       |        |         |P
    // |       |       |        |         |I
    // |       |       |        |         |P
    // ---------       |        |         |E
    //    |            |        |         |5
    //   add           |        |         |
    //    |            |        |         |
    //    -------------|        |         |
    //         |                |         |
    //        sub               |         |
    //         |                |         |
    //         -----------------|         |
    //                 |                  |
    //                mul                 |
    //                 |                  |
    //                 --------------------
    //                        |
    //                       add
    //                        |
    //                      sigmoid
    //                        |
    //                       relu
    //                        |
    //

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto dataFlow   = DataFlow(20, {{act1, AnchorReturnType("All")}});

    SessionOptions userOptions;
    userOptions.virtualGraphMode = VirtualGraphMode::Manual;
    userOptions.enablePipelining = withPipelining;

    constexpr int64_t nIpus{6};
    auto optimizer = ConstSGD(0.01);

    auto device = createTestDevice(TEST_TARGET, nIpus);
    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                act,
                &optimizer,
                *device,
                userOptions,
                Patterns(PatternsLevel::Default)});

    int64_t nMultiSource = 0;

    // prepare the final log:
    std::array<std::stringstream, nIpus> sss;

    auto sched = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (auto op : sched) {
      auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(op);
      if (!ipuCopyOp) {
        auto vgid = op->getVirtualGraphId();
        sss[vgid] << op->wrtLossStr() << "  " << op->getName() << "     "
                  << op->str() << "\n";
      } else {
        int64_t vgid = 0;
        if (ipuCopyOp->getMinSourceIpu() == ipuCopyOp->getMaxSourceIpu()) {
          vgid = ipuCopyOp->getSourceIpu();
        } else {
          nMultiSource += 1;
        }

        sss[vgid] << op->wrtLossStr() << "  " << op->getName() << "     "
                  << op->str() << "      " << ipuCopyOp->getFromToStr() << "\n";
      }
    }

    for (int64_t ipu = 0; ipu < nIpus; ++ipu) {
      std::cout << "On IPU " << ipu << std::endl;
      std::cout << sss[ipu].str() << "\n\n" << std::endl;
    }

    if (withPipelining) {
      BOOST_CHECK(nMultiSource == 0);
    } else {
      std::cout << "Verifying that without pipelining there is a multi-source "
                   "ipu copy. This is not strictly necessary, and future "
                   "changes to sharding might change this"
                << std::endl;
      BOOST_CHECK(nMultiSource > 0);
    }
  };
  test(false);
  test(true);
}
