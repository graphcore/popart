#define BOOST_TEST_MODULE PipelineNoMultiSourceTest0

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

    auto getPipe = [&aiOnnx,
                    &aiGraphcore](TensorId id1, TensorId id2, std::string num) {
      auto act = aiOnnx.add({id1, id2}, "add1-" + num);
      act      = aiOnnx.sigmoid({act}, "sigmoid-" + num);
      act      = aiOnnx.cos({act}, "cos-" + num);
      act      = aiOnnx.mul({act, id1}, "mul-" + num);
      act      = aiOnnx.add({id2, act}, "add2-" + num);
      act      = aiOnnx.relu({act}, "relu-" + num);

      // Removing this additional non-linearity breaks the scheduler.
      // This is only true when pipelining is enabled (no pipelining =>
      // scheduler is fine). The task to fix this is the TODO T10403.
      act = aiOnnx.sigmoid({act}, "sigmoid-" + num);

      act = aiGraphcore.scale({act}, 5, "scale-" + num);
      return act;
    };

    auto act1 = getPipe(input1, w1, "1");
    auto act2 = getPipe(input2, w2, "2");
    auto act3 = getPipe(input3, w3, "3");
    auto act4 = getPipe(input4, w4, "4");
    auto act5 = getPipe(input5, w5, "5");

    auto act = aiOnnx.add({act1, act2}, "add1-final");
    act      = aiOnnx.sub({act, act3}, "sub-final");
    act      = aiOnnx.mul({act, act4}, "mul-final");
    act      = aiOnnx.add({act, act5}, "add2-final");
    act      = aiOnnx.sigmoid({act}, "sigmoid-final");
    act      = aiOnnx.relu({act}, "relu-final");

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

    builder->addOutputTensor(act);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto dataFlow   = DataFlow(20, {{act1, AnchorReturnType("ALL")}});

    SessionOptions userOptions;
    userOptions.enableVirtualGraphs = true;
    userOptions.autoVirtualGraph    = true;
    userOptions.enablePipelining    = withPipelining;

    constexpr int64_t nIpus{6};
    std::map<std::string, std::string> deviceOpts{
        {"numIPUs", std::to_string(nIpus)}};

    auto optimizer = ConstSGD(0.01);

    auto loss1 = std::unique_ptr<Loss>(
        new L1Loss(act, "l1LossVal_1", 0.1, ReductionType::MEAN));

    auto device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {loss1.get()},
                &optimizer,
                *device,
                userOptions,
                Patterns(PatternsLevel::DEFAULT)});

    int64_t nMultiSource = 0;

    // prepare the final log:
    std::array<std::stringstream, nIpus> sss;

    auto sched = ir.getOpSchedule({});
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
