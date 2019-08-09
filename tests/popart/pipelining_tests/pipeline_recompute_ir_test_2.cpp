#define BOOST_TEST_MODULE PipelineRecomputeIrTest2

#include "pipeline_recompute_string.hpp"
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>

BOOST_AUTO_TEST_CASE(PipelineRecomputeIrTest2) {

  bool withLogging = true;

  using namespace popart;

  enum class NlType { Sin, Sigmoid };

  auto test = [withLogging](NlType nlt, bool recomp) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    TensorInfo info{"FLOAT", std::vector<int64_t>{8, 4}};
    std::vector<float> wVals(4 * 8, 1.0f);
    ConstVoidData wData = {wVals.data(), info};

    auto input1 = builder->addInputTensor(info);
    auto w1     = builder->addInitializedInputTensor(wData);

    auto act = aiOnnx.add({input1, w1});

    auto getPipe = [nlt, &aiOnnx, &builder, &aiGraphcore](TensorId act,
                                                          VGraphId vgid) {
      //    >-------- [8,4] ---------------------------------------
      //             /     \                                       |
      //      slice left  slice right                              |
      //          scale          scale                             |
      //          [6,4]            [6,4]--------                   |
      //             |                          \                  |
      //             |                         / \
      //            / \                       /   \                |
      //           /   \                     /     \
      //          /     \             slice left  slice right      |
      //   slice left  slice right         scale   scale
      //        scale   scale            [4,4]       [4,4]
      //      [4,4]       [4,4]             |          |           |
      //        |           |            postnl      postnl
      //      postnl      postnl            |        /
      //        \          /                |      /
      //         \        /                 |    /                 |
      //          \      /                 matmul
      //           matmul                   /
      //             |                  sigmoid
      //           sigmoid              /
      //              \                |                           |
      //                ------------- cat [8,4] ---------->       add
      //                                    |                      |
      //                                    |                    sigmoid
      //                                    |                      |
      //                                    ------------------->  add
      //                                                           |
      //                                                           -->
      //                where postnl is either sin or sigmoid
      // The difference between sin and sigmoid, is that the
      // gradients require the input and output, respectively

      auto postnl = [&aiOnnx, &builder, nlt](TensorId id) {
        if (nlt == NlType::Sigmoid) {
          return aiOnnx.sigmoid({id});
        } else {
          return aiOnnx.sin({id});
        }
      };

      auto actIn = act;
      auto act0  = aiOnnx.slice({act}, {6, 4}, {0, 0}, {0, 1});
      act0       = aiGraphcore.scale({act0}, 0.6);

      auto act1 = aiOnnx.slice({act}, {8, 4}, {2, 0}, {0, 1});
      act1      = aiGraphcore.scale({act1}, 0.7);

      auto act00 = aiOnnx.slice({act0}, {4, 4}, {0, 0}, {0, 1});
      act00      = aiGraphcore.scale({act00}, 0.8);
      act00      = postnl(act00);

      auto act01 = aiOnnx.slice({act0}, {6, 4}, {2, 0}, {0, 1});
      act01      = aiGraphcore.scale({act01}, 0.9);
      act01      = postnl(act01);

      auto act10 = aiOnnx.slice({act1}, {4, 4}, {0, 0}, {0, 1});
      act10      = aiGraphcore.scale({act10}, 1.1);
      act10      = postnl(act10);

      auto act11 = aiOnnx.slice({act1}, {6, 4}, {2, 0}, {0, 1});
      act11      = aiGraphcore.scale({act11}, 1.2);
      act11      = postnl(act11);

      act0 = aiOnnx.matmul({act00, act01});
      act0 = aiOnnx.sigmoid({act0});

      act1 = aiOnnx.matmul({act10, act11});
      act1 = aiOnnx.sigmoid({act1});

      auto cat = aiOnnx.concat({act0, act1}, 0);
      act      = aiOnnx.add({cat, actIn});
      act      = aiOnnx.sigmoid({act});

      act = aiOnnx.add({act, cat});

      return act;
    };

    act = getPipe(act, 0);
    act = getPipe(act, 1);
    act = getPipe(act, 2);
    act = getPipe(act, 3);

    builder->addOutputTensor(act);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto dataFlow   = DataFlow(100, {{act, AnchorReturnType("ALL")}});

    SessionOptions userOptions;
    userOptions.enableVirtualGraphs = true;
    userOptions.enableOutlining     = false;
    userOptions.autoVirtualGraph    = true;
    userOptions.enablePipelining    = true;
    if (recomp) {
      userOptions.autoRecomputation = RecomputationType::Standard;
    }

    constexpr int64_t nIpus{4};

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

    auto sched = ir.getMainGraph().getOpSchedule({});

    std::vector<int64_t> stashIpus(nIpus, 0);
    for (auto op : sched) {
      if (dynamic_cast<StashOp *>(op)) {
        ++stashIpus[op->getVirtualGraphId()];
      }

      // Backwards pass Ops must not be RECOMPUTE
      if (op->fromLoss == PathFromLoss::Yes) {
        BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
      }
    }

    for (auto ipu = 0; ipu < nIpus - 1; ++ipu) {
      if (nlt == NlType::Sigmoid && recomp == false) {
        BOOST_CHECK(stashIpus[ipu] == 7);
      }

      else if (nlt == NlType::Sigmoid && recomp == true) {
        // Two of the sigmoid outputs will be recomputed
        BOOST_CHECK(stashIpus[ipu] == 5);
      }
    }

    if (withLogging) {
      std::array<std::stringstream, nIpus> sss;
      pipeline_recompute_util::fillLogStreams(sss, sched);
      for (int64_t ipu = 0; ipu < nIpus; ++ipu) {
        std::cout << "On IPU " << ipu << std::endl;
        std::cout << sss[ipu].str() << "\n\n" << std::endl;
      }
    }
  };

  // Assumptions made in this test:
  // - The auto-sharder will put 1 of the above pipes on each IPU.
  // - Sin grad requires Sin's input and Sigmoid grad requires Sigmoid's output
  test(NlType::Sigmoid, true);
  test(NlType::Sigmoid, false);

  // testing with NlType::Sin assumes that the Cos for the backwards pass is
  // always scheduled in the forwards pass, which seems like a bad long-term
  // assumption. So not testing NlType::Sin
}
