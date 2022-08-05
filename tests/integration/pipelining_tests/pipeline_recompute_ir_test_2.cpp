// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineRecomputeIrTest2

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/stash.hpp>
#include <popart/sgd.hpp>

#include "pipeline_recompute_string.hpp"
#include "popart/builder.gen.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vertex.hpp"
#include "popart/voiddata.hpp"

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

    int numSigmoidsPerPipe      = 3;
    int numBinaryPostNlsPerPipe = 4;

    auto getPipe = [nlt, &aiOnnx, &aiGraphcore](TensorId act, VGraphId vgid) {
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

      auto postnl = [&aiOnnx, nlt](TensorId id) {
        if (nlt == NlType::Sigmoid) {
          return aiOnnx.sigmoid({id});
        } else {
          return aiOnnx.sin({id});
        }
      };
      (void)vgid;

      auto actIn = act;
      auto act0  = aiOnnx.slice({actIn}, {6, 4}, {0, 0}, {0, 1});
      act0       = aiGraphcore.scale({act0}, 0.6);

      auto act1 = aiOnnx.slice({actIn}, {8, 4}, {2, 0}, {0, 1});
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

    auto l1 = builder->aiGraphcoreOpset1().l1loss({act}, 0.1);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto dataFlow   = DataFlow(100, {{act, AnchorReturnType("All")}});

    SessionOptions userOptions;
    userOptions.virtualGraphMode = VirtualGraphMode::Auto;
    userOptions.enableOutlining  = false;
    userOptions.enablePipelining = true;
    if (recomp) {
      userOptions.autoRecomputation = RecomputationType::Standard;
    }

    constexpr int64_t nIpus{4};

    std::map<std::string, std::string> deviceOpts{
        {"numIPUs", std::to_string(nIpus)}};

    auto optimizer = ConstSGD(0.01);

    auto device = createTestDevice(TEST_TARGET, nIpus);

    Patterns patterns(PatternsLevel::Default);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                userOptions,
                patterns});

    auto sched =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

    std::vector<int64_t> stashIpus(nIpus, 0);
    for (auto op : sched) {
      if (dynamic_cast<StashOp *>(op)) {
        ++stashIpus[op->getVirtualGraphId()];
      }

      // Backwards pass Ops must not be Recompute
      if (op->fromLoss == PathFromLoss::Yes) {
        BOOST_CHECK(op->settings.recomputeType == RecomputeType::Checkpoint);
      }
    }

    // NOTE: This calculation may not be quite correct. If this
    // test fails due to this figure being wrong, check what the calculation
    // should be based off the model. In particular, the following are somewhat
    // conjecture:
    //   1. expectedStashesPerPostNl for NlType::Sin, as this case seems to be
    //      intentionally untested.
    //   2. The calculation of the additional number of stashes due to the
    //      outplace ReshapeOps (with and without recomputation).
    //
    // SigmoidGrads require grads of the one output, SinGrads require grads of
    // both inputs.
    const auto expectedStashesPerPostNl = nlt == NlType::Sigmoid ? 1 : 2;
    // Two of the sigmoid outputs will be recomputed.
    const auto expectedNumRecomputedSigmoids = recomp ? 2 : 0;

    // Recall the Pattern `MatMulOp` will Reshape both inputs of all MatMuls in
    // the forward pass. In our model, due to the inputs being overlapping
    // slices of the same tensor, the Reshapes will not be inplace, and thus
    // their outputs will be stashed. However, given the model and the way
    // RecomputationType::Standard is currently implemented, all these Reshapes
    // will be recomputed in the backward pass and therefore not stashed.
    const auto reshapesPerMatMul = 2;
    const auto numMatMuls        = 2;
    const auto numOutplaceReshapes =
        recomp ? 0 : reshapesPerMatMul * numMatMuls;
    const auto expectedStashesPerIpu =
        numSigmoidsPerPipe + numOutplaceReshapes +
        (expectedStashesPerPostNl * numBinaryPostNlsPerPipe) -
        expectedNumRecomputedSigmoids;

    for (auto ipu = 0; ipu < nIpus - 1; ++ipu) {
      BOOST_CHECK(stashIpus[ipu] == expectedStashesPerIpu);
    }

    if (withLogging) {
      std::cout << "NlType = " << (nlt == NlType::Sigmoid ? "Sigmoid" : "Sin")
                << std::endl;
      std::cout << "recomp = " << std::boolalpha << recomp << std::noboolalpha
                << std::endl;

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
