// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AliasZeroCopyTest

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <random_util.hpp>
#include <popart/aliaszerocopy.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/call.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace popart;

enum class ZeroCopyTestModel {
  //      B              C
  //      |              |
  // A - Call(0) - X0 - Call(0) - X1 - Loss
  // X1 aliased as subgraph output
  // (meaning that "Subgraph(0)/X0" will be aliased to "X1"),
  // but not X0, since X0 will be required
  // in the backward pass and is therefore still alive through the second
  // Call(0)
  TwoMatMuls = 0,

  //      B              C
  //      |              |
  // A - Call(0) - X0 - Call(0) - X1 - Add - X2 - Loss
  //                \__________________/
  // X0/X1 - same as TwoMatMuls
  TwoMatMulsAddOutput,

  //      B -------------.
  //      |              |
  // A - Call(0) - X0 - Call(0) - X1 - Loss
  // B aliased as subgraph input, since both Call(0) use the same weight
  // X0/X1 - same as TwoMatMuls
  TwoMatMulsSharedWeight
};

void compare(const std::vector<std::vector<float>> &a,
             const std::vector<std::vector<float>> &b) {
  for (size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK_EQUAL_COLLECTIONS(
        a.at(i).begin(), a.at(i).end(), b.at(i).begin(), b.at(i).end());
  }
}

// Test zero-copy through aliasing behaviour
// Test models from ZeroCopyTestModel
// Verify that the right subgraph input/output tensors are aliased and
// that the weight update result is numerically identical to the same graph
// trained without alias zero copy enabled
BOOST_AUTO_TEST_CASE(AliasZeroCopyTest0) {
  auto test = [](bool aliasZeroCopy, ZeroCopyTestModel model) {
    std::vector<std::vector<float>> collection;

    int N = 8;

    // we will generate random initializations
    int seed = 1013;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-4.f, 4.f);

    // prepare a Builder for creating onnx model
    auto bder   = Builder::create();
    auto aiOnnx = bder->aiOnnxOpset9();

    // matrix A of shape N
    TensorInfo A_info{"FLOAT", std::vector<int64_t>{1, 1, N}};
    std::vector<float> v_A_init(A_info.nelms());
    for (auto &val : v_A_init) {
      val = fdis(eng);
    }
    TensorId A_id =
        bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

    // matrix B of shape N x N
    TensorInfo B_info{"FLOAT", std::vector<int64_t>{1, N, N}};
    std::vector<float> v_B_init(B_info.nelms());
    for (auto &val : v_B_init) {
      val = fdis(eng);
    }
    TensorId B_id =
        bder->addInitializedInputTensor({v_B_init.data(), B_info}, "B");

    // matrix C of shape N x N
    TensorInfo C_info{"FLOAT", std::vector<int64_t>{1, N, N}};
    std::vector<float> v_C_init(C_info.nelms());
    for (auto &val : v_C_init) {
      val = fdis(eng);
    }
    TensorId C_id =
        bder->addInitializedInputTensor({v_C_init.data(), C_info}, "C");

    TensorInfo X_info{"FLOAT", std::vector<int64_t>{N}};

    TensorId X_id;

    switch (model) {
    case ZeroCopyTestModel::TwoMatMuls: {
      X_id = aiOnnx.matmul({A_id, B_id});
      X_id = aiOnnx.matmul({X_id, C_id});
      break;
    };
    case ZeroCopyTestModel::TwoMatMulsAddOutput: {
      TensorId X1_id = aiOnnx.matmul({A_id, B_id});
      TensorId X2_id = aiOnnx.matmul({X1_id, C_id});
      X_id           = aiOnnx.add({X1_id, X2_id});
      break;
    };
    case ZeroCopyTestModel::TwoMatMulsSharedWeight: {
      X_id = aiOnnx.matmul({A_id, B_id});
      X_id = aiOnnx.matmul({X_id, B_id});
      break;
    };
    }

    // l1 loss with penalty term, will be applied to C
    float lossLambda = 0.26;
    auto l1          = bder->aiGraphcoreOpset1().l1loss({X_id}, lossLambda);

    auto proto      = bder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    auto art        = AnchorReturnType("All");

    // one batch per step
    int batchesPerStep = 1;
    std::map<TensorId, AnchorReturnType> anchorIds;
    auto dataFlow = DataFlow(batchesPerStep, anchorIds);

    auto device = popart::createTestDevice(TEST_TARGET);

    auto opts                           = SessionOptions();
    opts.explicitRecomputation          = true;
    opts.enableOutlining                = true;
    opts.outlineThreshold               = -1.0;
    opts.enableOutliningCopyCostPruning = false;
    opts.aliasZeroCopy                  = aliasZeroCopy;
    opts.delayVarUpdates                = false;

    // training info
    float learnRate = 0.321;
    auto optimizer  = ConstSGD(learnRate);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    std::map<popart::TensorId, popart::IArray &> anchors = {};

    auto &ir = session->getIr();

    session->prepareDevice();

    // inputs:
    popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
    popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
    popart::NDArrayWrapper<float> C_wrapper(v_C_init.data(), C_info);
    std::map<popart::TensorId, popart::IArray &> inputs = {
        {A_id, A_wrapper}, {B_id, B_wrapper}, {C_id, C_wrapper}};

    popart::StepIO stepio(inputs, anchors);

    session->weightsFromHost();
    session->run(stepio);

    // Verify alias zero copy
    if (aliasZeroCopy) {
      auto &dev = session->getDevice();
      auto azc  = dev.lowering().getAliasZeroCopy();

      std::set<TensorId> tensorIdsWithAliases;
      for (Tensor *t : azc->getTensorsWithPostIRAliases()) {
        tensorIdsWithAliases.insert(t->id);
      }
      // For debugging purposes
      logging::trace("Tensors with aliases: {}", tensorIdsWithAliases);

      auto checkAliased = [&ir, &dev, &azc](CallOp *call,
                                            std::pair<int, bool> index,
                                            bool expectAliased) {
        Graph &calledGraph = call->getCalledGraph();
        Tensor *t0;
        Tensor *t1;
        if (index.second) {
          t0 = call->input->tensor(index.first);
          t1 = ir.getTensor(calledGraph.getInputId(index.first));
        } else {
          t0 = call->output->tensor(index.first);
          t1 = ir.getTensor(calledGraph.getOutputId(index.first));
        }
        auto aliases = azc->getActiveAliasedTensors({t0}, true);
        bool aliased = aliases.find(t1) != aliases.end();
        logging::trace(
            "Testing aliasing of {} <-> {}, expected: {}, observed: {}",
            t0->id,
            t1->id,
            expectAliased ? "yes" : "no",
            aliased ? "yes" : "no");
        if (expectAliased) {
          // AliasZeroCopy reports tensors as actively aliased
          BOOST_CHECK(aliased);
          // Additionally verify the poplar tensors agree
          BOOST_CHECK(poplar::concat({dev.lowering().tensors().get(t0->id),
                                      dev.lowering().tensors().get(t1->id)},
                                     0)
                          .containsAliases());
        } else {
          BOOST_CHECK(!aliased);
        }
      };

      auto inTensor = [](InIndex index) {
        return std::pair<InIndex, bool>(index, true);
      };

      auto outTensor = [](OutIndex index) {
        return std::pair<OutIndex, bool>(index, false);
      };

      // Count the number of calls
      size_t callIndex = 0;

      for (Op *op :
           ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes)) {
        logging::trace("Op: {}", op->debugName());

        if (CallOp *call = dynamic_cast<CallOp *>(op)) {
          // Depending on the model, we expect different inputs/outputs to be
          // aliased
          switch (model) {
          case ZeroCopyTestModel::TwoMatMuls: {
            if (callIndex == 0) {
              checkAliased(call, inTensor(0), false);  // A
              checkAliased(call, inTensor(1), false);  // B
              checkAliased(call, outTensor(0), false); // X0
            }
            if (callIndex == 1) {
              checkAliased(call, inTensor(0), false); // X0
              checkAliased(call, inTensor(1), false); // C
              checkAliased(call, outTensor(0), true); // X1
            }
            if (callIndex == 2) {
              checkAliased(call, inTensor(0), false); // grad_X1
              checkAliased(call, inTensor(1), false); // C
              checkAliased(call, inTensor(2), false); // X0
              checkAliased(call, outTensor(0), true); // grad_X0
            }
            if (callIndex == 3) {
              checkAliased(call, inTensor(0), false); // grad_X0
              checkAliased(call, inTensor(1), false); // B
              checkAliased(call, inTensor(2), false); // A
              checkAliased(call, outTensor(0), true); // grad_A
            }
            break;
          };
          case ZeroCopyTestModel::TwoMatMulsAddOutput: {
            if (callIndex == 0) {
              checkAliased(call, inTensor(0), false);  // A
              checkAliased(call, inTensor(1), false);  // B
              checkAliased(call, outTensor(0), false); // X0
            }
            if (callIndex == 1) {
              checkAliased(call, inTensor(0), false); // X0
              checkAliased(call, inTensor(1), false); // C
              checkAliased(call, outTensor(0), true); // X1
            }
            if (callIndex == 2) {
              checkAliased(call, inTensor(0), false); // X0
              checkAliased(call, inTensor(1), true);  // X1
              checkAliased(call, outTensor(0), true); // X2
            }
            if (callIndex == 3) {
              checkAliased(call, inTensor(0), false); // grad_X2
              checkAliased(call, inTensor(1), false); // X0
              checkAliased(call, inTensor(2), false); // C
              checkAliased(call, outTensor(0), true); // grad_X1
            }
            if (callIndex == 4) {
              checkAliased(call, inTensor(0), false); // grad_X2
              checkAliased(call, inTensor(1), false); // grad_X1
              checkAliased(call, outTensor(0), true); // grad_X0
            }
            if (callIndex == 5) {
              checkAliased(call, inTensor(0), false); // grad_X0
              checkAliased(call, inTensor(1), false); // A
              checkAliased(call, inTensor(2), false); // B
              checkAliased(call, outTensor(0), true); // grad_A
            }
            break;
          };
          case ZeroCopyTestModel::TwoMatMulsSharedWeight: {
            if (callIndex == 0) {
              checkAliased(call, inTensor(0), false);  // A
              checkAliased(call, inTensor(1), true);   // B
              checkAliased(call, outTensor(0), false); // X0
            }
            if (callIndex == 1) {
              checkAliased(call, inTensor(0), false); // X0
              checkAliased(call, inTensor(1), true);  // B
              checkAliased(call, outTensor(0), true); // X1
            }
            if (callIndex == 2) {
              checkAliased(call, inTensor(0), false);  // grad_X1
              checkAliased(call, inTensor(1), true);   // B
              checkAliased(call, outTensor(0), false); // grad_X0
            }
            if (callIndex == 3) {
              checkAliased(call, inTensor(0), false); // grad_X0
              checkAliased(call, inTensor(1), true);  // B
              checkAliased(call, outTensor(0), true); // grad_A
            }
            if (callIndex == 4) {
              checkAliased(call, inTensor(0), false);  // grad_X1
              checkAliased(call, inTensor(1), false);  // X0
              checkAliased(call, outTensor(0), false); // grad_B_part
            }
            if (callIndex == 5) {
              checkAliased(call, inTensor(0), false); // grad_X0
              checkAliased(call, inTensor(1), false); // A
              checkAliased(call, outTensor(0), true); // grad_B_part
            }
            break;
          };
          }

          ++callIndex;
        }
      }
    }

    WeightsIO weightsRead;
    // to be readback:
    std::vector<float> A_readback(A_info.nelms(), -1.0f);
    std::vector<float> B_readback(B_info.nelms(), -1.0f);
    std::vector<float> C_readback(C_info.nelms(), -1.0f);
    weightsRead.insert(A_id, {A_readback.data(), A_info});
    weightsRead.insert(B_id, {B_readback.data(), B_info});
    weightsRead.insert(C_id, {C_readback.data(), C_info});

    session->weightsToHost();
    session->readWeights(weightsRead);

    collection.push_back(A_readback);
    collection.push_back(B_readback);
    collection.push_back(C_readback);
    return collection;
  };

  {
    auto baseline = test(false, ZeroCopyTestModel::TwoMatMuls);
    auto zerocopy = test(true, ZeroCopyTestModel::TwoMatMuls);
    compare(baseline, zerocopy);
  }
  {
    auto baseline = test(false, ZeroCopyTestModel::TwoMatMulsAddOutput);
    auto zerocopy = test(true, ZeroCopyTestModel::TwoMatMulsAddOutput);
    compare(baseline, zerocopy);
  }
  {
    auto baseline = test(false, ZeroCopyTestModel::TwoMatMulsSharedWeight);
    auto zerocopy = test(true, ZeroCopyTestModel::TwoMatMulsSharedWeight);
    compare(baseline, zerocopy);
  }
}
