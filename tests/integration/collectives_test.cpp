// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CollectivesTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/commgroup.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/vendored/any.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

BOOST_AUTO_TEST_CASE(ReplicatedAllReduceInplaceTest) {

  const int numIPUs           = 2;
  const int replicationFactor = 2;
  int64_t N                   = 10;
  auto bder                   = Builder::create();
  auto aiOnnx                 = bder->aiOnnxOpset9();
  auto aiGraphcore            = bder->aiGraphcoreOpset1();

  // Tensor A of shape N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
  std::vector<float> v_A_init(A_info.nelms());

  TensorInfo A_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};
  std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

  int k = 0;
  for (int i = 0; i < replicationFactor; ++i) {
    for (int j = 0; j < N; ++j) {
      v_A_init_replicated[k] = (float)j;
      ++k;
    }
  }

  TensorId A_id = bder->addInputTensor(A_info, "A");

  TensorInfo B_info{"FLOAT", std::vector<int64_t>{N}};
  TensorId B_id = bder->customOp(
      Onnx::CustomOperators::ReplicatedAllReduceInplace, 1, {A_id}, 1, {})[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET, numIPUs);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                          A_info_replicated);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

  std::vector<float> raw_B_out(replicationFactor * B_info.nelms());
  TensorInfo B_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};

  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                          B_info_replicated.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts                   = SessionOptions();
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);
    session->run(stepio);

    for (int i = 0; i < N; ++i) {
      BOOST_CHECK_CLOSE(raw_B_out[i], 2 * (float)i, 1e-6f);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReplicatedAllReduceTest) {
  for (auto variant : {CollectiveOperator::Add,
                       CollectiveOperator::Mul,
                       CollectiveOperator::Max,
                       CollectiveOperator::Min}) {
    const int numIPUs           = 2;
    const int replicationFactor = 2;
    int64_t N                   = 10;
    auto bder                   = Builder::create();
    auto aiOnnx                 = bder->aiOnnxOpset9();
    auto aiGraphcore            = bder->aiGraphcoreOpset1();

    // Tensor A of shape N
    TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
    std::vector<float> v_A_init(A_info.nelms());

    TensorInfo A_info_replicated{"FLOAT",
                                 std::vector<int64_t>{replicationFactor, N}};
    std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

    int k = 0;
    for (int i = 0; i < replicationFactor; ++i) {
      for (int j = 0; j < N; ++j) {
        v_A_init_replicated[k] =
            (float)(j * replicationFactor +
                    ((j % replicationFactor == i) ? 1 : 0));
        ++k;
      }
    }

    TensorId A_id = bder->addInputTensor(A_info, "A");

    TensorInfo B_info{"FLOAT", std::vector<int64_t>{N}};
    TensorId B_id =
        bder->customOp(Onnx::CustomOperators::ReplicatedAllReduce,
                       1,
                       {A_id},
                       1,
                       {{sCollectiveOperator, static_cast<int>(variant)}})[0];

    bder->addOutputTensor(B_id);

    auto proto         = bder->getModelProto();
    auto modelProto    = io::getModelFromString(proto);
    auto art           = AnchorReturnType("All");
    int batchesPerStep = 1;
    auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
    auto device        = createTestDevice(TEST_TARGET, numIPUs);

    // inputs:
    popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                            A_info_replicated);

    std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

    std::vector<float> raw_B_out(replicationFactor * B_info.nelms());
    TensorInfo B_info_replicated{"FLOAT",
                                 std::vector<int64_t>{replicationFactor, N}};

    popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                            B_info_replicated.shape());
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {B_id, B_wrapper},
    };

    if (device != nullptr) {
      auto opts                   = SessionOptions();
      opts.enableReplicatedGraphs = true;
      opts.replicatedGraphCount   = replicationFactor;

      auto session = popart::InferenceSession::createFromOnnxModel(
          proto,
          dataFlow,
          device,
          popart::InputShapeInfo(),
          opts,
          popart::Patterns(PatternsLevel::Default));
      session->prepareDevice();
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);

      int k = 0;
      for (int i = 0; i < replicationFactor; ++i) {
        for (int j = 0; j < N; ++j) {
          switch (variant) {
          case CollectiveOperator::Add:
            BOOST_CHECK_CLOSE(raw_B_out[k], 4 * (float)j + 1, 1e-6f);
            break;
          case CollectiveOperator::Mul:
            BOOST_CHECK_CLOSE(
                raw_B_out[k], (float)((2 * j + 1) * (2 * j)), 1e-6f);
            break;
          case CollectiveOperator::Max:
            BOOST_CHECK_CLOSE(
                raw_B_out[k], (float)(std::max((2 * j + 1), (2 * j))), 1e-6f);
            break;
          case CollectiveOperator::Min:
            BOOST_CHECK_CLOSE(
                raw_B_out[k], (float)(std::min((2 * j + 1), (2 * j))), 1e-6f);
            break;
          default:
            throw error("Unsupported variant {}", variant);
          }
          ++k;
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(ReplicatedAllReduceIOTileTest) {
  const int numIPUs           = 2;
  const int replicationFactor = 2;
  int64_t N                   = 10;
  auto bder                   = Builder::create();
  auto aiOnnx                 = bder->aiOnnxOpset9();
  auto aiGraphcore            = bder->aiGraphcoreOpset1();

  // Tensor A of shape N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
  std::vector<float> v_A_init(A_info.nelms());

  TensorInfo A_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};
  std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

  int k = 0;
  for (int i = 0; i < replicationFactor; ++i) {
    for (int j = 0; j < N; ++j) {
      v_A_init_replicated[k] = (float)j;
      ++k;
    }
  }

  TensorId A_id = bder->addInputTensor(A_info, "A");

  TensorId A2_id = aiOnnx.add({A_id, A_id});

  TensorInfo B_info{"FLOAT", std::vector<int64_t>{N}};
  TensorId B_id =
      bder->customOp(Onnx::CustomOperators::ReplicatedAllReduce,
                     1,
                     {A2_id},
                     1,
                     {{sTileSetAttribute, static_cast<int>(TileSet::IO)}})[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET, numIPUs);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                          A_info_replicated);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

  std::vector<float> raw_B_out(replicationFactor * B_info.nelms());
  TensorInfo B_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};

  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                          B_info_replicated.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts                   = SessionOptions();
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;
    opts.numIOTiles             = 128;

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    auto &ir      = session->getIr();
    auto schedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

    BOOST_CHECK(schedule.size() == 3);
    BOOST_CHECK(schedule.at(1)->opid == Onnx::CustomOperators::IoTileCopy);
    BOOST_CHECK(schedule.at(2)->opid ==
                Onnx::CustomOperators::ReplicatedAllReduce);

    session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);
    session->run(stepio);

    for (int i = 0; i < N; ++i) {
      BOOST_CHECK_CLOSE(raw_B_out[i], 4 * (float)i, 1e-6f);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReplicatedReduceScatterTest) {
  for (auto variant : {CollectiveOperator::Add,
                       CollectiveOperator::Local,
                       CollectiveOperator::Mul,
                       CollectiveOperator::Max,
                       CollectiveOperator::Min}) {
    const int numIPUs           = 2;
    const int replicationFactor = 2;
    int64_t N                   = 11;
    auto bder                   = Builder::create();
    auto aiOnnx                 = bder->aiOnnxOpset9();
    auto aiGraphcore            = bder->aiGraphcoreOpset1();

    // Tensor A of shape N
    TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
    std::vector<float> v_A_init(A_info.nelms());

    TensorInfo A_info_replicated{"FLOAT",
                                 std::vector<int64_t>{replicationFactor, N}};
    std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

    int k = 0;
    for (int i = 0; i < replicationFactor; ++i) {
      for (int j = 0; j < N; ++j) {
        v_A_init_replicated[k] =
            (float)(j * replicationFactor +
                    ((j % replicationFactor == i) ? 1 : 0));
        ++k;
      }
    }

    TensorId A_id = bder->addInputTensor(A_info, "A");

    int64_t outShape = (N + replicationFactor - 1) / replicationFactor;
    TensorInfo B_info{"FLOAT", std::vector<int64_t>{outShape}};
    TensorId B_id =
        bder->customOp(Onnx::CustomOperators::ReplicatedReduceScatter,
                       1,
                       {A_id},
                       1,
                       {{sCollectiveOperator, static_cast<int>(variant)}})[0];

    bder->addOutputTensor(B_id);

    auto proto         = bder->getModelProto();
    auto modelProto    = io::getModelFromString(proto);
    auto art           = AnchorReturnType("All");
    int batchesPerStep = 1;
    auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
    auto device        = createTestDevice(TEST_TARGET, numIPUs);

    // inputs:
    popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                            A_info_replicated);

    std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

    std::vector<float> raw_B_out(replicationFactor * outShape);
    TensorInfo B_info_replicated{
        "FLOAT", std::vector<int64_t>{replicationFactor, outShape}};

    popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                            B_info_replicated.shape());
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {B_id, B_wrapper},
    };

    if (device != nullptr) {
      auto opts                   = SessionOptions();
      opts.enableReplicatedGraphs = true;
      opts.replicatedGraphCount   = replicationFactor;

      auto session = popart::InferenceSession::createFromOnnxModel(
          proto,
          dataFlow,
          device,
          popart::InputShapeInfo(),
          opts,
          popart::Patterns(PatternsLevel::Default));
      session->prepareDevice();
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);

      int k = 0;
      for (int i = 0; i < replicationFactor; ++i) {
        for (int j = 0; j < (N - 1) / 2 + 1; ++j) {
          if (k >= N) {
            // Zero padded element at the end
            BOOST_CHECK_CLOSE(raw_B_out.back(), 0.0f, 1e-6f);
          } else {
            switch (variant) {
            case CollectiveOperator::Add:
              BOOST_CHECK_CLOSE(raw_B_out[k], 4 * (float)k + 1, 1e-6f);
              break;
            case CollectiveOperator::Local:
              BOOST_CHECK_CLOSE(raw_B_out[k],
                                (float)(k * replicationFactor +
                                        ((k % replicationFactor == i) ? 1 : 0)),
                                1e-6f);
              break;
            case CollectiveOperator::Mul:
              BOOST_CHECK_CLOSE(
                  raw_B_out[k], (float)((2 * k + 1) * (2 * k)), 1e-6f);
              break;
            case CollectiveOperator::Max:
              BOOST_CHECK_CLOSE(
                  raw_B_out[k], (float)(std::max((2 * k + 1), (2 * k))), 1e-6f);
              break;
            case CollectiveOperator::Min:
              BOOST_CHECK_CLOSE(
                  raw_B_out[k], (float)(std::min((2 * k + 1), (2 * k))), 1e-6f);
              break;
            default:
              throw error("Unsupported variant {}", variant);
            }
            ++k;
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(ReplicatedAllGatherTest) {

  const int numIPUs           = 2;
  const int replicationFactor = 2;
  int64_t N                   = 11;
  auto bder                   = Builder::create();
  auto aiOnnx                 = bder->aiOnnxOpset9();
  auto aiGraphcore            = bder->aiGraphcoreOpset1();

  // Tensor A of shape N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
  std::vector<float> v_A_init(A_info.nelms());

  TensorInfo A_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};
  std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

  int k = 0;
  for (int i = 0; i < replicationFactor; ++i) {
    for (int j = 0; j < N; ++j) {
      v_A_init_replicated[k] = i * N + (float)j;
      ++k;
    }
  }

  TensorId A_id = bder->addInputTensor(A_info, "A");

  int64_t outShape = replicationFactor * N;
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{outShape}};
  TensorId B_id = bder->customOp(
      Onnx::CustomOperators::ReplicatedAllGather, 1, {A_id}, 1, {})[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET, numIPUs);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                          A_info_replicated);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

  std::vector<float> raw_B_out(replicationFactor * outShape);
  TensorInfo B_info_replicated{
      "FLOAT", std::vector<int64_t>{replicationFactor, outShape}};

  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                          B_info_replicated.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts                   = SessionOptions();
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);
    session->run(stepio);

    for (int r = 0; r < replicationFactor; ++r) {
      for (int i = 0; i < replicationFactor; ++i) {
        for (int j = 0; j < N; ++j) {
          BOOST_CHECK_CLOSE(raw_B_out[(r * replicationFactor + i) * N + j],
                            i * N + (float)j,
                            1e-6f);
        }
      }
    }
  }
}

template <typename OpTy> struct OpToIdentifierMap {};

template <> struct OpToIdentifierMap<popart::ReplicatedAllGatherOp> {
  static const OperatorIdentifier id;
};

template <> struct OpToIdentifierMap<popart::ReplicatedAllReduceOp> {
  static const OperatorIdentifier id;
};

template <> struct OpToIdentifierMap<popart::ReplicatedAllReduceInplaceOp> {
  static const OperatorIdentifier id;
};

template <> struct OpToIdentifierMap<popart::ReplicatedReduceScatterOp> {
  static const OperatorIdentifier id;
};

const OperatorIdentifier OpToIdentifierMap<popart::ReplicatedAllGatherOp>::id =
    Onnx::CustomOperators::ReplicatedAllGather;

const OperatorIdentifier OpToIdentifierMap<popart::ReplicatedAllReduceOp>::id =
    Onnx::CustomOperators::ReplicatedAllReduce;

const OperatorIdentifier
    OpToIdentifierMap<popart::ReplicatedAllReduceInplaceOp>::id =
        Onnx::CustomOperators::ReplicatedAllReduceInplace;

const OperatorIdentifier
    OpToIdentifierMap<popart::ReplicatedReduceScatterOp>::id =
        Onnx::CustomOperators::ReplicatedReduceScatter;

template <typename OpTy>
static std::vector<const OpTy *>
findAllOps(const std::unique_ptr<popart::Session> &session) {
  const auto allOps = session->getIr().getAllOps();
  std::vector<const OpTy *> result;

  std::copy_if(
      allOps.cbegin(),
      allOps.cend(),
      std::back_inserter(result),
      [](const Op *op) { return op->opid == OpToIdentifierMap<OpTy>::id; });
  return result;
}

template <typename OpTy, typename SessionTy>
static const OpTy *findFirstOp(const std::unique_ptr<SessionTy> &session) {
  const auto allOps = session->getIr().getAllOps();
  auto iter = std::find_if(allOps.cbegin(), allOps.cend(), [](const Op *op) {
    return op->opid == OpToIdentifierMap<OpTy>::id;
  });
  BOOST_ASSERT(iter != allOps.cend());
  return static_cast<const OpTy *>(*iter);
}

BOOST_AUTO_TEST_CASE(ReplicatedAllGatherTest_CommGroup_All) {
  const int numIPUs           = 2;
  const int replicationFactor = 2;
  int64_t N                   = 11;
  auto bder                   = Builder::create();
  auto aiOnnx                 = bder->aiOnnxOpset9();
  auto aiGraphcore            = bder->aiGraphcoreOpset1();

  // Tensor A of shape N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
  std::vector<float> v_A_init(A_info.nelms());

  TensorInfo A_info_replicated{"FLOAT",
                               std::vector<int64_t>{replicationFactor, N}};
  std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

  int k = 0;
  for (int i = 0; i < replicationFactor; ++i) {
    for (int j = 0; j < N; ++j) {
      v_A_init_replicated[k] = i * N + (float)j;
      ++k;
    }
  }

  TensorId A_id = bder->addInputTensor(A_info, "A");

  int64_t outShape = replicationFactor * N;
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{outShape}};
  TensorId B_id =
      bder->customOp(Onnx::CustomOperators::ReplicatedAllGather,
                     1,
                     {A_id},
                     1,
                     {{sCollectiveCommGroup, std::vector<int64_t>{0, 2}}})[0];
  TensorId B2_id =
      bder->customOp(Onnx::CustomOperators::ReplicatedAllReduce,
                     1,
                     {A_id},
                     1,
                     {{sCollectiveCommGroup, std::vector<int64_t>{1, 42}}})[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET, numIPUs);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                          A_info_replicated);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

  std::vector<float> raw_B_out(replicationFactor * outShape);
  TensorInfo B_info_replicated{
      "FLOAT", std::vector<int64_t>{replicationFactor, outShape}};

  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                          B_info_replicated.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts                   = SessionOptions();
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);
    session->run(stepio);

    const popart::ReplicatedAllGatherOp *allGather =
        findFirstOp<popart::ReplicatedAllGatherOp>(session);
    BOOST_TEST(allGather->getGCLCommGroup().type == popart::CommGroupType::All);
    BOOST_TEST(allGather->getGCLCommGroup().replicaGroupSize == 2);
  }
}

// Test if ReplicatedReduceScatter -> ReplicatedAllReduce(Inplace) produces the
// correct output when running the two Ops across orthogonal CommGroups
BOOST_AUTO_TEST_CASE(ReplicatedScatterAndReduceCommGroupTest) {
  int rOffset = 10000;

  for (int testId = 0; testId < 4; ++testId) {
    bool useInplace         = testId % 2 != 0;
    auto collectiveOperator = ((testId / 2) % 2) == 0
                                  ? CollectiveOperator::Add
                                  : CollectiveOperator::Local;

    logging::info("[ReplicatedScatterAndReduceCommGroupTest] Use inplace: {}, "
                  "collectiveOperator: {}",
                  useInplace,
                  collectiveOperator);

    const int numIPUs           = 4;
    const int replicationFactor = 4;
    const int groupSize         = 2;
    int64_t N                   = 4;
    auto bder                   = Builder::create();
    auto aiOnnx                 = bder->aiOnnxOpset9();
    auto aiGraphcore            = bder->aiGraphcoreOpset1();

    // Tensor A of shape N
    TensorInfo A_info{"FLOAT", std::vector<int64_t>{N}};
    std::vector<float> v_A_init(A_info.nelms());

    TensorInfo A_info_replicated{"FLOAT",
                                 std::vector<int64_t>{replicationFactor, N}};
    std::vector<float> v_A_init_replicated(replicationFactor * A_info.nelms());

    int k = 0;
    for (int i = 0; i < replicationFactor; ++i) {
      for (int j = 0; j < N; ++j) {
        // Fill all replicas with the same base values (j) and add the
        // replica index (i) offset to it so it's easy to analyze which values
        // got summed up.
        v_A_init_replicated[k] = i * rOffset + (float)j;
        ++k;
      }
    }

    TensorId A_id = bder->addInputTensor(A_info, "A");

    int64_t outShape = N / groupSize;

    TensorId B_id = bder->customOp(
        Onnx::CustomOperators::ReplicatedReduceScatter,
        1,
        {A_id},
        1,
        {{sCollectiveCommGroup, std::vector<int64_t>{1, groupSize}},
         {sCollectiveOperator, static_cast<int64_t>(collectiveOperator)}})[0];
    TensorId C_id = bder->customOp(
        useInplace ? Onnx::CustomOperators::ReplicatedAllReduceInplace
                   : Onnx::CustomOperators::ReplicatedAllReduce,
        1,
        {B_id},
        1,
        {{sCollectiveCommGroup,
          std::vector<int64_t>{2, replicationFactor / groupSize}}})[0];

    bder->addOutputTensor(B_id);
    bder->addOutputTensor(C_id);

    auto proto         = bder->getModelProto();
    auto modelProto    = io::getModelFromString(proto);
    auto art           = AnchorReturnType("All");
    int batchesPerStep = 1;
    auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}, {C_id, art}});
    auto device        = createTestDevice(TEST_TARGET, numIPUs);

    // inputs:
    popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                            A_info_replicated);

    std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

    std::vector<float> raw_B_out(replicationFactor * outShape);
    TensorInfo B_info_replicated{
        "FLOAT", std::vector<int64_t>{replicationFactor, outShape}};

    popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(),
                                            B_info_replicated.shape());

    std::vector<float> raw_C_out(replicationFactor * outShape);
    TensorInfo C_info_replicated{
        "FLOAT", std::vector<int64_t>{replicationFactor, outShape}};

    popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(),
                                            C_info_replicated.shape());

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {B_id, B_wrapper},
        {C_id, C_wrapper},
    };

    if (device != nullptr) {
      auto opts                   = SessionOptions();
      opts.enableReplicatedGraphs = true;
      opts.replicatedGraphCount   = replicationFactor;

      auto session = popart::InferenceSession::createFromOnnxModel(
          proto,
          dataFlow,
          device,
          popart::InputShapeInfo(),
          opts,
          popart::Patterns(PatternsLevel::Default));
      session->prepareDevice();
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);

      // Check operations have correct attributes set
      const popart::ReplicatedReduceScatterOp *reduceScatter =
          findFirstOp<popart::ReplicatedReduceScatterOp>(session);
      BOOST_CHECK(reduceScatter->getGCLCommGroup().type ==
                  popart::CommGroupType::Consecutive);
      BOOST_CHECK(reduceScatter->getGCLCommGroup().replicaGroupSize ==
                  groupSize);

      if (useInplace) {
        const popart::ReplicatedAllReduceInplaceOp *allReduce =
            findFirstOp<popart::ReplicatedAllReduceInplaceOp>(session);
        BOOST_CHECK(allReduce->getGCLCommGroup().type ==
                    popart::CommGroupType::Orthogonal);
        BOOST_CHECK(allReduce->getGCLCommGroup().replicaGroupSize ==
                    replicationFactor / groupSize);
      } else {
        const popart::ReplicatedAllReduceOp *allReduce =
            findFirstOp<popart::ReplicatedAllReduceOp>(session);
        BOOST_CHECK(allReduce->getGCLCommGroup().type ==
                    popart::CommGroupType::Orthogonal);
        BOOST_CHECK(allReduce->getGCLCommGroup().replicaGroupSize ==
                    replicationFactor / groupSize);
      }

      // Prefix sum of all replicas indices * replica index offsets
      auto sum = 0.f;
      for (int r = 0; r < replicationFactor; ++r) {
        sum += r * rOffset;
      }

      for (int r = 0; r < replicationFactor / groupSize; ++r) {
        // Lower and upper replica index expected in the result value
        float lower = r * groupSize;
        float upper = (r + 1) * groupSize - 1;
        for (int j = 0; j < groupSize * outShape; ++j) {
          if (collectiveOperator == CollectiveOperator::Add) {
            // Check ReduceScatter result
            if (!useInplace) {
              BOOST_CHECK_CLOSE(raw_B_out[r * (groupSize * outShape) + j],
                                groupSize * (upper + lower) / 2.f * rOffset +
                                    j * groupSize,
                                1e-6f);
            }

            // Check AllReduce result
            BOOST_CHECK_CLOSE(raw_C_out[r * (groupSize * outShape) + j],
                              sum + j * replicationFactor,
                              1e-6f);
          }
          if (collectiveOperator == CollectiveOperator::Local) {
            // Check ReduceScatter result
            if (!useInplace) {
              BOOST_CHECK_CLOSE(raw_B_out[r * (groupSize * outShape) + j],
                                (j / outShape + r * groupSize) * rOffset + j,
                                1e-6f);
            }

            // Check AllReduce result
            BOOST_CHECK_CLOSE(raw_C_out[r * (groupSize * outShape) + j],
                              (j / outShape + (j / outShape) + groupSize) *
                                      rOffset +
                                  j * replicationFactor / groupSize,
                              1e-6f);
          }
        }
      }
    }
  }
}
