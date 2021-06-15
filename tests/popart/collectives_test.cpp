// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CollectivesTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/remote.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

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
    TensorId B_id = bder->customOp(Onnx::CustomOperators::ReplicatedAllReduce,
                                   1,
                                   {A_id},
                                   1,
                                   {{"op", static_cast<int>(variant)}})[0];

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

    int64_t out_shape = (N + replicationFactor - 1) / replicationFactor;
    TensorInfo B_info{"FLOAT", std::vector<int64_t>{out_shape}};
    TensorId B_id =
        bder->customOp(Onnx::CustomOperators::ReplicatedReduceScatter,
                       1,
                       {A_id},
                       1,
                       {{"op", static_cast<int>(variant)}})[0];

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

    std::vector<float> raw_B_out(replicationFactor * out_shape);
    TensorInfo B_info_replicated{
        "FLOAT", std::vector<int64_t>{replicationFactor, out_shape}};

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

  int64_t out_shape = replicationFactor * N;
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{out_shape}};
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

  std::vector<float> raw_B_out(replicationFactor * out_shape);
  TensorInfo B_info_replicated{
      "FLOAT", std::vector<int64_t>{replicationFactor, out_shape}};

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

const OperatorIdentifier OpToIdentifierMap<popart::ReplicatedAllGatherOp>::id =
    Onnx::CustomOperators::ReplicatedAllGather;

const OperatorIdentifier OpToIdentifierMap<popart::ReplicatedAllReduceOp>::id =
    Onnx::CustomOperators::ReplicatedAllReduce;

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

  int64_t out_shape = replicationFactor * N;
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{out_shape}};
  TensorId B_id =
      bder->customOp(Onnx::CustomOperators::ReplicatedAllGather,
                     1,
                     {A_id},
                     1,
                     {{sCollectiveCommGroup, std::vector<int64_t>{0, 4}}})[0];
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

  std::vector<float> raw_B_out(replicationFactor * out_shape);
  TensorInfo B_info_replicated{
      "FLOAT", std::vector<int64_t>{replicationFactor, out_shape}};

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
    BOOST_ASSERT(allGather->getGCLCommGroup().type ==
                 popart::CommGroupType::All);
    BOOST_ASSERT(allGather->getGCLCommGroup().replicaGroupSize == 4);
  }
}
