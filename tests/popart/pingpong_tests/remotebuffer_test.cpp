// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RemoteBufferTest

#include <../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/cache.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/region.hpp>
#include <popart/session.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace popart;

BOOST_AUTO_TEST_CASE(RemoteBufferLoadStoreTest_0) {

  int64_t N = 111;
  int64_t K = 3;

  // we will generate random initializations
  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A of shape K x N x N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  bder->customOp(Onnx::CustomOperators::CacheStore,
                 1,
                 {A_id},
                 0,
                 {{"bufferid", 0}},
                 "store");

  TensorId C_id = aiGraphcore.init({K, N, N},
                                   static_cast<int64_t>(DataType::FLOAT),
                                   static_cast<int64_t>(InitType::NoInit));

  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  TensorId B_id = bder->customOp(Onnx::CustomOperators::CacheLoad,
                                 1,
                                 {C_id},
                                 1,
                                 {{"bufferid", 0}},
                                 "load")[0];

  bder->addOutputTensor(B_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});
  auto device        = createTestDevice(TEST_TARGET);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper}};

  std::vector<float> raw_B_out(B_info.nelms());
  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(), B_info.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts    = SessionOptions();
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

    for (size_t i = 0; i < B_info.nelms(); ++i) {
      BOOST_CHECK_CLOSE(raw_B_out[i], v_A_init[i], 1e-4f);
    }
  }
}

BOOST_AUTO_TEST_CASE(RemoteBufferLoadStoreTest_1) {

  int64_t N = 4;
  int64_t K = 3;

  // we will generate random initializations
  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A of shape K x N x N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  // Tensor W of shape 1 x N x N
  TensorInfo W_info{"FLOAT", std::vector<int64_t>{1, N, N}};
  std::vector<float> v_W_init(W_info.nelms());
  for (auto &val : v_W_init) {
    val = fdis(eng);
  }
  TensorId W_id =
      bder->addInitializedInputTensor({v_W_init.data(), W_info}, "W");

  // Tensor idx0,1,2 of shape 1
  TensorInfo idx_info{"UINT32", std::vector<int64_t>{1}};
  std::vector<uint32_t> v_idx0_init(idx_info.nelms(), 0);
  std::vector<uint32_t> v_idx1_init(idx_info.nelms(), 1);
  std::vector<uint32_t> v_idx2_init(idx_info.nelms(), 2);

  ConstVoidData idx0cv = {v_idx0_init.data(),
                          {"UINT32", std::vector<int64_t>{1}}};
  ConstVoidData idx1cv = {v_idx1_init.data(),
                          {"UINT32", std::vector<int64_t>{1}}};
  ConstVoidData idx2cv = {v_idx2_init.data(),
                          {"UINT32", std::vector<int64_t>{1}}};

  TensorId idx0_id = aiOnnx.constant(idx0cv, "idx0_id");
  TensorId idx1_id = aiOnnx.constant(idx1cv, "idx1_id");
  TensorId idx2_id = aiOnnx.constant(idx2cv, "idx2_id");

  TensorId A_sub0_id =
      aiGraphcore.dynamicslice({A_id, idx0_id}, {0}, {1}, true);
  TensorId A_sub1_id =
      aiGraphcore.dynamicslice({A_id, idx1_id}, {0}, {1}, true);
  TensorId A_sub2_id =
      aiGraphcore.dynamicslice({A_id, idx2_id}, {0}, {1}, true);

  TensorId mmb0_id = aiOnnx.add({A_sub0_id, W_id});
  TensorId mmb1_id = aiOnnx.add({A_sub1_id, W_id});
  TensorId mmb2_id = aiOnnx.add({A_sub2_id, W_id});

  TensorId concat_id = aiGraphcore.init({K, N, N},
                                        static_cast<int64_t>(DataType::FLOAT),
                                        static_cast<int64_t>(InitType::NoInit));

  TensorId B_cc0_id =
      aiGraphcore.dynamicupdate({concat_id, idx0_id, mmb0_id}, {0}, {1}, true);
  TensorId B_cc1_id =
      aiGraphcore.dynamicupdate({B_cc0_id, idx1_id, mmb1_id}, {0}, {1}, true);
  TensorId B_cc2_id =
      aiGraphcore.dynamicupdate({B_cc1_id, idx2_id, mmb2_id}, {0}, {1}, true);

  bder->customOp(Onnx::CustomOperators::CacheStore,
                 1,
                 {B_cc2_id},
                 0,
                 {{"bufferid", 0}, {"__schedule_priority", 1.f}},
                 "store");

  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  TensorId B_l_id =
      bder->customOp(Onnx::CustomOperators::CacheLoad,
                     1,
                     {A_id},
                     1,
                     {{"bufferid", 0}, {"__schedule_priority", -1.f}},
                     "load")[0];

  bder->addOutputTensor(B_l_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_l_id, art}});
  auto device        = createTestDevice(TEST_TARGET);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> W_wrapper(v_W_init.data(), W_info);

  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {W_id, W_wrapper}};

  std::vector<float> raw_B_out(B_info.nelms());
  std::vector<float> groundTruth(B_info.nelms());
  popart::NDArrayWrapper<float> B_wrapper(raw_B_out.data(), B_info.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {
      {B_l_id, B_wrapper},
  };

  if (device != nullptr) {
    auto opts    = SessionOptions();
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

    for (size_t k = 0; k < K; ++k) {
      for (size_t n0 = 0; n0 < N; ++n0) {
        for (size_t n1 = 0; n1 < N; ++n1) {
          size_t i       = n1 + n0 * N + k * N * N;
          groundTruth[i] = v_A_init[i] + v_W_init[n1 + n0 * N];
          BOOST_CHECK_CLOSE(raw_B_out[i], groundTruth[i], 1e-4f);
        }
      }
    }
  }
}

// Overwrite A and B with CacheLoad, aliased as C and D
// Check if subgraph and main tensors are aliased correctly
// Before outlining:
// A -> CacheLoad -> C (C aliases A)
// B -> CacheLoad -> D (B aliases D)
// After outlining (threshold -1):
// X: A' -> CacheLoad -> C' (C' aliases A')
// A -> Call(X) -> C (C aliases A)
// B -> Call(X) -> D (B aliases D)
BOOST_AUTO_TEST_CASE(RemoteBufferLoadOutlineTest) {
  int64_t N = 111;
  int64_t K = 3;

  // we will generate random initializations
  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A of shape K x N x N
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  // Tensor B of shape K x N x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id =
      bder->addInitializedInputTensor({v_B_init.data(), B_info}, "B");

  TensorInfo C_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  TensorId C_id = bder->customOp(Onnx::CustomOperators::CacheLoad,
                                 1,
                                 {A_id},
                                 1,
                                 {{"bufferid", 0}},
                                 "load")[0];

  TensorInfo D_info{"FLOAT", std::vector<int64_t>{K, N, N}};
  TensorId D_id = bder->customOp(Onnx::CustomOperators::CacheLoad,
                                 1,
                                 {B_id},
                                 1,
                                 {{"bufferid", 0}},
                                 "load")[0];

  bder->addOutputTensor(C_id);
  bder->addOutputTensor(D_id);

  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}, {D_id, art}});
  auto device        = createTestDevice(TEST_TARGET);

  std::map<popart::TensorId, popart::IArray &> inputs = {};

  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::vector<float> raw_D_out(D_info.nelms());
  popart::NDArrayWrapper<float> D_wrapper(raw_D_out.data(), D_info.shape());
  std::map<popart::TensorId, popart::IArray &> anchors = {{D_id, D_wrapper},
                                                          {C_id, C_wrapper}};

  if (device != nullptr) {
    auto opts = SessionOptions();

    opts.outlineThreshold = -1.0;
    opts.enableOutlining  = true;

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

    auto &ir = session->getIr();

    // Check that the CallOp has inherited access types, modified and aliases
    // from the CacheLoadOp that are executed in the subgraph.
    for (Op *op : ir.getAllOps()) {
      if (CallOp *callOp = dynamic_cast<CallOp *>(op)) {
        BOOST_CHECK(callOp->aliases(0, 0).front() ==
                    view::Region::getFull(A_info.shape()));
        BOOST_CHECK(callOp->modifies(0).front() ==
                    view::Region::getFull(A_info.shape()));
        BOOST_CHECK(callOp->modifies(0).front().getAccessType() ==
                    view::AccessType::Write);
      }
      if (CacheLoadOp *cacheLoadOp = dynamic_cast<CacheLoadOp *>(op)) {
        BOOST_CHECK(cacheLoadOp
                        ->aliases(CacheLoadOp::getCachedTensorInIndex(),
                                  CacheLoadOp::getCachedTensorOutIndex())
                        .front() == view::Region::getFull(A_info.shape()));
        BOOST_CHECK(cacheLoadOp->modifies(CacheLoadOp::getCachedTensorInIndex())
                        .front() == view::Region::getFull(A_info.shape()));
        BOOST_CHECK(cacheLoadOp->modifies(CacheLoadOp::getCachedTensorInIndex())
                        .front()
                        .getAccessType() == view::AccessType::Write);
      }
    }
  }
}

// Test:
// C = matmul (A, B) + D where A, B and D are cached weight matrices,
// loss = lambda*|C|_1
static void remoteBufferPingPongWeightTestBase(SessionOptions opts,
                                               int K     = 6,
                                               int M     = 7,
                                               int N     = 8,
                                               float eps = 1e-2) {

  auto R = opts.replicatedGraphCount;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, 4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  TensorInfo A_anch_info{"FLOAT", std::vector<int64_t>{R, M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  TensorInfo B_anch_info{"FLOAT", std::vector<int64_t>{R, K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // bias matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorInfo D_anch_info{"FLOAT", std::vector<int64_t>{R, M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = fdis(eng);
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorInfo C_anch_info{"FLOAT", std::vector<int64_t>{R, M, N}};

  TensorId E_id = bder->customOp(Onnx::AiOnnx::OpSet9::MatMul,
                                 9,
                                 {A_id, B_id},
                                 1,
                                 {{"__ping_pong_phase", 0}},
                                 "MatMul")[0];

  TensorId C_id = bder->customOp(Onnx::AiOnnx::OpSet9::Add,
                                 9,
                                 {E_id, D_id},
                                 1,
                                 {{"__ping_pong_phase", 1}},
                                 "Add")[0];

  bder->addOutputTensor(C_id);

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  // compute the baseline
  std::vector<float> v_C_data(C_info.nelms());
  std::vector<float> v_C_grad(C_info.nelms());
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int index       = m * N + n;
      v_C_data[index] = 0;
      for (int k = 0; k < K; ++k) {
        v_C_data[index] += v_A_init[m * K + k] * v_B_init[k * N + n];
      }
      v_C_data[index] += v_D_init[index];
      v_C_grad[index] = 2 * (v_C_data[index] > 0) - 1;
      v_C_grad[index] *= lossLambda;
    }
  }

  // gradients of A, B, D
  // dA = dC.BT
  // dB = AT.dC
  std::vector<float> v_A_grad(A_info.nelms(), 0);
  std::vector<float> v_B_grad(B_info.nelms(), 0);
  std::vector<float> v_D_grad(D_info.nelms(), 0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int index = m * N + n;
      for (int k = 0; k < K; ++k) {
        v_A_grad[m * K + k] += v_C_grad[index] * v_B_init[k * N + n];
        v_B_grad[k * N + n] += v_C_grad[index] * v_A_init[m * K + k];
      }
      v_D_grad[index] = v_C_grad[index];
    }
  }

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep,
                           {{C_id, art},
                            {reservedGradientPrefix() + A_id, art},
                            {reservedGradientPrefix() + B_id, art},
                            {reservedGradientPrefix() + D_id, art}});

  auto device = createTestDevice(TestDeviceType::Hw,
                                 2 * opts.replicatedGraphCount,
                                 pattern = SyncPattern::Full);

  opts.virtualGraphMode      = VirtualGraphMode::PingPong;
  opts.explicitRecomputation = true;
  opts.pingPongPhases        = 2;

  // training info
  float learnRate = 0.321;

  // R replicas doing the same work: compensate by dividing learning rate by R
  auto optimizer = ConstSGD(learnRate / R);

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      l1,
      optimizer,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::Default));

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_anch_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(),
                                          C_anch_info.shape());

  // the gradient of A,
  std::vector<float> raw_A_grad_out(A_anch_info.nelms());
  popart::NDArrayWrapper<float> A_grad_wrapper(raw_A_grad_out.data(),
                                               A_anch_info.shape());
  // and the gradient of B.
  std::vector<float> raw_B_grad_out(B_anch_info.nelms());
  popart::NDArrayWrapper<float> B_grad_wrapper(raw_B_grad_out.data(),
                                               B_anch_info.shape());

  // and the gradient of D.
  std::vector<float> raw_D_grad_out(D_anch_info.nelms());
  popart::NDArrayWrapper<float> D_grad_wrapper(raw_D_grad_out.data(),
                                               D_anch_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
      {reservedGradientPrefix() + A_id, A_grad_wrapper},
      {reservedGradientPrefix() + B_id, B_grad_wrapper},
      {reservedGradientPrefix() + D_id, D_grad_wrapper}};

  session->prepareDevice();

  std::map<popart::TensorId, popart::IArray &> inputs = {};
  popart::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->run(stepio);

  // confirm the gradient values agree
  for (size_t i = 0; i < v_C_data.size(); ++i) {
    BOOST_CHECK_CLOSE(v_C_data.at(i), raw_C_out.at(i), eps);
  }

  for (size_t i = 0; i < v_A_grad.size(); ++i) {
    BOOST_CHECK_CLOSE(v_A_grad.at(i), raw_A_grad_out.at(i), eps);
  }

  for (size_t i = 0; i < v_B_grad.size(); ++i) {
    BOOST_CHECK_CLOSE(v_B_grad.at(i), raw_B_grad_out.at(i), eps);
  }

  for (size_t i = 0; i < v_D_grad.size(); ++i) {
    BOOST_CHECK_CLOSE(v_D_grad.at(i), raw_D_grad_out.at(i), eps);
  }

  // we will read the updated weights back, and check that they are correct
  std::vector<float> v_A_updated_baseline = v_A_init;
  std::vector<float> v_B_updated_baseline = v_B_init;
  std::vector<float> v_D_updated_baseline = v_D_init;
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      v_B_updated_baseline[k * N + n] -= learnRate * v_B_grad[k * N + n];
    }

    for (int m = 0; m < M; ++m) {
      v_A_updated_baseline[m * K + k] -= learnRate * v_A_grad[m * K + k];
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      v_D_updated_baseline[m * N + n] -= learnRate * v_D_grad[m * N + n];
    }
  }

  WeightsIO weightsRead;
  // to be readback:
  std::vector<float> A_readback(A_info.nelms(), -1.0f);
  std::vector<float> B_readback(B_info.nelms(), -1.0f);
  std::vector<float> D_readback(D_info.nelms(), -1.0f);
  weightsRead.insert(A_id, {A_readback.data(), A_info});
  weightsRead.insert(B_id, {B_readback.data(), B_info});
  weightsRead.insert(D_id, {D_readback.data(), D_info});

  session->weightsToHost();
  session->readWeights(weightsRead);

  for (size_t i = 0; i < v_A_updated_baseline.size(); ++i) {
    BOOST_CHECK_CLOSE(v_A_updated_baseline.at(i), A_readback.at(i), eps);
  }

  for (size_t i = 0; i < v_B_updated_baseline.size(); ++i) {
    BOOST_CHECK_CLOSE(v_B_updated_baseline.at(i), B_readback.at(i), eps);
  }

  for (size_t i = 0; i < v_D_updated_baseline.size(); ++i) {
    BOOST_CHECK_CLOSE(v_D_updated_baseline.at(i), D_readback.at(i), eps);
  }
}

// Test pingpong training
BOOST_AUTO_TEST_CASE(RemoteBufferPingPongWeightTest_0) {
  auto opts            = SessionOptions();
  opts.enableOutlining = false;
  remoteBufferPingPongWeightTestBase(opts, 8, 4, 6);
}

// Test pingpong replicated training
BOOST_AUTO_TEST_CASE(RemoteBufferPingPongWeightReplicaTest_0) {
  auto opts                     = SessionOptions();
  opts.enableOutlining          = false;
  opts.replicatedGraphCount     = 4;
  opts.enableReplicatedGraphs   = true;
  opts.replicatedWeightSharding = false;
  remoteBufferPingPongWeightTestBase(opts, 7, 3, 5);
  remoteBufferPingPongWeightTestBase(opts, 113, 103, 89, 1e-0);
  remoteBufferPingPongWeightTestBase(opts, 32, 13, 128, 1e-0);
}

// Test pingpong replicated weight sharding without I/O tiles
BOOST_AUTO_TEST_CASE(RemoteBufferPingPongWeightReplicaShardedTest_0) {
  auto opts                                   = SessionOptions();
  opts.enableOutlining                        = false;
  opts.replicatedGraphCount                   = 4;
  opts.enableReplicatedGraphs                 = true;
  opts.replicatedWeightSharding               = true;
  opts.replicatedWeightShardingMinNumElements = 0;
  opts.numIOTiles                             = 0;
  remoteBufferPingPongWeightTestBase(opts, 7, 3, 5);
  remoteBufferPingPongWeightTestBase(opts, 113, 103, 89, 1e-0);
  remoteBufferPingPongWeightTestBase(opts, 32, 13, 128, 1e-0);
}

// Test pingpong replicated weight sharding with I/O tiles
BOOST_AUTO_TEST_CASE(RemoteBufferPingPongWeightReplicaShardedTest_1) {
  auto opts                                   = SessionOptions();
  opts.enableOutlining                        = false;
  opts.replicatedGraphCount                   = 4;
  opts.enableReplicatedGraphs                 = true;
  opts.replicatedWeightSharding               = true;
  opts.replicatedWeightShardingMinNumElements = 0;
  opts.numIOTiles                             = 128;
  remoteBufferPingPongWeightTestBase(opts, 7, 3, 5);
  remoteBufferPingPongWeightTestBase(opts, 113, 103, 89, 1e-0);
  remoteBufferPingPongWeightTestBase(opts, 32, 13, 128, 1e-0);
}
