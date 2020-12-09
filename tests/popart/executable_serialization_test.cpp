// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExecutableSerializationTest

#include <boost/test/unit_test.hpp>
#include <random_util.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace popart;

void compare_tensors(const Tensor *t1,
                     const Tensor *t2,
                     bool compare_data = false) {
  BOOST_CHECK(t1->id == t2->id);
  BOOST_CHECK(t1->info == t2->info);
  BOOST_CHECK(t1->tensorLocationInfo.isSharded() ==
              t2->tensorLocationInfo.isSharded());
  BOOST_CHECK(t1->tensorLocationInfo.isRemote() ==
              t2->tensorLocationInfo.isRemote());
  BOOST_CHECK(t1->tensorLocationInfo.getRemoteBufferInfo() ==
              t2->tensorLocationInfo.getRemoteBufferInfo());

  auto nbytes = t1->info.nbytes();
  if (compare_data) {
    BOOST_CHECK(memcmp(t1->tensorData()->data(),
                       t2->tensorData()->data(),
                       nbytes) == 0);
  }
}

void compare_executables(const popx::Executablex &exe1,
                         const popx::Executablex &exe2) {

  BOOST_CHECK(exe2.getWeightTensors().size() == exe1.getWeightTensors().size());
  BOOST_CHECK(exe2.getAnchorTensors().size() == exe1.getAnchorTensors().size());
  BOOST_CHECK(exe2.getOptimizerTensors().size() ==
              exe1.getOptimizerTensors().size());
  BOOST_CHECK(exe2.getDataStreamTensors().size() ==
              exe1.getDataStreamTensors().size());

  for (int i = 0; i < exe1.getWeightTensors().size(); ++i) {
    auto t1 = exe1.getWeightTensors()[i];
    auto t2 = exe2.getTensor(t1->id);
    compare_tensors(t1, t2, true);
  }

  for (int i = 0; i < exe1.getOptimizerTensors().size(); ++i) {
    auto t1 = exe1.getOptimizerTensors()[i];
    auto t2 = exe2.getTensor(t1->id);
    compare_tensors(t1, t2, true);
  }

  for (int i = 0; i < exe1.getDataStreamTensors().size(); ++i) {
    auto t1 = exe1.getDataStreamTensors()[i];
    auto t2 = exe2.getTensor(t1->id);
    compare_tensors(t1, t2, false);
  }

  for (int i = 0; i < exe1.getAnchorTensors().size(); ++i) {
    auto t1 = exe1.getAnchorTensors()[i];
    auto t2 = exe2.getTensor(t1->id);
    compare_tensors(t1, t2, false);
  }

  BOOST_CHECK(exe2.getSeedTensor() == exe1.getSeedTensor());

  auto tileMapExe1 = exe1.lowering().getTensorTileMap();
  auto tileMapExe2 = exe2.lowering().getTensorTileMap();
  BOOST_CHECK(tileMapExe1 == tileMapExe2);
  BOOST_CHECK(exe1.lowering().getLinearlyCreatedInputTensors() ==
              exe2.lowering().getLinearlyCreatedInputTensors());
  BOOST_CHECK(exe1.lowering().getEfficientlyCreatedInputTensors() ==
              exe2.lowering().getEfficientlyCreatedInputTensors());
  BOOST_CHECK(exe1.lowering().getHostReduceStreamIds() ==
              exe2.lowering().getHostReduceStreamIds());
  BOOST_CHECK(exe1.lowering().getHostReduceStreamIds() ==
              exe2.lowering().getHostReduceStreamIds());

  const auto &cbhrsExe1 = exe1.getCollectiveBalancedHostRearrangements();
  const auto &cbhrsExe2 = exe2.getCollectiveBalancedHostRearrangements();
  BOOST_CHECK(cbhrsExe1.size() == cbhrsExe2.size());

  auto it2 = cbhrsExe2.begin();
  for (auto it1 = cbhrsExe1.begin(); it1 != cbhrsExe1.end(); ++it1) {
    BOOST_CHECK(it1->first == it2->first);
    BOOST_CHECK(it1->second.replicationFactor == it2->second.replicationFactor);
    BOOST_CHECK(it1->second.totalElementsPerReplica ==
                it2->second.totalElementsPerReplica);
    BOOST_CHECK(it1->second.gatheredToRefSlices ==
                it2->second.gatheredToRefSlices);
    ++it2;
  }
}

BOOST_AUTO_TEST_CASE(serialize_deserialize) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto device = popart::createTestDevice(TestDeviceType::Hw);

  auto opts = SessionOptions();

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

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
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  session->prepareDevice();

  const char *serializedExecutableFilePath = "temp.capnp";
  const auto &executable                   = session->getExecutable();
  {
    std::ofstream out(serializedExecutableFilePath);
    popx::serialization::serializeExecutable(out, executable);
  }

  {
    Ir ir;
    ir.setDataFlow(dataFlow);
    ir.setUserOptions(opts);
    ir.setOnnxModel(modelProto);
    std::ifstream ifs(serializedExecutableFilePath);
    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable =
        popx::serialization::deserializeExecutable(ifs, ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }
}

// Test is copied from `remotebuffer_test.cpp`.
// This test is included here to test the serialization of the
// collective balanced host rearrangements structures
BOOST_AUTO_TEST_CASE(
    serialize_deserialize_collective_balanced_host_rearrangements) {
  auto opts                                          = SessionOptions();
  opts.enableOutlining                               = false;
  opts.replicatedGraphCount                          = 2;
  opts.enableReplicatedGraphs                        = true;
  opts.weightTensorLocationSettings.location.storage = TensorStorage::OnChip;
  opts.weightTensorLocationSettings.location.replicatedTensorSharding =
      ReplicatedTensorSharding::On;
  opts.weightTensorLocationSettings.minElementsForOffChip                  = 0;
  opts.weightTensorLocationSettings.minElementsForReplicatedTensorSharding = 2;
  opts.numIOTiles = 128;

  auto R = opts.replicatedGraphCount;

  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

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
                                 {{"__execution_phase", 0}},
                                 "MatMul")[0];

  TensorId C_id = bder->customOp(Onnx::AiOnnx::OpSet9::Add,
                                 9,
                                 {E_id, D_id},
                                 1,
                                 {{"__execution_phase", 1}},
                                 "Add")[0];

  bder->addOutputTensor(C_id);

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

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

  auto device = createTestDevice(
      TestDeviceType::Hw, 2 * opts.replicatedGraphCount, 0, SyncPattern::Full);

  opts.virtualGraphMode              = VirtualGraphMode::ExecutionPhases;
  opts.explicitRecomputation         = true;
  opts.executionPhaseSettings.phases = 2;

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

  session->prepareDevice();

  const char *serializedExecutableFilePath = "temp.capnp";
  const auto &executable                   = session->getExecutable();
  {
    std::ofstream out(serializedExecutableFilePath);
    popx::serialization::serializeExecutable(out, executable);
  }

  {
    Ir ir;
    ir.setDataFlow(dataFlow);
    ir.setUserOptions(opts);
    ir.setOnnxModel(modelProto);
    std::ifstream ifs(serializedExecutableFilePath);
    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable =
        popx::serialization::deserializeExecutable(ifs, ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cacheDir = "./tmp_1" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache1";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                = SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = cachePath;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);

  std::vector<float> A_readback1_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1_init(B_info.nelms(), -99.0f);

  size_t irBundleHash1 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback1_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback1_init.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead1);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead2;
    weightsRead2.insert(A_id, {A_readback1.data(), A_info});
    weightsRead2.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead2);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  std::vector<float> A_readback2(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2(B_info.nelms(), -99.0f);

  std::vector<float> A_readback2_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2_init(B_info.nelms(), -99.0f);
  size_t irBundleHash2 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback2_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback2_init.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead1);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead2;
    weightsRead2.insert(A_id, {A_readback2.data(), A_info});
    weightsRead2.insert(B_id, {B_readback2.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead2);
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(raw_C_out.begin(),
                                raw_C_out.end(),
                                C_ground_truth.begin(),
                                C_ground_truth.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1_init.begin(),
                                A_readback1_init.end(),
                                A_readback2_init.begin(),
                                A_readback2_init.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1_init.begin(),
                                B_readback1_init.end(),
                                B_readback2_init.begin(),
                                B_readback2_init.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                A_readback1.end(),
                                A_readback2.begin(),
                                A_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                B_readback1.end(),
                                B_readback2.begin(),
                                B_readback2.end());
}

// Test is copied from `remotebuffer_test.cpp`.
// This test is included here to test the serialization of the
// collective balanced host rearrangements structures.
BOOST_AUTO_TEST_CASE(
    serialize_deserialize_collective_balanced_host_rearrangements_session_run) {
  auto opts                                          = SessionOptions();
  opts.enableOutlining                               = false;
  opts.replicatedGraphCount                          = 2;
  opts.enableReplicatedGraphs                        = true;
  opts.weightTensorLocationSettings.location.storage = TensorStorage::OnChip;
  opts.weightTensorLocationSettings.location.replicatedTensorSharding =
      ReplicatedTensorSharding::On;
  opts.weightTensorLocationSettings.minElementsForOffChip                  = 0;
  opts.weightTensorLocationSettings.minElementsForReplicatedTensorSharding = 2;
  opts.numIOTiles = 128;

  auto R = opts.replicatedGraphCount;

  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

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
                                 {{"__execution_phase", 0}},
                                 "MatMul")[0];

  TensorId C_id = bder->customOp(Onnx::AiOnnx::OpSet9::Add,
                                 9,
                                 {E_id, D_id},
                                 1,
                                 {{"__execution_phase", 1}},
                                 "Add")[0];

  bder->addOutputTensor(C_id);

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;

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

  auto dataFlow = DataFlow(batchesPerStep, {{C_id, art}});

  std::map<popart::TensorId, popart::IArray &> anchors = {{C_id, C_wrapper}};

  std::map<popart::TensorId, popart::IArray &> inputs = {};
  popart::StepIO stepio(inputs, anchors);

  auto cacheDir = "./tmp_2" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache2";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  opts.virtualGraphMode              = VirtualGraphMode::ExecutionPhases;
  opts.explicitRecomputation         = true;
  opts.executionPhaseSettings.phases = 2;
  opts.enableEngineCaching           = true;
  opts.cachePath                     = cachePath;

  // training info
  float learnRate = 0.321;

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  // R replicas doing the same work: compensate by dividing learning rate by R
  auto optimizer = ConstSGD(learnRate / R);
  std::vector<float> A_readback1(A_info.nelms(), -1.0f);
  std::vector<float> B_readback1(B_info.nelms(), -1.0f);
  std::vector<float> D_readback1(D_info.nelms(), -1.0f);
  size_t irBundleHash1 = 0;
  {
    auto device = createTestDevice(TestDeviceType::Hw,
                                   2 * opts.replicatedGraphCount,
                                   0,
                                   SyncPattern::Full);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();
    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);
    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    // to be readback:
    weightsRead.insert(A_id, {A_readback1.data(), A_info});
    weightsRead.insert(B_id, {B_readback1.data(), B_info});
    weightsRead.insert(D_id, {D_readback1.data(), D_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  std::vector<float> A_readback2(A_info.nelms(), -1.0f);
  std::vector<float> B_readback2(B_info.nelms(), -1.0f);
  std::vector<float> D_readback2(D_info.nelms(), -1.0f);

  auto C_ground_truth = raw_C_out;

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);
  size_t irBundleHash2 = 0;
  {
    auto device = createTestDevice(TestDeviceType::Hw,
                                   2 * opts.replicatedGraphCount,
                                   0,
                                   SyncPattern::Full);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);
    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    // to be readback:
    weightsRead.insert(A_id, {A_readback2.data(), A_info});
    weightsRead.insert(B_id, {B_readback2.data(), B_info});
    weightsRead.insert(D_id, {D_readback2.data(), D_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(raw_C_out.begin(),
                                raw_C_out.end(),
                                C_ground_truth.begin(),
                                C_ground_truth.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                A_readback1.end(),
                                A_readback2.begin(),
                                A_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                B_readback1.end(),
                                B_readback2.begin(),
                                B_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(D_readback1.begin(),
                                D_readback1.end(),
                                D_readback2.begin(),
                                D_readback2.end());
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_inference) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cacheDir = "./tmp_3" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache1";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                = SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = cachePath;

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);
  size_t irBundleHash1 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback1.data(), A_info});
    weightsRead.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  std::vector<float> A_readback2(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2(B_info.nelms(), -99.0f);
  size_t irBundleHash2 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback2.data(), A_info});
    weightsRead.insert(B_id, {B_readback2.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(raw_C_out.begin(),
                                raw_C_out.end(),
                                C_ground_truth.begin(),
                                C_ground_truth.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                A_readback1.end(),
                                A_readback2.begin(),
                                A_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                B_readback1.end(),
                                B_readback2.begin(),
                                B_readback2.end());
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_random_seed) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cacheDir = "./tmp_4" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache3";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                     = SessionOptions();
  opts.enableEngineCaching      = true;
  opts.cachePath                = cachePath;
  opts.enableStochasticRounding = true;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);
  size_t irBundleHash1   = 0;
  const uint64_t seedVal = 42;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    session->setRandomSeed(seedVal);
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    const Tensor *seedTensor = session->getExecutable().getSeedTensor();
    uint64_t seedValue =
        *reinterpret_cast<const uint64_t *>(seedTensor->tensorData()->data());

    BOOST_CHECK(seedValue == seedVal);

    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback1.data(), A_info});
    weightsRead.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  std::vector<float> A_readback2(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2(B_info.nelms(), -99.0f);
  size_t irBundleHash2 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    session->setRandomSeed(42);
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);
    const Tensor *seedTensor = session->getExecutable().getSeedTensor();
    uint64_t seedValue =
        *reinterpret_cast<const uint64_t *>(seedTensor->tensorData()->data());

    BOOST_CHECK(seedValue == seedVal);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback2.data(), A_info});
    weightsRead.insert(B_id, {B_readback2.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(raw_C_out.begin(),
                                raw_C_out.end(),
                                C_ground_truth.begin(),
                                C_ground_truth.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                A_readback1.end(),
                                A_readback2.begin(),
                                A_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                B_readback1.end(),
                                B_readback2.begin(),
                                B_readback2.end());
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_reset_host_weights) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cacheDir = "./tmp_5" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache1";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                = SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = cachePath;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);
  size_t irBundleHash1 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback1.data(), A_info});
    weightsRead.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  std::vector<float> A_readback2(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2(B_info.nelms(), -99.0f);
  size_t irBundleHash2 = 0;
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    session->weightsFromHost();
    session->run(stepio);
    session->resetHostWeights(proto, true);
    session->weightsFromHost();

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback2.data(), A_info});
    weightsRead.insert(B_id, {B_readback2.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);

    BOOST_CHECK_EQUAL_COLLECTIONS(v_A_init.begin(),
                                  v_A_init.end(),
                                  A_readback2.begin(),
                                  A_readback2.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_B_init.begin(),
                                  v_B_init.end(),
                                  B_readback2.begin(),
                                  B_readback2.end());
  }
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_checkpoint) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cacheDir = "./tmp_6" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto dir                    = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache1";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = dir / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                = SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = cachePath;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);

  std::vector<float> A_readback1_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1_init(B_info.nelms(), -99.0f);
  size_t irBundleHash1   = 0;
  std::string modelPath_ = "model.onnx";
  auto relModelPath_     = dir / modelPath_;
  auto modelPath         = relModelPath_.string();
  {
    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);
    session->weightsFromHost();

    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback1_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback1_init.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead1);

    session->run(stepio);
    session->modelToHost(modelPath);

    WeightsIO weightsRead;
    weightsRead.insert(A_id, {A_readback1.data(), A_info});
    weightsRead.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  BOOST_CHECK(boost::filesystem::exists(modelPath));

  std::vector<float> A_readback2_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2_init(B_info.nelms(), -99.0f);
  size_t irBundleHash2 = 0;
  {
    popart::io::confirmRegularFile(modelPath);
    std::ifstream input(modelPath, std::ios::in | std::ios::binary);
    BOOST_CHECK(input.is_open());
    std::string model((std::istreambuf_iterator<char>(input)),
                      (std::istreambuf_iterator<char>()));

    auto device = popart::createTestDevice(TestDeviceType::Hw);

    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::TrainingSession::createFromOnnxModel(
        model,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    session->weightsFromHost();
    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback2_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback2_init.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead1);

    BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                  A_readback1.end(),
                                  A_readback2_init.begin(),
                                  A_readback2_init.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                  B_readback1.end(),
                                  B_readback2_init.begin(),
                                  B_readback2_init.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_A_init.begin(),
                                  v_A_init.end(),
                                  A_readback1_init.begin(),
                                  A_readback1_init.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_B_init.begin(),
                                  v_B_init.end(),
                                  B_readback1_init.begin(),
                                  B_readback1_init.end());

    BOOST_CHECK(A_readback2_init != v_A_init);
    BOOST_CHECK(B_readback2_init != v_B_init);

    BOOST_CHECK(A_readback2_init != A_readback1_init);
    BOOST_CHECK(B_readback2_init != B_readback1_init);
  }
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_update_optimizer) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // we will generate random initializations
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C = A * B (output of network)
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId C_id = aiOnnx.matmul({A_id, B_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({C_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto device = popart::createTestDevice(TestDeviceType::Hw);

  auto cacheDir = "./tmp_7" + randomString(10);
  BOOST_CHECK(boost::filesystem::create_directory(cacheDir));
  auto d                      = boost::filesystem::path(cacheDir);
  const std::string cacheName = "session_cache1";
  auto n                      = boost::filesystem::path(cacheName);
  auto cachePath_             = d / n;
  auto cachePath              = cachePath_.string();

  boost::filesystem::remove(popart::Ir::getPopartCachePath(cachePath));
  boost::filesystem::remove(popx::IrLowering::getPoplarCachePath(cachePath));
  boost::filesystem::remove(
      popx::Executablex::getExecutablexCachePath(cachePath));

  auto opts                = SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = cachePath;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}},
                        {"defaultMomentum", {0.9, false}}});

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  std::vector<float> A_readback1(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1(B_info.nelms(), -99.0f);

  std::vector<float> A_readback1_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback1_init(B_info.nelms(), -99.0f);

  size_t irBundleHash1 = 0;
  {
    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);
  }

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(
      boost::filesystem::exists(popart::Ir::getPopartCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::IrLowering::getPoplarCachePath(cachePath)));
  BOOST_CHECK(boost::filesystem::exists(
      popx::Executablex::getExecutablexCachePath(cachePath)));

  size_t irBundleHash2 = 0;
  {
    // This session will load the PopART state and poplar
    // executable produced by the previous session.
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    BOOST_CHECK(session->getIr().hashMatched());
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    BOOST_CHECK(session->getExecutable().isDeserialized());

    float newLearningRate = 0.01f;
    float newMomentum     = 0.09f;
    auto newOptimizer = SGD({{"defaultLearningRate", {newLearningRate, false}},
                             {"defaultMomentum", {newMomentum, false}}});

    session->updateOptimizerFromHost(&newOptimizer);
    const auto &optimizerTensors =
        session->getExecutable().getOptimizerTensors();
    for (const auto &o : optimizerTensors) {
      if (boost::algorithm::icontains(o->id, "learning")) {
        float val = *reinterpret_cast<float *>(o->tensorData()->data());
        BOOST_CHECK(val == newLearningRate);
      } else if (boost::algorithm::icontains(o->id, "momentum")) {
        float val = *reinterpret_cast<float *>(o->tensorData()->data());
        BOOST_CHECK(val == newMomentum);
      }
    }
  }
}
