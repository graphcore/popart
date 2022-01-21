// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExecutableSerializationTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <random_util.hpp>
#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
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

popx::serialization::Reader createReader(const std::string &executablePath) {
  auto ifs =
      std::make_shared<std::ifstream>(executablePath, std::fstream::binary);
  return popx::serialization::Reader(ifs);
}

std::string createDirForTest() {
  auto executableDir = "./tmp_1" + randomString(10);
  boost::filesystem::remove(executableDir);
  BOOST_CHECK(boost::filesystem::create_directories(executableDir));

  return executableDir;
}

std::string addPaths(const std::string &lhs, const std::string &rhs) {
  auto dirPath = boost::filesystem::path(lhs);
  auto dstPath = dirPath / rhs;
  return dstPath.string();
}

std::string getCachePath(const std::string &testDir) {
  const std::string cacheDir = "session_cache1";
  return addPaths(testDir, cacheDir);
}

std::string getExecutablePath(const std::string &testDir) {
  const std::string executableName = "executable.popef";
  return addPaths(testDir, executableName);
}

std::string getModelPath(const std::string &testDir) {
  auto dirPath                = boost::filesystem::path(testDir);
  const std::string modelName = "model.onnx";
  return addPaths(testDir, modelName);
}

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

  BOOST_CHECK(exe1.lowering().getLinearlyCreatedInputTensors() ==
              exe2.lowering().getLinearlyCreatedInputTensors());
  BOOST_CHECK(exe1.lowering().getEfficientlyCreatedInputTensors() ==
              exe2.lowering().getEfficientlyCreatedInputTensors());

  const auto &cbrIdsExe1 = exe1.getCollectiveBalancedHostRearrangementIds();
  const auto &cbrIdsExe2 = exe2.getCollectiveBalancedHostRearrangementIds();
  BOOST_CHECK(cbrIdsExe1 == cbrIdsExe2);

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

std::unique_ptr<Builder> getBuiltModel() {
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
  bder->addOutputTensor(l1);

  return bder;
}

BOOST_AUTO_TEST_CASE(serialize_deserialize) {
  auto bder       = getBuiltModel();
  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto l1         = bder->getOutputTensorIds()[0];
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{l1, art}});

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

  session->prepareDevice();

  const std::string testDir        = createDirForTest();
  const std::string executablePath = getExecutablePath(testDir);
  const auto &executable           = session->getExecutable();
  {
    std::ofstream out(executablePath);
    popx::serialization::serializeEngineExecutable(
        out, nullptr, &executable, 0);
  }

  {
    Ir ir;
    ir.setDataFlow(dataFlow);
    ir.setUserOptions(opts);
    ir.setOnnxModel(modelProto);
    popx::serialization::Reader reader(createReader(executablePath));
    BOOST_CHECK(reader.containsExecutable());
    BOOST_CHECK(!reader.containsPoplarExecutable());

    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable = reader.deserializeExecutable(ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

// T36910 identified that the Accum tensor was getting saved in the executable.
// This is not needed in the current implementation. This test sets the above
// but with an Adam optimizer and gradient accumulation to ensure the creation
// of the Accum___ tensors.
BOOST_AUTO_TEST_CASE(serialize_deserialize_adam) {
  auto bder       = getBuiltModel();
  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto l1         = bder->getOutputTensorIds()[0];
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{l1, art}});

  auto device = popart::createTestDevice(TestDeviceType::Hw);

  auto opts                       = SessionOptions();
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 10;

  // training info
  auto optimizer = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::AdaMax,
      WeightDecayMode::L2Regularization,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

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

  const std::string testDir        = createDirForTest();
  const std::string executablePath = getExecutablePath(testDir);
  const auto &executable           = session->getExecutable();
  {
    std::ofstream out(executablePath);
    popx::serialization::serializeEngineExecutable(
        out, nullptr, &executable, 0);
  }

  {
    Ir ir;
    ir.setDataFlow(dataFlow);
    ir.setUserOptions(opts);
    ir.setOnnxModel(modelProto);

    popx::serialization::Reader reader(createReader(executablePath));
    BOOST_CHECK(reader.containsExecutable());
    BOOST_CHECK(!reader.containsPoplarExecutable());

    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable = reader.deserializeExecutable(ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

BOOST_AUTO_TEST_CASE(serialize_deserialize_adam_pre_prepared_ir) {
  auto bder       = getBuiltModel();
  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto l1         = bder->getOutputTensorIds()[0];
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{l1, art}});

  auto device = popart::createTestDevice(TestDeviceType::Hw);

  auto opts                       = SessionOptions();
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 10;

  // training info
  auto optimizer = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::AdaMax,
      WeightDecayMode::L2Regularization,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

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

  const std::string testDir        = createDirForTest();
  const std::string executablePath = getExecutablePath(testDir);
  const auto &executable           = session->getExecutable();
  {
    std::ofstream out(executablePath);
    popx::serialization::serializeEngineExecutable(
        out, nullptr, &executable, 0);
  }

  {
    Ir ir;
    ir.prepare({io::getModelFromString(proto),
                popart::InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                opts,
                popart::Patterns(PatternsLevel::Default)});

    popx::serialization::Reader reader(createReader(executablePath));
    BOOST_CHECK(reader.containsExecutable());
    BOOST_CHECK(!reader.containsPoplarExecutable());

    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable = reader.deserializeExecutable(ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

BOOST_AUTO_TEST_CASE(
    serialize_deserialize_adam_pre_prepared_ir_different_optimizer_state) {
  auto bder       = getBuiltModel();
  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto l1         = bder->getOutputTensorIds()[0];
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{l1, art}});

  auto device = popart::createTestDevice(TestDeviceType::Hw);

  auto opts                       = SessionOptions();
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 10;

  // training info
  auto optimizer = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::AdaMax,
      WeightDecayMode::L2Regularization,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

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

  const std::string testDir        = createDirForTest();
  const std::string executablePath = getExecutablePath(testDir);
  const auto &executable           = session->getExecutable();
  {
    std::ofstream out(executablePath);
    popx::serialization::serializeEngineExecutable(
        out, nullptr, &executable, 0);
  }

  {
    auto newOptimizer = SGD({{"defaultLearningRate", {0.01, false}}});

    Ir ir;
    ir.prepare({io::getModelFromString(proto),
                popart::InputShapeInfo(),
                dataFlow,
                l1,
                &newOptimizer,
                *device,
                opts,
                popart::Patterns(PatternsLevel::Default)});

    popx::serialization::Reader reader(createReader(executablePath));
    BOOST_CHECK(reader.containsExecutable());
    BOOST_CHECK(!reader.containsPoplarExecutable());

    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);

    // Ir passed by reference to 'deserializeExecutable' has different
    // 'additionalModelProtoTensors' to Ir used to serialize executable.
    BOOST_CHECK_THROW(reader.deserializeExecutable(ir, ir_lowering),
                      popart::error);
  }

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir        = createDirForTest();
  const std::string executablePath = getExecutablePath(testDir);
  const auto &executable           = session->getExecutable();
  {
    std::ofstream out(executablePath);
    popx::serialization::serializeEngineExecutable(
        out, nullptr, &executable, 0);
  }

  {
    Ir ir;
    ir.setDataFlow(dataFlow);
    ir.setUserOptions(opts);
    ir.setOnnxModel(modelProto);
    popx::serialization::Reader reader(createReader(executablePath));
    BOOST_CHECK(reader.containsExecutable());
    BOOST_CHECK(!reader.containsPoplarExecutable());
    bool skipGraphCompilation = true;
    popx::IrLowering ir_lowering(ir, device, skipGraphCompilation);
    auto deserializedExecutable = reader.deserializeExecutable(ir, ir_lowering);
    compare_executables(executable, *deserializedExecutable);
  }

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

void testSessionRunFromSerializedExe(bool useCache) {
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

  auto opts = SessionOptions();
  if (useCache) {
    opts.enableEngineCaching = true;
    opts.cachePath           = cachePath;
  }

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

  size_t irBundleHash1       = 0;
  std::string executablePath = useCache ? "" : getExecutablePath(testDir);
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

    if (!useCache) {
      session->compileAndExport(executablePath);
      session->loadEngineAndConnectStreams();
    } else {
      executablePath = session->getExecutable().getCachePath(cachePath);
    }

    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback1_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback1_init.data(), B_info});

    session->weightsFromHost();
    session->weightsToHost();
    session->readWeights(weightsRead1);

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

  BOOST_CHECK(boost::filesystem::exists(executablePath));

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
    if (useCache) {
      session->prepareDevice();
    } else {
      session->loadExecutableFromFile(executablePath);
      session->getDevice().prepare();
      session->loadEngineAndConnectStreams();
    }
    irBundleHash2 = session->getIr().getIrBundleHash();
    BOOST_CHECK(irBundleHash1 == irBundleHash2);

    if (useCache) {
      BOOST_CHECK(session->getIr().hashMatched());
      BOOST_CHECK(session->getIrLowering().usingCachedExecutable());
    }
    BOOST_CHECK(session->getExecutable().isDeserialized());

    WeightsIO weightsRead1;
    weightsRead1.insert(A_id, {A_readback2_init.data(), A_info});
    weightsRead1.insert(B_id, {B_readback2_init.data(), B_info});

    session->weightsFromHost();
    session->weightsToHost();
    session->readWeights(weightsRead1);

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

BOOST_AUTO_TEST_CASE(
    session_run_from_serialized_exe_using_save_load_functionality) {
  testSessionRunFromSerializedExe(false);
}

BOOST_AUTO_TEST_CASE(session_run_from_serialized_exe_using_cache) {
  testSessionRunFromSerializedExe(true);
}

BOOST_AUTO_TEST_CASE(session_run_on_ipu_from_offlineipu_serialized_exe) {
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

  auto opts                = SessionOptions();
  opts.enableEngineCaching = false;
  opts.cachePath           = cachePath;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});

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

  auto devManager = popart::DeviceManager::createDeviceManager();
  devManager.setOnDemandAttachTimeout(900);
  std::vector<std::shared_ptr<popart::DeviceInfo>> devices;

  // NOTE: We work out the devices we will use ahead of time so we don't have
  // to rely on the mechanism that the second device will be different from the
  // first device because we're still attached to the first device, as this can
  // contribute to deadlocks.
  devManager.enumerate(devices,
                       1,
                       SyncPattern::Full,
                       DeviceType::Ipu,
                       DeviceConnectionType::OnDemand,
                       0);

  BOOST_REQUIRE(devices.size() >= 2);

  int initialDeviceId = 0;
  std::string initialDeviceArchString;

  {

    auto initialDevice =
        devManager.tryAcquireDeviceById(devices.at(0)->getId(),
                                        SyncPattern::Full,
                                        DeviceConnectionType::OnDemand);
    BOOST_REQUIRE(initialDevice);
    initialDeviceId         = initialDevice->getId();
    initialDeviceArchString = initialDevice->getTarget().getTargetArchString();

    // Engine caching is enabled so this session will store
    // the serialized PopART state and poplar executable
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        initialDevice,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);

    session->weightsFromHost();

    session->run(stepio);

    WeightsIO weightsRead2;
    weightsRead2.insert(A_id, {A_readback1.data(), A_info});
    weightsRead2.insert(B_id, {B_readback1.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead2);
  }

  auto C_ground_truth = raw_C_out;

  opts.enableEngineCaching = true;
  size_t irBundleHash1     = 0;
  std::string cacheFile;
  {
    auto device =
        popart::createTestDevice(TestDeviceType::OfflineIpu,
                                 1,
                                 0,
                                 SyncPattern::Full,
                                 {{"ipuVersion", initialDeviceArchString}});

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
    session->prepareDevice(false);
    irBundleHash1 = session->getIr().getIrBundleHash();

    BOOST_CHECK_THROW(session->run(stepio), popart::error);

    BOOST_CHECK(session->getExecutable().isDeserialized() == false);
    BOOST_CHECK(session->getIrLowering().usingCachedExecutable() == false);
    BOOST_CHECK(session->getIr().hashMatched() == false);
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

  std::vector<float> A_readback2(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2(B_info.nelms(), -99.0f);

  std::vector<float> A_readback2_init(A_info.nelms(), -9.0f);
  std::vector<float> B_readback2_init(B_info.nelms(), -99.0f);
  size_t irBundleHash2     = 0;
  int deserializedDeviceId = -1;
  {
    // Run on a different device to the initialDevice. This forces data to be
    // copied instead of relying on unchanged memory on the device.
    auto device =
        devManager.tryAcquireDeviceById(devices.at(1)->getId(),
                                        SyncPattern::Full,
                                        DeviceConnectionType::OnDemand);
    BOOST_REQUIRE(device);
    deserializedDeviceId = device->getId();
    BOOST_REQUIRE_NE(initialDeviceId, deserializedDeviceId);

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

  BOOST_CHECK_EQUAL_COLLECTIONS(A_readback1.begin(),
                                A_readback1.end(),
                                A_readback2.begin(),
                                A_readback2.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_readback1.begin(),
                                B_readback1.end(),
                                B_readback2.begin(),
                                B_readback2.end());

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

  opts.virtualGraphMode              = VirtualGraphMode::ExecutionPhases;
  opts.explicitRecomputation         = true;
  opts.executionPhaseSettings.phases = 2;
  opts.enableEngineCaching           = true;
  opts.cachePath                     = cachePath;

  // training info
  float learnRate = 0.321;

  // R replicas doing the same work: compensate by dividing learning rate by R
  auto optimizer = ConstSGD(learnRate / R);
  std::vector<float> A_readback1(A_info.nelms(), -1.0f);
  std::vector<float> B_readback1(B_info.nelms(), -1.0f);
  std::vector<float> D_readback1(D_info.nelms(), -1.0f);
  size_t irBundleHash1 = 0;
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  std::vector<float> A_readback2(A_info.nelms(), -1.0f);
  std::vector<float> B_readback2(B_info.nelms(), -1.0f);
  std::vector<float> D_readback2(D_info.nelms(), -1.0f);

  auto C_ground_truth = raw_C_out;

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

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
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

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
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

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
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

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
  size_t irBundleHash1        = 0;
  const std::string modelPath = getModelPath(testDir);
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  auto C_ground_truth = raw_C_out;

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
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

  const std::string testDir   = createDirForTest();
  const std::string cachePath = getCachePath(testDir);

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
  std::string cacheFile;
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
    cacheFile = session->getExecutable().getCachePath(cachePath);
  }

  // reset output values
  std::fill(raw_C_out.begin(), raw_C_out.end(), -9.0f);

  BOOST_CHECK(boost::filesystem::exists(cacheFile));

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

  BOOST_CHECK(boost::filesystem::remove_all(testDir));
}

BOOST_AUTO_TEST_CASE(optimizer_hash_tests) {

  auto sgd1 = SGD({{"defaultLearningRate", {0.01, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.1, false}},
                   {"defaultMomentum", {0.9, false}}});

  auto sgd2 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.8, false}}});

  // Changing parameter values should result in the same hash if nonConst.
  BOOST_CHECK(sgd1.hash() == sgd2.hash());

  auto sgd3 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.0, false}}});

  // Momentum is non const which means that the momentum tensor is still added
  // to the graph
  BOOST_CHECK(sgd3.hash() == sgd1.hash());

  auto sgd4 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.0, true}}});

  // Momentum is disabled
  BOOST_CHECK(sgd4.hash() != sgd1.hash());

  auto sgd5 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.0, true}}});
  sgd5.insertSpecific("foo", {{"momentum", {0.1, true}}});

  // Momentum is disabled, but we added a specific momentum tensor
  BOOST_CHECK(sgd5.hash() != sgd4.hash());
  BOOST_CHECK(sgd5.hash() != sgd1.hash());

  auto sgd6 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.6, true}}});
  sgd6.insertSpecific("foo", {{"momentum", {0.1, true}}});

  // Momentum enabled, but we added a specific momentum tensor
  BOOST_CHECK(sgd6.hash() != sgd1.hash());

  auto sgd7 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.6, true}}});

  // Momentum enabled, but we changed constness
  BOOST_CHECK(sgd7.hash() != sgd1.hash());

  auto sgd8 = SGD({{"defaultLearningRate", {0.02, false}},
                   {"defaultDampening", {0.1, true}},
                   {"defaultWeightDecay", {0.2, false}},
                   {"defaultMomentum", {0.6, false}}});

  sgd8.insertSpecific("foo", {{"learningRate", {0.1, true}}});
  // Added a specific learning rate
  BOOST_CHECK(sgd8.hash() != sgd1.hash());

  auto sgd9 = SGD({{"defaultLearningRate", {0.01, false}},
                   {"defaultDampening", {0.2, true}},
                   {"defaultWeightDecay", {0.1, false}},
                   {"defaultMomentum", {0.9, false}}});
  // Changed a const parameter.
  BOOST_CHECK(sgd9.hash() != sgd1.hash());

  auto adam1 = Adam(
      {
          {"defaultLearningRate", {0.01, false}},
          {"defaultWeightDecay", {0.1, false}},
          {"defaultBeta1", {0.1, false}},
          {"defaultBeta2", {0.1, false}},
          {"defaultEps", {0.1, false}},
          {"lossScaling", {0.1, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

  auto adam2 = Adam(
      {
          {"defaultLearningRate", {0.01, false}},
          {"defaultWeightDecay", {0.1, false}},
          {"defaultBeta1", {0.1, false}},
          {"defaultBeta2", {0.1, false}},
          {"defaultEps", {0.1, true}},
          {"lossScaling", {0.1, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

  // changed constness of defaultEps
  BOOST_CHECK(adam1.hash() != adam2.hash());

  auto adam3 = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

  // changing parameter values should have no impact
  BOOST_CHECK(adam3.hash() == adam1.hash());

  auto adam4 = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::Decay,
      DataType::FLOAT16,
      DataType::FLOAT,
      DataType::FLOAT);

  // changed data type
  BOOST_CHECK(adam4.hash() != adam1.hash());

  auto adam5 = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::L2Regularization,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

  // changed weightDecayMode
  BOOST_CHECK(adam5.hash() != adam1.hash());

  auto adam6 = Adam(
      {
          {"defaultLearningRate", {0.02, false}},
          {"defaultWeightDecay", {0.2, false}},
          {"defaultBeta1", {0.2, false}},
          {"defaultBeta2", {0.2, false}},
          {"defaultEps", {0.2, false}},
          {"lossScaling", {0.2, false}},
      },
      AdamMode::AdaMax,
      WeightDecayMode::L2Regularization,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);

  // changed adamMode
  BOOST_CHECK(adam6.hash() != adam1.hash());

  auto adam7 = Adam(
      std::map<std::string, std::pair<float, bool>>{
          {"defaultLearningRate", {0.01, false}},
          {"defaultWeightDecay", {0.1, false}},
          {"defaultBeta1", {0.1, false}},
          {"defaultBeta2", {0.1, false}},
          {"defaultEps", {0.1, false}},
          {"lossScaling", {0.1, false}},
      },
      AdamMode::Lamb,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT,
      {},
      true);
  // changed scaledOptimizerState
  BOOST_CHECK(adam7.hash() != adam1.hash());

  auto adaptive1 = Adaptive({{"defaultLearningRate", {0.02, false}},
                             {"defaultWeightDecay", {0.2, false}},
                             {"defaultAlpha", {0.2, false}},
                             {"defaultMomentum", {0.2, false}},
                             {"defaultEps", {0.2, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::L2Regularization,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  auto adaptive2 = Adaptive({{"defaultLearningRate", {0.02, true}},
                             {"defaultWeightDecay", {0.2, false}},
                             {"defaultAlpha", {0.2, false}},
                             {"defaultMomentum", {0.2, false}},
                             {"defaultEps", {0.2, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::L2Regularization,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  // changed learning rate constness
  BOOST_CHECK(adaptive1.hash() != adaptive2.hash());

  auto adaptive3 = Adaptive({{"defaultLearningRate", {0.2, false}},
                             {"defaultWeightDecay", {0.3, false}},
                             {"defaultAlpha", {0.3, false}},
                             {"defaultMomentum", {0.3, false}},
                             {"defaultEps", {0.3, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::L2Regularization,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  // changing parameter values should not impact hash
  BOOST_CHECK(adaptive1.hash() == adaptive3.hash());

  auto adaptive4 = Adaptive({{"defaultLearningRate", {0.2, false}},
                             {"defaultWeightDecay", {0.3, false}},
                             {"defaultAlpha", {0.3, false}},
                             {"defaultMomentum", {0.3, false}},
                             {"defaultEps", {0.3, false}}},
                            AdaptiveMode::AdaGrad,
                            WeightDecayMode::L2Regularization,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  // changed adaptivemode
  BOOST_CHECK(adaptive1.hash() != adaptive4.hash());

  auto adaptive5 = Adaptive({{"defaultLearningRate", {0.2, false}},
                             {"defaultWeightDecay", {0.3, false}},
                             {"defaultAlpha", {0.3, false}},
                             {"defaultMomentum", {0.3, false}},
                             {"defaultEps", {0.3, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::Decay,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  // changed decay
  BOOST_CHECK(adaptive1.hash() != adaptive5.hash());

  auto adaptive6 = Adaptive({{"defaultLearningRate", {0.2, false}},
                             {"defaultWeightDecay", {0.3, false}},
                             {"defaultAlpha", {0.3, false}},
                             {"defaultMomentum", {0.3, false}},
                             {"defaultEps", {0.3, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::Decay,
                            DataType::FLOAT16,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT);

  // changed accumtype
  BOOST_CHECK(adaptive1.hash() != adaptive6.hash());

  auto adaptive7 = Adaptive({{"defaultLearningRate", {0.2, false}},
                             {"defaultWeightDecay", {0.3, false}},
                             {"defaultAlpha", {0.3, false}},
                             {"defaultMomentum", {0.3, false}},
                             {"defaultEps", {0.3, false}}},
                            AdaptiveMode::RMSProp,
                            WeightDecayMode::Decay,
                            DataType::FLOAT,
                            DataType::FLOAT,
                            DataType::FLOAT16,
                            DataType::FLOAT);

  // changed accltype
  BOOST_CHECK(adaptive1.hash() != adaptive7.hash());
}
