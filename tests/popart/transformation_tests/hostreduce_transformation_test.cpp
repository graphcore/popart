// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE HostReduceTransformationTest

#include <../random_util.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/sync.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace popart;

// TODO(T15374) : refactor the tests to reduce code duplication by
// separating creation of models into functions
// TODO(T16598) Handle different devices by some logic or separate files.

void checkOpSchedule(const std::vector<Op *> &opSchedule,
                     const SessionOptions &options) {
  int numCopiesToHost   = 0;
  int numCopiesToDevice = 0;
  for (const auto &op : opSchedule) {
    if (op->isConvertibleTo<GradCopyToHostOp>()) {
      ++numCopiesToHost;
    } else if (op->isConvertibleTo<GradCopyFromHostOp>()) {
      ++numCopiesToDevice;
    } else if (op->isConvertibleTo<HostSGD0VarUpdate>()) {
      ++numCopiesToDevice;
    }
  }
  BOOST_CHECK(numCopiesToHost == numCopiesToDevice);
}

// Checks that kernel boot commandline is correct for setting up OATT
bool OATT_enabled() {
  const char *oatt_dev0 = "/dev/ipu0_mem";
  const char *ipu_driver_memmap_start =
      "ipu_driver.memmap_start=0x400000000,0x5000000000";
  const char *ipu_driver_memmap_size =
      "ipu_driver.memmap_size=0x400000000,0x400000000";

  std::ifstream t("/proc/cmdline");
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  auto found_start = str.find(ipu_driver_memmap_start) != std::string::npos;
  auto found_size  = str.find(ipu_driver_memmap_size) != std::string::npos;

  const bool oatt_kernel_cmdline_correct = found_start && found_size;
  const bool ipu_mem_device_exists =
      boost::filesystem::exists(boost::filesystem::path(oatt_dev0));

  return oatt_kernel_cmdline_correct && ipu_mem_device_exists;
}

std::shared_ptr<DeviceInfo> acquireAvailableDevice(int numDevices = 1) {
  auto device =
      DeviceManager::createDeviceManager().acquireAvailableDevice(numDevices);

  if (!device) {
    return nullptr;
  }
  return device;
}

// Test: SGD0VarUpdateOps should be replaced by HostReduceVarUpdate ops
// Check that the gradient value is as expected
// Check the writing the weights is as expected
BOOST_AUTO_TEST_CASE(HostReduceTransformationSessionRun) {
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
      v_C_grad[index] = 2 * (v_C_data[index] > 0) - 1;
      v_C_grad[index] *= lossLambda;
    }
  }

  // gradients of A and B,
  // dA = dC.BT
  // dB = AT.dC
  std::vector<float> v_A_grad(A_info.nelms(), 0);
  std::vector<float> v_B_grad(B_info.nelms(), 0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        v_A_grad[m * K + k] += v_C_grad[m * N + n] * v_B_init[k * N + n];
        v_B_grad[k * N + n] += v_C_grad[m * N + n] * v_A_init[m * K + k];
      }
    }
  }

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts             = SessionOptions();
  opts.hostAllReduce    = true;
  opts.hostWeightUpdate = true;

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

  std::vector<float> raw_A_grad_out(A_info.nelms());
  std::vector<float> raw_B_grad_out(B_info.nelms());

  std::vector<float> A_dummy_data(A_info.nelms());
  std::vector<float> B_dummy_data(B_info.nelms());
  for (int i = 0; i < A_dummy_data.size(); ++i) {
    A_dummy_data[i] = static_cast<float>(i);
  }
  for (int i = 0; i < B_dummy_data.size(); ++i) {
    B_dummy_data[i] = static_cast<float>(B_dummy_data.size() - i - 1);
  }

  BOOST_CHECK(session->getHostReduceStreamIds().size() == 4);
  // Careful iterating over getHostReduceStreamIds, no guarantee for order.
  for (const auto &stream_id : session->getHostReduceStreamIds()) {
    if (stream_id.compare(0,
                          strlen(gradientStoreStreamPrefix),
                          gradientStoreStreamPrefix) == 0) {
      const auto grad_id = stream_id.substr(strlen(gradientStoreStreamPrefix));
      if (grad_id == getGradId(A_id)) {
        void *dst   = raw_A_grad_out.data();
        size_t size = A_info.nbytes();
        session->connectStreamToCallback(
            stream_id, [dst, size](void *g) { std::memcpy(dst, g, size); });
      } else if (grad_id == getGradId(B_id)) {
        void *dst   = raw_B_grad_out.data();
        size_t size = B_info.nbytes();
        session->connectStreamToCallback(
            stream_id, [dst, size](void *g) { std::memcpy(dst, g, size); });
      }
    } else if (stream_id.compare(0,
                                 strlen(weightLoadStreamPrefix),
                                 weightLoadStreamPrefix) == 0) {
      const auto weight_id = stream_id.substr(strlen(weightLoadStreamPrefix));
      if (weight_id == A_id) {
        void *src   = A_dummy_data.data();
        size_t size = A_info.nbytes();
        session->connectStreamToCallback(
            stream_id, [src, size](void *w) { std::memcpy(w, src, size); });
      } else if (weight_id == B_id) {
        void *src   = B_dummy_data.data();
        size_t size = B_info.nbytes();
        session->connectStreamToCallback(
            stream_id, [src, size](void *w) { std::memcpy(w, src, size); });
      }
    } else {
      throw error("Unexpected stream id " + stream_id);
    }
  }

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  const auto &ir = session->getIr();
  auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  checkOpSchedule(ops, opts);

  session->weightsFromHost();
  session->run(stepio);

  WeightsIO weightsRead;
  std::vector<float> A_readback(A_info.nelms(), -9.0f);
  std::vector<float> B_readback(B_info.nelms(), -99.0f);
  weightsRead.insert(A_id, {A_readback.data(), A_info});
  weightsRead.insert(B_id, {B_readback.data(), B_info});

  session->weightsToHost();
  session->readWeights(weightsRead);
  BOOST_CHECK_EQUAL_COLLECTIONS(v_A_grad.begin(),
                                v_A_grad.end(),
                                raw_A_grad_out.begin(),
                                raw_A_grad_out.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(v_B_grad.begin(),
                                v_B_grad.end(),
                                raw_B_grad_out.begin(),
                                raw_B_grad_out.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(A_dummy_data.begin(),
                                A_dummy_data.end(),
                                A_readback.begin(),
                                A_readback.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(B_dummy_data.begin(),
                                B_dummy_data.end(),
                                B_readback.begin(),
                                B_readback.end());
}

// Test: check that the execution order of the gradient and variable copies
// is as expected
BOOST_AUTO_TEST_CASE(HostReduceTransformationVarUpdateExecutionOrder) {
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

  // matrix C of shape N x M
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = fdis(eng);
  }
  TensorId C_id = bder->addInitializedInputTensor({v_C_init.data(), C_info});

  // matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = fdis(eng);
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  std::map<TensorId, TensorInfo> idToInfo{{getGradId(A_id), A_info},
                                          {getGradId(B_id), B_info},
                                          {getGradId(C_id), C_info},
                                          {getGradId(D_id), D_info}};

  TensorInfo E_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId E_id = aiOnnx.matmul({A_id, B_id});

  TensorInfo F_info{"FLOAT", std::vector<int64_t>{M, M}};
  TensorId F_id = aiOnnx.matmul({E_id, C_id});

  TensorInfo G_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId G_id = aiOnnx.matmul({F_id, D_id});

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;

  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({G_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{G_id, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts             = SessionOptions();
  opts.hostAllReduce    = true;
  opts.hostWeightUpdate = true;

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
  std::vector<float> raw_G_out(G_info.nelms());
  popart::NDArrayWrapper<float> G_wrapper(raw_G_out.data(), G_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {G_id, G_wrapper},
  };

  session->prepareDevice();

  const auto &ir = session->getIr();
  auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  std::vector<Op *> partial_op_schedule;
  for (auto op : ops) {
    if (dynamic_cast<GradCopyToHostOp *>(op) ||
        dynamic_cast<HostSGD0VarUpdate *>(op)) {
      partial_op_schedule.push_back(op);
    }
  }

  // first 4 should be gradient copies, then followed by var copies
  for (int i = 0; i < 4; ++i) {
    auto asHostReduce =
        dynamic_cast<GradCopyToHostOp *>(partial_op_schedule[i]);
    BOOST_CHECK(asHostReduce);
    auto tensorUpdateId = asHostReduce->inTensor(0)->id;
    const auto &inShape = partial_op_schedule[i]->inShape(0);
    BOOST_CHECK_EQUAL_COLLECTIONS(inShape.begin(),
                                  inShape.end(),
                                  idToInfo.at(tensorUpdateId).shape().begin(),
                                  idToInfo.at(tensorUpdateId).shape().end());
  }

  for (int i = 4; i < partial_op_schedule.size(); ++i) {
    BOOST_CHECK(dynamic_cast<HostSGD0VarUpdate *>(partial_op_schedule[i]));
  }

  std::vector<std::string> callback_handles;
  for (const auto &stream_id : session->getHostReduceStreamIds()) {

    session->connectStreamToCallback(stream_id,
                                     [&callback_handles, stream_id](void *g) {
                                       callback_handles.push_back(stream_id);
                                     });
  }

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  popart::NDArrayWrapper<float> C_wrapper(v_C_init.data(), C_info);
  popart::NDArrayWrapper<float> D_wrapper(v_D_init.data(), D_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {
      {A_id, A_wrapper},
      {B_id, B_wrapper},
      {C_id, C_wrapper},
      {D_id, D_wrapper},
  };

  popart::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->run(stepio);

  // Check that the callbacks are executed in the correct order (all gradients,
  // then all weights)
  for (int i = 0; i < 4; ++i) {
    BOOST_CHECK(callback_handles[i].compare(0,
                                            strlen(gradientStoreStreamPrefix),
                                            gradientStoreStreamPrefix) == 0);
  }
  for (int i = 4; i < 8; ++i) {
    BOOST_CHECK(callback_handles[i].compare(0,
                                            strlen(weightLoadStreamPrefix),
                                            weightLoadStreamPrefix) == 0);
  }
}

/*
#Ground truth for unit test
import torch

K = 6
M = 7
N = 8
replicationFactor = 2
lossLambda = 0.1

A = torch.ones(M,K, requires_grad=True)
B = torch.ones(K,N, requires_grad=True)
C = torch.ones(N,M, requires_grad=True)
D = torch.ones(M,N, requires_grad=True)
E = torch.matmul(A,B)
F = torch.matmul(E,C)
G = torch.matmul(F,D)
err = torch.sum(lossLambda*torch.abs(G))
err.backward()

print(replicationFactor * A.grad[0,0])
print(replicationFactor * B.grad[0,0])
print(replicationFactor * C.grad[0,0])
print(replicationFactor * D.grad[0,0])
*/
// Test: Training with replicated graphs
BOOST_AUTO_TEST_CASE(HostReduceHierarchicalReductionWithReplicatedGraphs) {
  // the dimensions of the matrices
  int K                       = 6;
  int M                       = 7;
  int N                       = 8;
  const int replicationFactor = 2;

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = 1.0f;
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = 1.0f;
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C of shape N x M
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = 1.0f;
  }
  TensorId C_id = bder->addInitializedInputTensor({v_C_init.data(), C_info});

  // matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = 1.0f;
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  std::map<TensorId, TensorInfo> idToInfo{{getGradId(A_id), A_info},
                                          {getGradId(B_id), B_info},
                                          {getGradId(C_id), C_info},
                                          {getGradId(D_id), D_info}};

  TensorInfo E_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId E_id = aiOnnx.matmul({A_id, B_id});

  TensorInfo F_info{"FLOAT", std::vector<int64_t>{M, M}};
  TensorId F_id = aiOnnx.matmul({E_id, C_id});

  TensorInfo G_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId G_id = aiOnnx.matmul({F_id, D_id});

  float lossLambda = 0.1f;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({G_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{G_id, art}});

  auto device = acquireAvailableDevice(replicationFactor);
  if (device != nullptr) {

    auto opts                   = SessionOptions();
    opts.hostAllReduce          = true;
    opts.hostWeightUpdate       = true;
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;

    // training info
    const float learningRate = 0.01f;
    auto optimizer = SGD({{"defaultLearningRate", {learningRate, false}}});

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
    std::vector<float> raw_G_out(replicationFactor * G_info.nelms());
    popart::NDArrayWrapper<float> G_wrapper(
        raw_G_out.data(),
        {replicationFactor, G_info.shape()[0], G_info.shape()[1]});

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {G_id, G_wrapper},
    };

    session->prepareDevice();

    std::unordered_map<TensorId, std::vector<float>> gradients;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        const auto grad_id =
            stream_id.substr(strlen(gradientStoreStreamPrefix));
        const int64_t nelms = idToInfo[grad_id].nelms();

        for (int i = 0; i < replicationFactor; ++i) {
          session->connectStreamToCallback(
              stream_id,
              [&gradients, grad_id, nelms](void *g) {
                float *f      = reinterpret_cast<float *>(g);
                auto gradient = std::vector<float>(f, f + nelms);
                auto found    = gradients.find(grad_id);

                // Check that the streamed gradients are the same for all
                // replicas
                if (found != gradients.end()) {
                  BOOST_CHECK_EQUAL_COLLECTIONS(found->second.begin(),
                                                found->second.end(),
                                                gradient.begin(),
                                                gradient.end());
                } else {
                  gradients[grad_id] = gradient;
                }
              },
              i);
        }
      } else if (stream_id.compare(0,
                                   strlen(weightLoadStreamPrefix),
                                   weightLoadStreamPrefix) == 0) {
        const auto weight_id = stream_id.substr(strlen(weightLoadStreamPrefix));
        const int64_t nelms  = idToInfo[getGradId(weight_id)].nelms();
        session->connectStreamToCallback(stream_id, [nelms](void *w) {
          float *f = reinterpret_cast<float *>(w);
          std::fill(f, f + nelms, 42.0f);
        });
      }
    }

    // inputs:
    std::vector<float> v_A_init_replicated(A_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo A_info_replicated(
        A_info.dataType(), {replicationFactor, A_info.dim(0), A_info.dim(1)});

    std::vector<float> v_B_init_replicated(B_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo B_info_replicated(
        B_info.dataType(), {replicationFactor, B_info.dim(0), B_info.dim(1)});

    std::vector<float> v_C_init_replicated(C_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo C_info_replicated(
        C_info.dataType(), {replicationFactor, C_info.dim(0), C_info.dim(1)});

    std::vector<float> v_D_init_replicated(D_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo D_info_replicated(
        D_info.dataType(), {replicationFactor, D_info.dim(0), D_info.dim(1)});

    popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                            A_info_replicated);
    popart::NDArrayWrapper<float> B_wrapper(v_B_init_replicated.data(),
                                            B_info_replicated);
    popart::NDArrayWrapper<float> C_wrapper(v_C_init_replicated.data(),
                                            C_info_replicated);
    popart::NDArrayWrapper<float> D_wrapper(v_D_init_replicated.data(),
                                            D_info_replicated);

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {A_id, A_wrapper},
        {B_id, B_wrapper},
        {C_id, C_wrapper},
        {D_id, D_wrapper},
    };

    popart::StepIO stepio(inputs, anchors);

    const auto &ir = session->getIr();
    auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    checkOpSchedule(ops, opts);

    session->weightsFromHost();
    session->run(stepio);

    // Ground truths are computed in PyTorch
    const float A_grad_ground_truth_val = 89.6f;
    const float B_grad_ground_truth_val = 78.4f;
    const float C_grad_ground_truth_val = 67.2f;
    const float D_grad_ground_truth_val = 67.2f;
    for (auto &&x : gradients[getGradId(A_id)]) {
      BOOST_CHECK_CLOSE(x, 89.6f, 1e-4f);
    }
    for (auto &&x : gradients[getGradId(B_id)]) {
      BOOST_CHECK_CLOSE(x, 78.4f, 1e-4f);
    }
    for (auto &&x : gradients[getGradId(C_id)]) {
      BOOST_CHECK_CLOSE(x, 67.2f, 1e-4f);
    }
    for (auto &&x : gradients[getGradId(D_id)]) {
      BOOST_CHECK_CLOSE(x, 67.2f, 1e-4f);
    }

    for (const auto &g : gradients) {
      BOOST_CHECK(idToInfo[g.first].nelms() == g.second.size());
    }

    WeightsIO weightsRead;
    std::vector<float> A_readback(A_info.nelms(), -9.0f);
    std::vector<float> B_readback(B_info.nelms(), -99.0f);
    std::vector<float> C_readback(C_info.nelms(), -99.0f);
    std::vector<float> D_readback(D_info.nelms(), -99.0f);
    weightsRead.insert(A_id, {A_readback.data(), A_info});
    weightsRead.insert(B_id, {B_readback.data(), B_info});
    weightsRead.insert(C_id, {C_readback.data(), C_info});
    weightsRead.insert(D_id, {D_readback.data(), D_info});

    session->weightsToHost();
    session->readWeights(weightsRead);

    for (auto &&x : A_readback) {
      BOOST_CHECK(x == 42.0f);
    }
    for (auto &&x : B_readback) {
      BOOST_CHECK(x == 42.0f);
    }
    for (auto &&x : C_readback) {
      BOOST_CHECK(x == 42.0f);
    }
    for (auto &&x : D_readback) {
      BOOST_CHECK(x == 42.0f);
    }
  }
}

/*
# Ground truth for unit test
import torch
K = 6
M = 7
N = 8

A = torch.ones(M,K, requires_grad=True)
B = torch.ones(K,N, requires_grad=True)
C = torch.ones(N,M, requires_grad=True)
D = torch.ones(M,N, requires_grad=True)
E = torch.matmul(A, B)
F = torch.matmul(E, C)
G = torch.matmul(F, D)

params = [A,B,C,D]
l1loss = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params, lr=1)

optimizer.zero_grad()
err = torch.sum(torch.abs(G))
err.backward()
optimizer.step()

print(A[0,0])
print(B[0,0])
print(C[0,0])
print(D[0,0])
*/
// Test: bidirectional gradient streaming and device side variable update
BOOST_AUTO_TEST_CASE(HostReduceTransformationGradientStoreGradientLoad) {
  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = 1.0f;
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = 1.0f;
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C of shape N x M
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = 1.0f;
  }
  TensorId C_id = bder->addInitializedInputTensor({v_C_init.data(), C_info});

  // matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = 1.0f;
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  std::map<TensorId, TensorInfo> idToInfo{{getGradId(A_id), A_info},
                                          {getGradId(B_id), B_info},
                                          {getGradId(C_id), C_info},
                                          {getGradId(D_id), D_info}};

  std::map<TensorId, float> idToGradVal;

  TensorInfo E_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId E_id = aiOnnx.matmul({A_id, B_id});

  TensorInfo F_info{"FLOAT", std::vector<int64_t>{M, M}};
  TensorId F_id = aiOnnx.matmul({E_id, C_id});

  TensorInfo G_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId G_id = aiOnnx.matmul({F_id, D_id});

  float lossLambda = 1.0f;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({G_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{G_id, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts             = SessionOptions();
  opts.hostAllReduce    = true;
  opts.hostWeightUpdate = false;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {1.0f, false}}});

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
  std::vector<float> raw_G_out(G_info.nelms());
  popart::NDArrayWrapper<float> G_wrapper(raw_G_out.data(), G_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {G_id, G_wrapper},
  };

  session->prepareDevice();

  const auto &ir = session->getIr();
  auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  std::vector<Op *> partial_op_schedule;
  for (auto op : ops) {
    if (dynamic_cast<GradCopyToHostOp *>(op) ||
        dynamic_cast<GradCopyFromHostOp *>(op)) {
      partial_op_schedule.push_back(op);
    }
  }

  // first 4 should be gradient copies, then followed by var copies
  for (int i = 0; i < 4; ++i) {
    auto asHostReduce =
        dynamic_cast<GradCopyToHostOp *>(partial_op_schedule[i]);
    BOOST_CHECK(asHostReduce);
    auto tensorUpdateId = asHostReduce->inTensor(0)->id;
    const auto &inShape = partial_op_schedule[i]->inShape(0);
    BOOST_CHECK_EQUAL_COLLECTIONS(inShape.begin(),
                                  inShape.end(),
                                  idToInfo.at(tensorUpdateId).shape().begin(),
                                  idToInfo.at(tensorUpdateId).shape().end());
  }

  for (int i = 4; i < partial_op_schedule.size(); ++i) {
    BOOST_CHECK(dynamic_cast<GradCopyFromHostOp *>(partial_op_schedule[i]));
  }

  std::vector<std::string> callback_handles;
  for (const auto &stream_id : session->getHostReduceStreamIds()) {
    if (stream_id.compare(0,
                          strlen(gradientStoreStreamPrefix),
                          gradientStoreStreamPrefix) == 0) {
      session->connectStreamToCallback(
          stream_id, [&callback_handles, &idToGradVal, stream_id](void *g) {
            callback_handles.push_back(stream_id);
            const auto grad_id =
                stream_id.substr(strlen(gradientStoreStreamPrefix));
            float *f             = reinterpret_cast<float *>(g);
            idToGradVal[grad_id] = f[0];
          });
    } else if (stream_id.compare(0,
                                 strlen(gradientLoadStreamPrefix),
                                 gradientLoadStreamPrefix) == 0) {
      session->connectStreamToCallback(
          stream_id,
          [&callback_handles, &idToGradVal, &idToInfo, stream_id](void *g) {
            callback_handles.push_back(stream_id);
            const auto grad_id =
                stream_id.substr(strlen(gradientLoadStreamPrefix));
            const float grad_val = idToGradVal[grad_id];
            const size_t nelms   = idToInfo[grad_id].nelms();
            float *f             = reinterpret_cast<float *>(g);
            std::fill(f, f + nelms, grad_val);
          });
    }
  }

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  popart::NDArrayWrapper<float> C_wrapper(v_C_init.data(), C_info);
  popart::NDArrayWrapper<float> D_wrapper(v_D_init.data(), D_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {
      {A_id, A_wrapper},
      {B_id, B_wrapper},
      {C_id, C_wrapper},
      {D_id, D_wrapper},
  };

  popart::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->run(stepio);

  // Check that the callbacks are executed in the correct order (all gradients,
  // then all weights)
  for (int i = 0; i < 4; ++i) {
    // "gr_" from gradientStoreStreamId
    BOOST_CHECK(callback_handles[i].compare(0,
                                            strlen(gradientStoreStreamPrefix),
                                            gradientStoreStreamPrefix) == 0);
  }
  for (int i = 4; i < 8; ++i) {
    BOOST_CHECK(callback_handles[i].compare(0,
                                            strlen(gradientLoadStreamPrefix),
                                            gradientLoadStreamPrefix) == 0);
  }

  WeightsIO weightsRead;
  std::vector<float> A_readback(A_info.nelms(), -9.0f);
  std::vector<float> B_readback(B_info.nelms(), -99.0f);
  std::vector<float> C_readback(C_info.nelms(), -99.0f);
  std::vector<float> D_readback(D_info.nelms(), -99.0f);
  weightsRead.insert(A_id, {A_readback.data(), A_info});
  weightsRead.insert(B_id, {B_readback.data(), B_info});
  weightsRead.insert(C_id, {C_readback.data(), C_info});
  weightsRead.insert(D_id, {D_readback.data(), D_info});

  session->weightsToHost();
  session->readWeights(weightsRead);
  for (auto &&x : A_readback) {
    BOOST_CHECK_CLOSE(x, -447.0f, 1e-5f);
  }
  for (auto &&x : B_readback) {
    BOOST_CHECK_CLOSE(x, -391.0f, 1e-5f);
  }
  for (auto &&x : C_readback) {
    BOOST_CHECK_CLOSE(x, -335.0f, 1e-5f);
  }
  for (auto &&x : D_readback) {
    BOOST_CHECK_CLOSE(x, -335.0f, 1e-5f);
  }
}

/*
# Ground truth for unit test
import torch
K = 6
M = 7
N = 8

A = torch.ones(M,K, requires_grad=True)
B = torch.ones(K,N, requires_grad=True)
C = torch.ones(N,M, requires_grad=True)
D = torch.ones(M,N, requires_grad=True)
E = torch.matmul(A, B)
F = torch.matmul(E, C)
G = torch.matmul(F, D)

replicationFactor = 2
params = [A,B,C,D]
l1loss = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params, lr=0.1)

optimizer.zero_grad()
err = torch.sum(torch.abs(G))
err.backward()
optimizer.step()

print(A[0,0])
print(B[0,0])
print(C[0,0])
print(D[0,0])

*/
// Test: bidirectional gradient streaming and device side variable update
BOOST_AUTO_TEST_CASE(
    HostReduceTransformationGradientStoreGradientLoadReplicated) {
  // the dimensions of the matrices
  int K                       = 6;
  int M                       = 7;
  int N                       = 8;
  const int replicationFactor = 2;

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = 1.0f;
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = 1.0f;
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C of shape N x M
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = 1.0f;
  }
  TensorId C_id = bder->addInitializedInputTensor({v_C_init.data(), C_info});

  // matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = 1.0f;
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  std::map<TensorId, TensorInfo> idToInfo{{getGradId(A_id), A_info},
                                          {getGradId(B_id), B_info},
                                          {getGradId(C_id), C_info},
                                          {getGradId(D_id), D_info}};

  std::map<TensorId, float> idToGradVal;

  TensorInfo E_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId E_id = aiOnnx.matmul({A_id, B_id});

  TensorInfo F_info{"FLOAT", std::vector<int64_t>{M, M}};
  TensorId F_id = aiOnnx.matmul({E_id, C_id});

  TensorInfo G_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId G_id = aiOnnx.matmul({F_id, D_id});

  float lossLambda = 1.0f;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({G_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;

  auto dataFlow = DataFlow(batchesPerStep, {{G_id, art}});

  auto device = acquireAvailableDevice(replicationFactor);

  if (device != nullptr) {
    auto opts                   = SessionOptions();
    opts.hostAllReduce          = true;
    opts.hostWeightUpdate       = false;
    opts.enableReplicatedGraphs = true;
    opts.replicatedGraphCount   = replicationFactor;

    // training info
    auto optimizer = SGD({{"defaultLearningRate", {0.1f, false}}});

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    std::vector<float> raw_G_out(replicationFactor * G_info.nelms());
    popart::NDArrayWrapper<float> G_wrapper(
        raw_G_out.data(),
        {replicationFactor, G_info.shape()[0], G_info.shape()[1]});

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {G_id, G_wrapper},
    };

    session->prepareDevice();

    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        for (int i = 0; i < replicationFactor; ++i) {
          session->connectStreamToCallback(
              stream_id,
              [&idToGradVal, stream_id](void *g) {
                const auto grad_id =
                    stream_id.substr(strlen(gradientStoreStreamPrefix));
                float *f             = reinterpret_cast<float *>(g);
                idToGradVal[grad_id] = f[0] / (float)replicationFactor;
              },
              i);
        }
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&idToGradVal, &idToInfo, stream_id](void *g) {
              const auto grad_id =
                  stream_id.substr(strlen(gradientLoadStreamPrefix));
              const float grad_val = idToGradVal[grad_id];
              const size_t nelms   = idToInfo[grad_id].nelms();
              float *f             = reinterpret_cast<float *>(g);
              std::fill(f, f + nelms, grad_val);
            });
      }
    }

    // inputs:
    std::vector<float> v_A_init_replicated(A_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo A_info_replicated(
        A_info.dataType(), {replicationFactor, A_info.dim(0), A_info.dim(1)});

    std::vector<float> v_B_init_replicated(B_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo B_info_replicated(
        B_info.dataType(), {replicationFactor, B_info.dim(0), B_info.dim(1)});

    std::vector<float> v_C_init_replicated(C_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo C_info_replicated(
        C_info.dataType(), {replicationFactor, C_info.dim(0), C_info.dim(1)});

    std::vector<float> v_D_init_replicated(D_info.nelms() * replicationFactor,
                                           1.0f);
    TensorInfo D_info_replicated(
        D_info.dataType(), {replicationFactor, D_info.dim(0), D_info.dim(1)});

    popart::NDArrayWrapper<float> A_wrapper(v_A_init_replicated.data(),
                                            A_info_replicated);
    popart::NDArrayWrapper<float> B_wrapper(v_B_init_replicated.data(),
                                            B_info_replicated);
    popart::NDArrayWrapper<float> C_wrapper(v_C_init_replicated.data(),
                                            C_info_replicated);
    popart::NDArrayWrapper<float> D_wrapper(v_D_init_replicated.data(),
                                            D_info_replicated);

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {A_id, A_wrapper},
        {B_id, B_wrapper},
        {C_id, C_wrapper},
        {D_id, D_wrapper},
    };

    popart::StepIO stepio(inputs, anchors);

    const auto &ir = session->getIr();
    auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    checkOpSchedule(ops, opts);

    session->weightsFromHost();
    session->run(stepio);

    WeightsIO weightsRead;
    std::vector<float> A_readback(A_info.nelms(), -9.0f);
    std::vector<float> B_readback(B_info.nelms(), -99.0f);
    std::vector<float> C_readback(C_info.nelms(), -99.0f);
    std::vector<float> D_readback(D_info.nelms(), -99.0f);
    weightsRead.insert(A_id, {A_readback.data(), A_info});
    weightsRead.insert(B_id, {B_readback.data(), B_info});
    weightsRead.insert(C_id, {C_readback.data(), C_info});
    weightsRead.insert(D_id, {D_readback.data(), D_info});

    session->weightsToHost();
    session->readWeights(weightsRead);

    for (auto &&x : A_readback) {
      BOOST_CHECK_CLOSE(x, -43.8f, 1e-4f);
    }
    for (auto &&x : B_readback) {
      BOOST_CHECK_CLOSE(x, -38.2f, 1e-4f);
    }
    for (auto &&x : C_readback) {
      BOOST_CHECK_CLOSE(x, -32.6f, 1e-4f);
    }
    for (auto &&x : D_readback) {
      BOOST_CHECK_CLOSE(x, -32.6f, 1e-4f);
    }
  }
}

// TODO see T24260
BOOST_AUTO_TEST_CASE(HostReduceTransformationWithAccumulation) {
  auto run = [](bool hostAllReduce) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int64_t steps              = 1;
    int64_t microBatchSize     = 6;
    int64_t batchesPerStep     = 3;
    int64_t accumulationFactor = 3;
    auto batchSize             = microBatchSize * accumulationFactor;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t samplesPerStep = batchesPerStep * batchSize;

    int64_t sampleHeight = 2;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

    std::vector<int64_t> weightShape = sampleShape;

    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{batchesPerStep,
                                       accumulationFactor,
                                       microBatchSize,
                                       sampleHeight,
                                       sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * microBatchSize * accumulationFactor;
    int64_t stepDataElms = batchElms * batchesPerStep;

    auto input1 = builder->addInputTensor(microBatchInfo, "1tupni");

    std::vector<float> w0Vals(sampleElms);
    for (auto &x : w0Vals) {
      x = fdis(eng);
    }
    ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
    auto w0              = builder->addInitializedInputTensor(w0Data);
    auto a0              = aiOnnx.add({w0, input1}, "act0");
    auto act0            = aiOnnx.sigmoid({a0});

    std::vector<float> w1Vals(sampleElms);
    for (auto &x : w1Vals) {
      x = fdis(eng);
    }
    ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
    auto w1              = builder->addInitializedInputTensor(w1Data);
    auto act1            = aiOnnx.add({w1, act0}, "act1");

    std::vector<float> w2Vals(sampleElms);
    for (auto &x : w2Vals) {
      x = fdis(eng);
    }
    ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
    auto w2              = builder->addInitializedInputTensor(w2Data);
    auto act2            = aiOnnx.add({w2, act1}, "act2");

    std::vector<float> w3Vals(sampleElms);
    for (auto &x : w3Vals) {
      x = fdis(eng);
    }
    ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
    auto w3              = builder->addInitializedInputTensor(w3Data);
    auto act3            = aiOnnx.add({w3, act2}, "act3");

    std::vector<float> w4Vals(sampleElms);
    for (auto &x : w4Vals) {
      x = fdis(eng);
    }
    ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
    auto w4              = builder->addInitializedInputTensor(w4Data);
    auto act4            = aiOnnx.add({w4, act3}, "act4");

    std::vector<float> w5Vals(sampleElms);
    for (auto &x : w5Vals) {
      x = fdis(eng);
    }
    ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
    auto w5              = builder->addInitializedInputTensor(w5Data);
    auto act5            = aiOnnx.add({w5, act4}, "act5");
    auto l1              = builder->aiGraphcoreOpset1().l1loss({act5}, 1.0);

    auto proto = builder->getModelProto();

    auto dataFlow = DataFlow(batchesPerStep, {{act5, AnchorReturnType("All")}});

    SessionOptions userOptions;
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

    userOptions.hostAllReduce              = hostAllReduce;
    userOptions.enableGradientAccumulation = accumulationFactor > 1;
    userOptions.accumulationFactor         = accumulationFactor;

    std::map<TensorId, TensorInfo> idToInfo{
        {reservedAcclToUpdatePrefix() + getGradId(w0), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w1), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w2), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w3), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w4), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w5), sampleInfo}};

    std::map<TensorId, std::vector<float>> idToGrad{
        {reservedAcclToUpdatePrefix() + w0, std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + w1, std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + w2, std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + w3, std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + w4, std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + w5, std::vector<float>(sampleElms)},
    };
    float learnRate = 1.0;
    auto optimizer  = ConstSGD(learnRate);

    auto device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();

    auto &ir              = session->getIr();
    const auto &mainGraph = ir.getMainGraph();

    if (userOptions.hostAllReduce) {
      checkOpSchedule(mainGraph.getOpSchedule({}, RequireOptimalSchedule::Yes),
                      userOptions);
    }

    std::vector<float> v_input_0(stepDataElms);

    std::vector<float> v_act5(stepDataElms);
    popart::NDArrayWrapper<float> act5_wrapper(v_act5.data(), stepDataShape);

    std::vector<float> w0_grad(stepDataElms);
    popart::NDArrayWrapper<float> w0_grad_wrapper(w0_grad.data(),
                                                  stepDataShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {act5, act5_wrapper}};

    WeightsIO weightsRead;
    std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
    weightsRead.insert(w0, {w0_readback.data(), weightInfo});
    weightsRead.insert(w1, {w1_readback.data(), weightInfo});
    weightsRead.insert(w2, {w2_readback.data(), weightInfo});
    weightsRead.insert(w3, {w3_readback.data(), weightInfo});
    weightsRead.insert(w4, {w4_readback.data(), weightInfo});
    weightsRead.insert(w5, {w5_readback.data(), weightInfo});

    session->weightsFromHost();

    std::vector<std::string> callback_handles;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&callback_handles, &idToGrad, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientStoreStreamPrefix));
              float *f   = reinterpret_cast<float *>(g);
              auto &grad = idToGrad.at(grad_id);
              std::copy(f, f + grad.size(), grad.data());
            });
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id,
            [&callback_handles, &idToGrad, &idToInfo, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientLoadStreamPrefix));
              const auto &grad = idToGrad.at(grad_id);
              float *f         = reinterpret_cast<float *>(g);
              std::copy(grad.begin(), grad.end(), f);
            });
      }
    }

    for (auto &x : v_input_0) {
      x = fdis(eng);
    }

    for (int i = 0; i < steps; ++i) {
      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};
      popart::StepIO stepio(inputs, anchors);

      session->run(stepio);
    }

    session->weightsToHost();
    session->readWeights(weightsRead);

    std::vector<std::vector<float>> ws = {w0_readback,
                                          w1_readback,
                                          w2_readback,
                                          w3_readback,
                                          w4_readback,
                                          w5_readback};
    return ws;
  };

  // Run the model with and without hostAllReduce
  auto hostAllReduceDisabled = run(false);
  auto hostAllReduceEnabled  = run(true);

  BOOST_REQUIRE(hostAllReduceDisabled.size() == hostAllReduceEnabled.size());

  for (int i = 0; i < hostAllReduceDisabled.size(); i++) {
    auto &lhs = hostAllReduceDisabled.at(i);
    auto &rhs = hostAllReduceEnabled.at(i);

    BOOST_REQUIRE(lhs.size() == rhs.size());
    for (int j = 0; j < lhs.size(); j++) {
      BOOST_CHECK(std::fabs(lhs.at(j) - rhs.at(j)) <= 1e-4f);
    }
  }
}

BOOST_AUTO_TEST_CASE(HostReduceTransformationWithPipelining) {
  auto run = [](bool hostAllReduce) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int64_t steps               = 1;
    int64_t batchSize           = 3;
    int64_t batchesPerStep      = 5;
    int64_t accumulationFactor  = 1;
    const bool enablePipelining = true;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t samplesPerStep = batchesPerStep * batchSize;

    int64_t sampleHeight = 2;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

    std::vector<int64_t> weightShape = sampleShape;

    std::vector<int64_t> batchShape{batchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{
        batchesPerStep, batchSize, sampleHeight, sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo batchInfo{"FLOAT", batchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * batchSize;
    int64_t stepDataElms = batchElms * batchesPerStep;

    auto input1 = builder->addInputTensor(batchInfo, "1tupni");

    std::vector<float> w0Vals(sampleElms);
    ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
    auto w0              = builder->addInitializedInputTensor(w0Data);
    auto a0              = aiOnnx.add({w0, input1}, "a0");
    for (auto &x : w0Vals) {
      x = fdis(eng);
    }

    std::vector<float> w1Vals(sampleElms);
    ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
    auto w1              = builder->addInitializedInputTensor(w1Data);
    auto act1            = aiOnnx.matmul({w1, a0}, "act1");
    for (auto &x : w1Vals) {
      x = fdis(eng);
    }

    std::vector<float> w2Vals(sampleElms);
    ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
    auto w2              = builder->addInitializedInputTensor(w2Data);
    auto act2            = aiOnnx.add({w2, act1}, "act2");
    for (auto &x : w2Vals) {
      x = fdis(eng);
    }

    std::vector<float> w3Vals(sampleElms);
    ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
    auto w3              = builder->addInitializedInputTensor(w3Data);
    auto act3            = aiOnnx.matmul({w3, act2}, "act3");
    for (auto &x : w3Vals) {
      x = fdis(eng);
    }

    std::vector<float> w4Vals(sampleElms);
    ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
    auto w4              = builder->addInitializedInputTensor(w4Data);
    auto act4            = aiOnnx.add({w4, act3}, "act4");
    auto a4              = aiOnnx.sigmoid({act4});
    for (auto &x : w4Vals) {
      x = fdis(eng);
    }

    std::vector<float> w5Vals(sampleElms);
    ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
    auto w5              = builder->addInitializedInputTensor(w5Data);
    auto act5            = aiOnnx.add({w5, a4}, "act5");
    auto a5              = aiOnnx.sigmoid({act5});

    float lambda = 1.0;
    auto l1      = builder->aiGraphcoreOpset1().l1loss({a5}, lambda);
    for (auto &x : w5Vals) {
      x = fdis(eng);
    }
    builder->addOutputTensor(a5);

    float learnRate = 1.0;
    auto optimizer  = ConstSGD(learnRate);

    auto proto = builder->getModelProto();

    auto dataFlow = DataFlow(batchesPerStep, {{a5, AnchorReturnType("All")}});

    SessionOptions userOptions;
    userOptions.enablePipelining = enablePipelining;

    std::map<std::string, std::string> deviceOpts;
    if (!userOptions.enablePipelining) {
      deviceOpts = std::map<std::string, std::string>({{"numIPUs", "1"}});
    } else {
      userOptions.virtualGraphMode = VirtualGraphMode::Auto;
      deviceOpts = std::map<std::string, std::string>({{"numIPUs", "3"}});
    }

    userOptions.hostAllReduce              = hostAllReduce;
    userOptions.enableGradientAccumulation = accumulationFactor > 1;
    userOptions.accumulationFactor         = accumulationFactor;

    std::string prefix = "";
    if (userOptions.enableGradientAccumulation) {
      prefix = reservedAcclToUpdatePrefix();
    }

    std::map<TensorId, TensorInfo> idToInfo{
        {prefix + getGradId(w0), sampleInfo},
        {prefix + getGradId(w1), sampleInfo},
        {prefix + getGradId(w2), sampleInfo},
        {prefix + getGradId(w3), sampleInfo},
        {prefix + getGradId(w4), sampleInfo},
        {prefix + getGradId(w5), sampleInfo}};

    std::map<TensorId, std::vector<float>> idToGrad{
        {prefix + getGradId(w0), std::vector<float>(sampleElms)},
        {prefix + getGradId(w1), std::vector<float>(sampleElms)},
        {prefix + getGradId(w2), std::vector<float>(sampleElms)},
        {prefix + getGradId(w3), std::vector<float>(sampleElms)},
        {prefix + getGradId(w4), std::vector<float>(sampleElms)},
        {prefix + getGradId(w5), std::vector<float>(sampleElms)},
    };

    auto device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();

    const auto &ir        = session->getIr();
    const auto &mainGraph = ir.getMainGraph();
    if (userOptions.hostAllReduce) {
      checkOpSchedule(mainGraph.getOpSchedule({}, RequireOptimalSchedule::Yes),
                      userOptions);
    }

    std::vector<float> v_input_0(stepDataElms);

    std::vector<float> v_a5(stepDataElms);
    popart::NDArrayWrapper<float> a5_wrapper(v_a5.data(), stepDataShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {a5, a5_wrapper},
    };

    WeightsIO weightsRead;
    std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
    weightsRead.insert(w0, {w0_readback.data(), weightInfo});
    weightsRead.insert(w1, {w1_readback.data(), weightInfo});
    weightsRead.insert(w2, {w2_readback.data(), weightInfo});
    weightsRead.insert(w3, {w3_readback.data(), weightInfo});
    weightsRead.insert(w4, {w4_readback.data(), weightInfo});
    weightsRead.insert(w5, {w5_readback.data(), weightInfo});

    session->weightsFromHost();

    std::vector<std::string> callback_handles;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&callback_handles, &idToGrad, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientStoreStreamPrefix));
              float *f   = reinterpret_cast<float *>(g);
              auto &grad = idToGrad.at(grad_id);
              std::copy(f, f + grad.size(), grad.data());
            });
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&callback_handles, &idToGrad, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientLoadStreamPrefix));
              const auto &grad = idToGrad.at(grad_id);
              float *f         = reinterpret_cast<float *>(g);
              std::copy(grad.begin(), grad.end(), f);
            });
      }
    }

    for (auto &x : v_input_0) {
      x = fdis(eng);
    }

    for (int i = 0; i < steps; ++i) {
      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);
    }

    session->weightsToHost();
    session->readWeights(weightsRead);

    std::vector<std::vector<float>> ws = {w0_readback,
                                          w1_readback,
                                          w2_readback,
                                          w3_readback,
                                          w4_readback,
                                          w5_readback};

    return ws;
  };

  // Run the model with and without hostAllReduce
  auto hostAllReduceDisabled = run(false);
  auto hostAllReduceEnabled  = run(true);

  BOOST_REQUIRE(hostAllReduceDisabled.size() == hostAllReduceEnabled.size());

  for (int i = 0; i < hostAllReduceDisabled.size(); i++) {
    auto &lhs = hostAllReduceDisabled[i];
    auto &rhs = hostAllReduceEnabled[i];

    BOOST_REQUIRE(lhs.size() == rhs.size());
    for (int j = 0; j < lhs.size(); j++) {
      BOOST_CHECK(std::fabs(lhs[j] - rhs[j]) <= 1e-4f);
    }
  }
}

// TODO see T24260
BOOST_AUTO_TEST_CASE(HostReduceTransformationWithPipeliningAndAccumulation) {
  auto run = [](bool hostAllReduce) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int64_t steps               = 2;
    int64_t microBatchSize      = 4;
    int64_t accumulationFactor  = 5;
    int64_t batchSize           = microBatchSize * accumulationFactor;
    int64_t batchesPerStep      = 6;
    const bool enablePipelining = true;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t samplesPerStep = batchesPerStep * batchSize;

    int64_t sampleHeight = 2;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

    std::vector<int64_t> weightShape = sampleShape;

    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{batchesPerStep,
                                       accumulationFactor,
                                       microBatchSize,
                                       sampleHeight,
                                       sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * batchSize;
    int64_t stepDataElms = batchElms * batchesPerStep;

    auto input1 = builder->addInputTensor(microBatchInfo, "1tupni");

    std::vector<float> w0Vals(sampleElms);
    ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
    auto w0              = builder->addInitializedInputTensor(w0Data);
    auto a0              = aiOnnx.add({w0, input1}, "a0");
    for (auto &x : w0Vals) {
      x = fdis(eng);
    }

    std::vector<float> w1Vals(sampleElms);
    ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
    auto w1              = builder->addInitializedInputTensor(w1Data);
    auto act1            = aiOnnx.matmul({w1, a0}, "act1");
    for (auto &x : w1Vals) {
      x = fdis(eng);
    }

    std::vector<float> w2Vals(sampleElms);
    ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
    auto w2              = builder->addInitializedInputTensor(w2Data);
    auto act2            = aiOnnx.add({w2, act1}, "act2");
    for (auto &x : w2Vals) {
      x = fdis(eng);
    }

    std::vector<float> w3Vals(sampleElms);
    ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
    auto w3              = builder->addInitializedInputTensor(w3Data);
    auto act3            = aiOnnx.matmul({w3, act2}, "act3");
    for (auto &x : w3Vals) {
      x = fdis(eng);
    }

    std::vector<float> w4Vals(sampleElms);
    ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
    auto w4              = builder->addInitializedInputTensor(w4Data);
    auto act4            = aiOnnx.add({w4, act3}, "act4");
    auto a4              = aiOnnx.sigmoid({act4});
    for (auto &x : w4Vals) {
      x = fdis(eng);
    }

    std::vector<float> w5Vals(sampleElms);
    ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
    auto w5              = builder->addInitializedInputTensor(w5Data);
    auto act5            = aiOnnx.add({w5, a4}, "act5");
    auto a5              = aiOnnx.sigmoid({act5});
    float lambda         = 1.0;
    auto l1              = builder->aiGraphcoreOpset1().l1loss({a5}, lambda);
    for (auto &x : w5Vals) {
      x = fdis(eng);
    }
    builder->addOutputTensor(a5);

    float learnRate = 1.0;
    auto optimizer  = ConstSGD(learnRate);

    auto proto = builder->getModelProto();

    auto dataFlow = DataFlow(batchesPerStep, {{a5, AnchorReturnType("All")}});

    SessionOptions userOptions;
    userOptions.enablePipelining = enablePipelining;

    std::map<std::string, std::string> deviceOpts;
    if (!userOptions.enablePipelining) {
      deviceOpts = std::map<std::string, std::string>({{"numIPUs", "1"}});
    } else {
      userOptions.virtualGraphMode = VirtualGraphMode::Auto;
      deviceOpts = std::map<std::string, std::string>({{"numIPUs", "3"}});
    }

    userOptions.hostAllReduce              = hostAllReduce;
    userOptions.enableGradientAccumulation = accumulationFactor > 1;
    userOptions.accumulationFactor         = accumulationFactor;

    std::string prefix = "";
    if (userOptions.enableGradientAccumulation) {
      prefix = reservedAcclToUpdatePrefix();
    }

    std::map<TensorId, std::vector<float>> idToGrad{
        {prefix + w0, std::vector<float>(sampleElms)},
        {prefix + w1, std::vector<float>(sampleElms)},
        {prefix + w2, std::vector<float>(sampleElms)},
        {prefix + w3, std::vector<float>(sampleElms)},
        {prefix + w4, std::vector<float>(sampleElms)},
        {prefix + w5, std::vector<float>(sampleElms)},
    };

    auto device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();

    const auto &ir        = session->getIr();
    const auto &mainGraph = ir.getMainGraph();
    if (userOptions.hostAllReduce) {
      checkOpSchedule(mainGraph.getOpSchedule({}, RequireOptimalSchedule::Yes),
                      userOptions);
    }

    std::vector<float> v_input_0(stepDataElms);

    std::vector<float> v_a5(stepDataElms);
    popart::NDArrayWrapper<float> a5_wrapper(v_a5.data(), stepDataShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {a5, a5_wrapper},
    };

    WeightsIO weightsRead;
    std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
    weightsRead.insert(w0, {w0_readback.data(), weightInfo});
    weightsRead.insert(w1, {w1_readback.data(), weightInfo});
    weightsRead.insert(w2, {w2_readback.data(), weightInfo});
    weightsRead.insert(w3, {w3_readback.data(), weightInfo});
    weightsRead.insert(w4, {w4_readback.data(), weightInfo});
    weightsRead.insert(w5, {w5_readback.data(), weightInfo});

    session->weightsFromHost();

    std::vector<std::string> callback_handles;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&callback_handles, &idToGrad, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientStoreStreamPrefix));
              float *f   = reinterpret_cast<float *>(g);
              auto &grad = idToGrad.at(grad_id);
              std::copy(f, f + grad.size(), grad.data());
            });
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(
            stream_id, [&callback_handles, &idToGrad, stream_id](void *g) {
              callback_handles.push_back(stream_id);
              const auto grad_id =
                  stream_id.substr(strlen(gradientLoadStreamPrefix));
              const auto &grad = idToGrad.at(grad_id);
              float *f         = reinterpret_cast<float *>(g);
              std::copy(grad.begin(), grad.end(), f);
            });
      }
    }

    for (auto &x : v_input_0) {
      x = fdis(eng);
    }

    for (int i = 0; i < steps; ++i) {
      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);
    }

    session->weightsToHost();
    session->readWeights(weightsRead);

    std::vector<std::vector<float>> ws = {w0_readback,
                                          w1_readback,
                                          w2_readback,
                                          w3_readback,
                                          w4_readback,
                                          w5_readback};
    return ws;
  };

  // Run the model with and without hostAllReduce
  auto hostAllReduceDisabled = run(false);
  auto hostAllReduceEnabled  = run(true);

  BOOST_REQUIRE(hostAllReduceDisabled.size() == hostAllReduceEnabled.size());

  for (int i = 0; i < hostAllReduceDisabled.size(); i++) {
    auto &lhs = hostAllReduceDisabled[i];
    auto &rhs = hostAllReduceEnabled[i];

    BOOST_REQUIRE(lhs.size() == rhs.size());
    for (int j = 0; j < lhs.size(); j++) {
      BOOST_CHECK(std::fabs(lhs[j] - rhs[j]) <= 1e-4f);
    }
  }
}

// TODO see T16010
BOOST_AUTO_TEST_CASE(OATTSimpleTest, *boost::unit_test::disabled()) {
  if (!OATT_enabled()) {
    return;
  }

  // the dimensions of the matrices
  int K = 6;
  int M = 7;
  int N = 8;

  // prepare a Builder for creating onnx model
  auto bder   = Builder::create();
  auto aiOnnx = bder->aiOnnxOpset9();

  // matrix A of shape M x K
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{M, K}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = 1.0f;
  }
  TensorId A_id = bder->addInitializedInputTensor({v_A_init.data(), A_info});

  // matrix B of shape K x N
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = 1.0f;
  }
  TensorId B_id = bder->addInitializedInputTensor({v_B_init.data(), B_info});

  // matrix C of shape N x M
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = 1.0f;
  }
  TensorId C_id = bder->addInitializedInputTensor({v_C_init.data(), C_info});

  // matrix D of shape M x N
  TensorInfo D_info{"FLOAT", std::vector<int64_t>{M, N}};
  std::vector<float> v_D_init(D_info.nelms());
  for (auto &val : v_D_init) {
    val = 1.0f;
  }
  TensorId D_id = bder->addInitializedInputTensor({v_D_init.data(), D_info});

  std::map<TensorId, TensorInfo> idToInfo{{getGradId(A_id), A_info},
                                          {getGradId(B_id), B_info},
                                          {getGradId(C_id), C_info},
                                          {getGradId(D_id), D_info}};

  std::map<TensorId, float> idToGradVal;

  TensorInfo E_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId E_id = aiOnnx.matmul({A_id, B_id});

  TensorInfo F_info{"FLOAT", std::vector<int64_t>{M, M}};
  TensorId F_id = aiOnnx.matmul({E_id, C_id});

  TensorInfo G_info{"FLOAT", std::vector<int64_t>{M, N}};
  TensorId G_id = aiOnnx.matmul({F_id, D_id});

  float lossLambda = 1.0f;
  auto l1 =
      bder->aiGraphcoreOpset1().l1loss({G_id}, lossLambda, ReductionType::Sum);

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{G_id, art}});

  auto device = acquireAvailableDevice();
  if (!device) {
    return;
  }
  auto opts                      = SessionOptions();
  opts.hostAllReduce             = true;
  opts.hostAllReduceRemoteBuffer = true;
  opts.hostWeightUpdate          = false;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {1.0f, false}}});

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
  std::vector<float> raw_G_out(G_info.nelms());
  popart::NDArrayWrapper<float> G_wrapper(raw_G_out.data(), G_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {G_id, G_wrapper},
  };

  session->prepareDevice();

  const auto &ir = session->getIr();
  auto ops       = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  // checkOpSchedule(ops, opts);

  std::vector<std::string> callback_handles;
  std::vector<float> temp_buffer(N * N);
  for (const auto &stream_id : session->getHostReduceStreamIds()) {
    if (stream_id.compare(0,
                          strlen(gradientStoreStreamPrefix),
                          gradientStoreStreamPrefix) == 0) {
      session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
        callback_handles.push_back(stream_id);
        const auto grad_id =
            stream_id.substr(strlen(gradientStoreStreamPrefix));
        session->copyFromRemoteBuffer(grad_id, temp_buffer.data(), 0);
        idToGradVal[grad_id] = temp_buffer[0];
      });
    } else if (stream_id.compare(0,
                                 strlen(gradientLoadStreamPrefix),
                                 gradientLoadStreamPrefix) == 0) {
      session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
        callback_handles.push_back(stream_id);
        const auto grad_id = stream_id.substr(strlen(gradientLoadStreamPrefix));

        const float grad_val = idToGradVal.at(grad_id);
        std::fill(temp_buffer.begin(), temp_buffer.end(), grad_val);
        session->copyToRemoteBuffer(temp_buffer.data(), grad_id, 0);
      });
    }
  }

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  popart::NDArrayWrapper<float> C_wrapper(v_C_init.data(), C_info);
  popart::NDArrayWrapper<float> D_wrapper(v_D_init.data(), D_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {
      {A_id, A_wrapper},
      {B_id, B_wrapper},
      {C_id, C_wrapper},
      {D_id, D_wrapper},
  };

  popart::StepIO stepio(inputs, anchors);

  session->weightsFromHost();
  session->run(stepio);

  BOOST_CHECK(callback_handles.size() == 8);

  std::unordered_set<std::string> streamedGradients;
  for (int i = 0; i < callback_handles.size() - 1; ++i) {

    if (callback_handles[i].compare(0,
                                    strlen(gradientStoreStreamPrefix),
                                    gradientStoreStreamPrefix) == 0) {
      streamedGradients.insert(
          callback_handles[i].substr(strlen(gradientStoreStreamPrefix)));
    }
    if (callback_handles[i].compare(0,
                                    strlen(gradientLoadStreamPrefix),
                                    gradientLoadStreamPrefix) == 0) {
      std::string grad_id =
          callback_handles[i].substr(strlen(gradientLoadStreamPrefix));
      // Ensure that the gradient had been streamed to host prior
      // to being streamed to device
      BOOST_CHECK(streamedGradients.count(grad_id) == 1);
    }
  }

  WeightsIO weightsRead;
  std::vector<float> A_readback(A_info.nelms(), -9.0f);
  std::vector<float> B_readback(B_info.nelms(), -99.0f);
  std::vector<float> C_readback(C_info.nelms(), -99.0f);
  std::vector<float> D_readback(D_info.nelms(), -99.0f);
  weightsRead.insert(A_id, {A_readback.data(), A_info});
  weightsRead.insert(B_id, {B_readback.data(), B_info});
  weightsRead.insert(C_id, {C_readback.data(), C_info});
  weightsRead.insert(D_id, {D_readback.data(), D_info});

  session->weightsToHost();
  session->readWeights(weightsRead);
  for (auto &&x : A_readback) {
    BOOST_CHECK_CLOSE(x, -447.0f, 1e-5f);
  }
  for (auto &&x : B_readback) {
    BOOST_CHECK_CLOSE(x, -391.0f, 1e-5f);
  }
  for (auto &&x : C_readback) {
    BOOST_CHECK_CLOSE(x, -335.0f, 1e-5f);
  }
  for (auto &&x : D_readback) {
    BOOST_CHECK_CLOSE(x, -335.0f, 1e-5f);
  }
}

// TODO see T16010
BOOST_AUTO_TEST_CASE(OATTWithAccumulation, *boost::unit_test::disabled()) {
  if (!OATT_enabled()) {
    return;
  }
  auto run = [](bool hostAllReduce) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int64_t steps              = 1;
    int64_t microBatchSize     = 6;
    int64_t batchesPerStep     = 3;
    int64_t accumulationFactor = 3;
    auto batchSize             = microBatchSize * accumulationFactor;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t samplesPerStep = batchesPerStep * batchSize;

    int64_t sampleHeight = 2;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

    std::vector<int64_t> weightShape = sampleShape;

    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{batchesPerStep,
                                       accumulationFactor,
                                       microBatchSize,
                                       sampleHeight,
                                       sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * microBatchSize * accumulationFactor;
    int64_t stepDataElms = batchElms * batchesPerStep;

    auto input1 = builder->addInputTensor(microBatchInfo, "1tupni");

    std::vector<float> w0Vals(sampleElms);
    for (auto &x : w0Vals) {
      x = fdis(eng);
    }
    ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
    auto w0              = builder->addInitializedInputTensor(w0Data);
    auto a0              = aiOnnx.add({w0, input1}, "act0");
    auto act0            = aiOnnx.sigmoid({a0});

    std::vector<float> w1Vals(sampleElms);
    for (auto &x : w1Vals) {
      x = fdis(eng);
    }
    ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
    auto w1              = builder->addInitializedInputTensor(w1Data);
    auto act1            = aiOnnx.add({w1, act0}, "act1");

    std::vector<float> w2Vals(sampleElms);
    for (auto &x : w2Vals) {
      x = fdis(eng);
    }
    ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
    auto w2              = builder->addInitializedInputTensor(w2Data);
    auto act2            = aiOnnx.add({w2, act1}, "act2");

    std::vector<float> w3Vals(sampleElms);
    for (auto &x : w3Vals) {
      x = fdis(eng);
    }
    ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
    auto w3              = builder->addInitializedInputTensor(w3Data);
    auto act3            = aiOnnx.add({w3, act2}, "act3");

    std::vector<float> w4Vals(sampleElms);
    for (auto &x : w4Vals) {
      x = fdis(eng);
    }
    ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
    auto w4              = builder->addInitializedInputTensor(w4Data);
    auto act4            = aiOnnx.add({w4, act3}, "act4");

    std::vector<float> w5Vals(sampleElms);
    for (auto &x : w5Vals) {
      x = fdis(eng);
    }
    ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
    auto w5              = builder->addInitializedInputTensor(w5Data);
    auto act5            = aiOnnx.add({w5, act4}, "act5");
    float lambda         = 1.0;
    auto l1              = builder->aiGraphcoreOpset1().l1loss({act5}, lambda);

    auto proto = builder->getModelProto();

    auto dataFlow = DataFlow(batchesPerStep, {{act5, AnchorReturnType("All")}});

    SessionOptions userOptions;
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

    if (hostAllReduce) {
      userOptions.hostAllReduceRemoteBuffer = true;
    }
    userOptions.hostAllReduce              = hostAllReduce;
    userOptions.enableGradientAccumulation = accumulationFactor > 1;
    userOptions.accumulationFactor         = accumulationFactor;

    std::map<TensorId, TensorInfo> idToInfo{
        {reservedAcclToUpdatePrefix() + getGradId(w0), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w1), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w2), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w3), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w4), sampleInfo},
        {reservedAcclToUpdatePrefix() + getGradId(w5), sampleInfo}};

    std::map<TensorId, std::vector<float>> idToGrad{
        {reservedAcclToUpdatePrefix() + getGradId(w0),
         std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + getGradId(w1),
         std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + getGradId(w2),
         std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + getGradId(w3),
         std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + getGradId(w4),
         std::vector<float>(sampleElms)},
        {reservedAcclToUpdatePrefix() + getGradId(w5),
         std::vector<float>(sampleElms)},
    };

    float learnRate = 1.0;
    auto optimizer  = ConstSGD(learnRate);

    auto device = acquireAvailableDevice();
    if (!device) {
      return std::vector<std::vector<float>>();
    }

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();

    auto &ir              = session->getIr();
    const auto &mainGraph = ir.getMainGraph();

    if (userOptions.hostAllReduce) {
      checkOpSchedule(mainGraph.getOpSchedule({}, RequireOptimalSchedule::Yes),
                      userOptions);
    }
    std::vector<float> v_input_0(stepDataElms);

    std::vector<float> v_act5(stepDataElms);
    popart::NDArrayWrapper<float> act5_wrapper(v_act5.data(), stepDataShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {act5, act5_wrapper}};

    WeightsIO weightsRead;
    std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
    weightsRead.insert(w0, {w0_readback.data(), weightInfo});
    weightsRead.insert(w1, {w1_readback.data(), weightInfo});
    weightsRead.insert(w2, {w2_readback.data(), weightInfo});
    weightsRead.insert(w3, {w3_readback.data(), weightInfo});
    weightsRead.insert(w4, {w4_readback.data(), weightInfo});
    weightsRead.insert(w5, {w5_readback.data(), weightInfo});

    session->weightsFromHost();
    std::vector<std::string> callback_handles;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
          callback_handles.push_back(stream_id);
          const auto grad_id =
              stream_id.substr(strlen(gradientStoreStreamPrefix));
          auto &grad = idToGrad.at(grad_id);
          session->copyFromRemoteBuffer(grad_id, grad.data(), 0);
        });
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
          callback_handles.push_back(stream_id);
          const auto grad_id =
              stream_id.substr(strlen(gradientLoadStreamPrefix));
          auto &grad = idToGrad.at(grad_id);
          session->copyToRemoteBuffer(grad.data(), grad_id, 0);
        });
      }
    }

    for (auto &x : v_input_0) {
      x = fdis(eng);
    }

    for (int i = 0; i < steps; ++i) {
      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};
      popart::StepIO stepio(inputs, anchors);

      session->run(stepio);
    }

    session->weightsToHost();
    session->readWeights(weightsRead);

    std::vector<std::vector<float>> ws = {w0_readback,
                                          w1_readback,
                                          w2_readback,
                                          w3_readback,
                                          w4_readback,
                                          w5_readback};
    return ws;
  };

  // Run the model with and without hostAllReduce
  auto hostAllReduceDisabled = run(false);
  auto hostAllReduceEnabled  = run(true);

  if (hostAllReduceDisabled.empty()) {
    return;
  }

  BOOST_REQUIRE(hostAllReduceDisabled.size() == hostAllReduceEnabled.size());

  for (int i = 0; i < hostAllReduceDisabled.size(); i++) {
    auto &lhs = hostAllReduceDisabled.at(i);
    auto &rhs = hostAllReduceEnabled.at(i);

    BOOST_REQUIRE(lhs.size() == rhs.size());
    for (int j = 0; j < lhs.size(); j++) {
      BOOST_CHECK(std::fabs(lhs.at(j) - rhs.at(j)) <= 1e-4f);
    }
  }
}

// TODO see T16010
BOOST_AUTO_TEST_CASE(OATTWithPipeliningAndAccumulation,
                     *boost::unit_test::disabled()) {
  if (!OATT_enabled()) {
    return;
  }
  auto run = [](bool hostAllReduce) {
    auto builder                = Builder::create();
    auto aiOnnx                 = builder->aiOnnxOpset9();
    auto aiGraphcore            = builder->aiGraphcoreOpset1();
    const bool enablePipelining = true;

    int64_t steps              = 2;
    int64_t microBatchSize     = 5;
    int64_t batchesPerStep     = 5;
    int64_t accumulationFactor = 7;
    auto batchSize             = microBatchSize * accumulationFactor;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t samplesPerStep = batchesPerStep * batchSize;

    int64_t sampleHeight = 2;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

    std::vector<int64_t> weightShape = sampleShape;

    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};

    std::vector<int64_t> stepDataShape{batchesPerStep,
                                       accumulationFactor,
                                       microBatchSize,
                                       sampleHeight,
                                       sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * microBatchSize * accumulationFactor;
    int64_t stepDataElms = batchElms * batchesPerStep;

    auto input1 = builder->addInputTensor(microBatchInfo, "1tupni");

    std::vector<float> w0Vals(sampleElms);
    ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
    auto w0              = builder->addInitializedInputTensor(w0Data);
    auto a0              = aiOnnx.add({w0, input1}, "a0");
    for (auto &x : w0Vals) {
      x = fdis(eng);
    }

    std::vector<float> w1Vals(sampleElms);
    ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
    auto w1              = builder->addInitializedInputTensor(w1Data);
    auto act1            = aiOnnx.matmul({w1, a0}, "act1");
    for (auto &x : w1Vals) {
      x = fdis(eng);
    }

    std::vector<float> w2Vals(sampleElms);
    ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
    auto w2              = builder->addInitializedInputTensor(w2Data);
    auto act2            = aiOnnx.add({w2, act1}, "act2");
    for (auto &x : w2Vals) {
      x = fdis(eng);
    }

    std::vector<float> w3Vals(sampleElms);
    ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
    auto w3              = builder->addInitializedInputTensor(w3Data);
    auto act3            = aiOnnx.matmul({w3, act2}, "act3");
    for (auto &x : w3Vals) {
      x = fdis(eng);
    }

    std::vector<float> w4Vals(sampleElms);
    ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
    auto w4              = builder->addInitializedInputTensor(w4Data);
    auto act4            = aiOnnx.add({w4, act3}, "act4");
    auto a4              = aiOnnx.sigmoid({act4});
    for (auto &x : w4Vals) {
      x = fdis(eng);
    }

    std::vector<float> w5Vals(sampleElms);
    ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
    auto w5              = builder->addInitializedInputTensor(w5Data);
    auto act5            = aiOnnx.add({w5, a4}, "act5");
    auto a5              = aiOnnx.sigmoid({act5});
    for (auto &x : w5Vals) {
      x = fdis(eng);
    }
    float lambda = 1.0;
    auto l1      = builder->aiGraphcoreOpset1().l1loss({a5}, lambda);

    float learnRate = 1.0;
    auto optimizer  = ConstSGD(learnRate);
    auto proto      = builder->getModelProto();

    auto dataFlow = DataFlow(batchesPerStep, {{a5, AnchorReturnType("All")}});

    SessionOptions userOptions;
    userOptions.enablePipelining = enablePipelining;

    if (hostAllReduce) {
      userOptions.hostAllReduceRemoteBuffer = true;
    }

    userOptions.virtualGraphMode           = VirtualGraphMode::Auto;
    userOptions.hostAllReduce              = hostAllReduce;
    userOptions.enableGradientAccumulation = accumulationFactor > 1;
    userOptions.accumulationFactor         = accumulationFactor;

    std::string prefix = "";
    if (userOptions.enableGradientAccumulation) {
      prefix = reservedAcclToUpdatePrefix();
    }

    std::map<TensorId, std::vector<float>> idToGrad{
        {prefix + getGradId(w0), std::vector<float>(sampleElms)},
        {prefix + getGradId(w1), std::vector<float>(sampleElms)},
        {prefix + getGradId(w2), std::vector<float>(sampleElms)},
        {prefix + getGradId(w3), std::vector<float>(sampleElms)},
        {prefix + getGradId(w4), std::vector<float>(sampleElms)},
        {prefix + getGradId(w5), std::vector<float>(sampleElms)},
    };

    auto device = acquireAvailableDevice(4);
    if (!device) {
      return std::vector<std::vector<float>>();
    }

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        l1,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();
    const auto &ir        = session->getIr();
    const auto &mainGraph = ir.getMainGraph();
    if (userOptions.hostAllReduce) {
      checkOpSchedule(mainGraph.getOpSchedule({}, RequireOptimalSchedule::Yes),
                      userOptions);
    }

    std::vector<float> v_input_0(stepDataElms);

    std::vector<float> v_a5(stepDataElms);
    popart::NDArrayWrapper<float> a5_wrapper(v_a5.data(), stepDataShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {a5, a5_wrapper},
    };

    WeightsIO weightsRead;
    std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
    std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
    weightsRead.insert(w0, {w0_readback.data(), weightInfo});
    weightsRead.insert(w1, {w1_readback.data(), weightInfo});
    weightsRead.insert(w2, {w2_readback.data(), weightInfo});
    weightsRead.insert(w3, {w3_readback.data(), weightInfo});
    weightsRead.insert(w4, {w4_readback.data(), weightInfo});
    weightsRead.insert(w5, {w5_readback.data(), weightInfo});

    session->weightsFromHost();
    std::vector<std::string> callback_handles;
    for (const auto &stream_id : session->getHostReduceStreamIds()) {
      if (stream_id.compare(0,
                            strlen(gradientStoreStreamPrefix),
                            gradientStoreStreamPrefix) == 0) {
        session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
          callback_handles.push_back(stream_id);
          const auto grad_id =
              stream_id.substr(strlen(gradientStoreStreamPrefix));
          auto &grad = idToGrad.at(grad_id);
          session->copyFromRemoteBuffer(grad_id, grad.data(), 0);
        });
      } else if (stream_id.compare(0,
                                   strlen(gradientLoadStreamPrefix),
                                   gradientLoadStreamPrefix) == 0) {
        session->connectStreamToCallback(stream_id, [&, stream_id](void *g) {
          callback_handles.push_back(stream_id);
          const auto grad_id =
              stream_id.substr(strlen(gradientLoadStreamPrefix));
          auto &grad = idToGrad.at(grad_id);
          session->copyToRemoteBuffer(grad.data(), grad_id, 0);
        });
      }
    }

    for (auto &x : v_input_0) {
      x = fdis(eng);
    }

    for (int i = 0; i < steps; ++i) {
      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};
      popart::StepIO stepio(inputs, anchors);
      session->run(stepio);
    }

    session->weightsToHost();
    session->readWeights(weightsRead);

    std::vector<std::vector<float>> ws = {w0_readback,
                                          w1_readback,
                                          w2_readback,
                                          w3_readback,
                                          w4_readback,
                                          w5_readback};
    return ws;
  };

  // Run the model with and without hostAllReduce
  auto hostAllReduceDisabled = run(false);
  auto hostAllReduceEnabled  = run(true);
  if (hostAllReduceDisabled.empty()) {
    return;
  }

  BOOST_REQUIRE(hostAllReduceDisabled.size() == hostAllReduceEnabled.size());

  for (int i = 0; i < hostAllReduceDisabled.size(); i++) {
    auto &lhs = hostAllReduceDisabled[i];
    auto &rhs = hostAllReduceEnabled[i];

    BOOST_REQUIRE(lhs.size() == rhs.size());
    for (int j = 0; j < lhs.size(); j++) {
      BOOST_CHECK(std::fabs(lhs[j] - rhs[j]) <= 1e-4f);
    }
  }
}
