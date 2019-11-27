#define BOOST_TEST_MODULE HostReduceTransformationTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

using namespace popart;

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
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<float> fdis(-4, 4);

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
  bder->addOutputTensor(C_id);

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;

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
  auto art        = AnchorReturnType("ALL");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{C_id, art}});

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  auto opts          = SessionOptions();
  opts.hostAllReduce = true;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});
  std::unique_ptr<Loss> l1_loss(
      new L1Loss(C_id, "l1LossVal", lossLambda, ReductionType::SUM));
  std::vector<Loss *> losses{l1_loss.get()};

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      losses,
      optimizer,
      cpuDevice,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::DEFAULT));

  // prepare the anchors. We have the output C,
  std::vector<float> raw_C_out(C_info.nelms());
  popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {C_id, C_wrapper},
  };

  session->prepareDevice();

  std::vector<float> raw_A_grad_out(A_info.nelms());
  std::vector<float> raw_B_grad_out(B_info.nelms());
  std::vector<std::vector<float>> raw_grads_out = {raw_A_grad_out,
                                                   raw_B_grad_out};

  std::vector<float> A_dummy_data(A_info.nelms());
  std::vector<float> B_dummy_data(B_info.nelms());
  for (int i = 0; i < A_dummy_data.size(); ++i) {
    A_dummy_data[i] = static_cast<float>(i);
  }
  for (int i = 0; i < B_dummy_data.size(); ++i) {
    B_dummy_data[i] = static_cast<float>(B_dummy_data.size() - i - 1);
  }

  std::vector<std::vector<float>> dummy_data = {A_dummy_data, B_dummy_data};

  BOOST_CHECK(session->getGradAndVarStreamIds().size() == 2);
  // Careful iterating over getGradAndVarStreamIds, no guarantee for order.
  for (const auto &gv : session->getGradAndVarStreamIds()) {
    const auto &grad_stream_id   = gv.first;
    const auto &weight_stream_id = gv.second;

    int i{};
    if (weight_stream_id == "wl_init_input") {
      // This is the stream for A
      i = 0;
    } else if (weight_stream_id == "wl_init_input/1") {
      // This is the stream for B
      i = 1;
    } else {
      throw error("Unexpected weight_stream_id: " + weight_stream_id);
    }

    void *grad_dst = raw_grads_out[i].data();
    auto grad_size = raw_grads_out[i].size() * sizeof(float);
    session->connectStreamToCallback(grad_stream_id,
                                     [grad_dst, grad_size](void *g) {
                                       std::memcpy(grad_dst, g, grad_size);
                                     });

    void *weight_src = dummy_data[i].data();
    auto weight_size = dummy_data[i].size() * sizeof(float);
    session->connectStreamToCallback(weight_stream_id,
                                     [weight_src, weight_size](void *w) {
                                       std::memcpy(w, weight_src, weight_size);
                                     });
  }

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
  popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
  std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                         {B_id, B_wrapper}};

  popart::StepIO stepio(inputs, anchors);

  session->optimizerFromHost();
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
                                raw_grads_out[0].begin(),
                                raw_grads_out[0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(v_B_grad.begin(),
                                v_B_grad.end(),
                                raw_grads_out[1].begin(),
                                raw_grads_out[1].end());

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
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<float> fdis(-4, 4);

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

  bder->addOutputTensor(G_id);

  // l1 loss with penalty term, will be applied to C
  float lossLambda = 0.26;

  auto proto      = bder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("ALL");
  // one batch per step
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{G_id, art}});

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  auto opts          = SessionOptions();
  opts.hostAllReduce = true;

  // training info
  auto optimizer = SGD({{"defaultLearningRate", {0.01, false}}});
  std::unique_ptr<Loss> l1_loss(
      new L1Loss(G_id, "l1LossVal", lossLambda, ReductionType::SUM));
  std::vector<Loss *> losses{l1_loss.get()};

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      losses,
      optimizer,
      cpuDevice,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::DEFAULT));

  // prepare the anchors. We have the output C,
  std::vector<float> raw_G_out(G_info.nelms());
  popart::NDArrayWrapper<float> G_wrapper(raw_G_out.data(), G_info.shape());

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {G_id, G_wrapper},
  };

  session->prepareDevice();

  const auto &ir = session->getIr();
  auto ops       = ir.getOpSchedule({});
  std::vector<Op *> partial_op_schedule;
  for (auto op : ops) {
    if (dynamic_cast<HostReduceGradCopyOp *>(op) ||
        dynamic_cast<HostSGD0VarUpdate *>(op)) {
      partial_op_schedule.push_back(op);
    }
  }

  // first 4 should be gradient copies, then followed by var copies
  for (int i = 0; i < 4; ++i) {
    auto asHostReduce =
        dynamic_cast<HostReduceGradCopyOp *>(partial_op_schedule[i]);
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
  for (const auto &gv : session->getGradAndVarStreamIds()) {
    const auto &grad_stream_id   = gv.first;
    const auto &weight_stream_id = gv.second;

    session->connectStreamToCallback(
        grad_stream_id, [&callback_handles, grad_stream_id](void *g) {
          callback_handles.push_back(grad_stream_id);
        });

    session->connectStreamToCallback(
        weight_stream_id, [&callback_handles, weight_stream_id](void *w) {
          callback_handles.push_back(weight_stream_id);
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

  session->optimizerFromHost();
  session->weightsFromHost();
  session->run(stepio);

  // Check that the callbacks are executed in the correct order (all gradients,
  // then all weights)
  for (int i = 0; i < 4; ++i) {
    // "gr_" from gradientStoreStreamId
    BOOST_CHECK(callback_handles[i].substr(0, 3) == "gr_");
  }
  for (int i = 4; i < 8; ++i) {
    // "wl_" from weightLoadStreamId
    BOOST_CHECK(callback_handles[i].substr(0, 3) == "wl_");
  }
}
