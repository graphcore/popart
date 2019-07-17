#define BOOST_TEST_MODULE Train0MatmulTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

// Test:
// C = matmul (A, B) where both A and B are weight matrices,
// loss = lambda*|C|_1
BOOST_AUTO_TEST_CASE(DatalessTrainingMatmul) {

  // genPdf : generate of dot file pdf of the training computation
  auto test = [](bool genPdf) {
    using namespace poponnx;

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
    auto dataFlow      = DataFlow(batchesPerStep,
                             {{C_id, art},
                              {reservedGradientPrefix() + A_id, art},
                              {reservedGradientPrefix() + B_id, art}});

    auto cpuDevice =
        poponnx::DeviceManager::createDeviceManager().createCpuDevice();

    auto opts            = SessionOptions();
    opts.enableOutlining = true;
    if (genPdf) {
      opts.firstDotOp = 0;
      opts.finalDotOp = 100;
      opts.dotChecks.insert(DotCheck::FINAL);
      opts.logDir = ".";
    }

    // training info
    float learnRate = 0.321;
    auto optimizer  = ConstSGD(learnRate);
    std::unique_ptr<Loss> l1_loss(
        new L1Loss(C_id, "l1LossVal", lossLambda, ReductionType::SUM));
    std::vector<Loss *> losses{l1_loss.get()};

    auto session = poponnx::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        losses,
        optimizer,
        cpuDevice,
        poponnx::InputShapeInfo(),
        opts,
        poponnx::Patterns(PatternsLevel::DEFAULT));

    // prepare the anchors. We have the output C,
    std::vector<float> raw_C_out(C_info.nelms());
    poponnx::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

    // the gradient of A,
    std::vector<float> raw_A_grad_out(A_info.nelms());
    poponnx::NDArrayWrapper<float> A_grad_wrapper(raw_A_grad_out.data(),
                                                  A_info.shape());
    // and the gradient of B.
    std::vector<float> raw_B_grad_out(B_info.nelms());
    poponnx::NDArrayWrapper<float> B_grad_wrapper(raw_B_grad_out.data(),
                                                  B_info.shape());

    std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
        {C_id, C_wrapper},
        {reservedGradientPrefix() + A_id, A_grad_wrapper},
        {reservedGradientPrefix() + B_id, B_grad_wrapper}};

    session->prepareDevice();

    // inputs:
    poponnx::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
    poponnx::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
    std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{A_id, A_wrapper},
                                                             {B_id, B_wrapper}};

    poponnx::StepIO stepio(inputs, anchors);

    session->weightsFromHost();
    session->run(stepio);

    // confirm the gradient values agree (exactly...)
    BOOST_CHECK_EQUAL_COLLECTIONS(
        v_C_data.begin(), v_C_data.end(), raw_C_out.begin(), raw_C_out.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_A_grad.begin(),
                                  v_A_grad.end(),
                                  raw_A_grad_out.begin(),
                                  raw_A_grad_out.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_B_grad.begin(),
                                  v_B_grad.end(),
                                  raw_B_grad_out.begin(),
                                  raw_B_grad_out.end());

    // we will read the updated weights back, and check that they are correct
    std::vector<float> v_A_updated_baseline = v_A_init;
    std::vector<float> v_B_updated_baseline = v_B_init;
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        v_B_updated_baseline[k * N + n] -= learnRate * v_B_grad[k * N + n];
      }

      for (int m = 0; m < M; ++m) {
        v_A_updated_baseline[m * K + k] -= learnRate * v_A_grad[m * K + k];
      }
    }
    WeightsIO weightsRead;
    // to be readback:
    std::vector<float> A_readback(A_info.nelms(), -9.0f);
    std::vector<float> B_readback(B_info.nelms(), -99.0f);
    weightsRead.insert(A_id, {A_readback.data(), A_info});
    weightsRead.insert(B_id, {B_readback.data(), B_info});

    session->weightsToHost();
    session->readWeights(weightsRead);

    BOOST_CHECK_EQUAL_COLLECTIONS(v_A_updated_baseline.begin(),
                                  v_A_updated_baseline.end(),
                                  A_readback.begin(),
                                  A_readback.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(v_B_updated_baseline.begin(),
                                  v_B_updated_baseline.end(),
                                  B_readback.begin(),
                                  B_readback.end());

    // dot -Tpdf -o final.pdf final.dot
    if (genPdf) {
      for (auto check : opts.dotChecks) {
        auto dot_string = getDotCheckString(check);
        std::stringstream command_ss;
        command_ss << "dot "
                   << " -Tpdf "
                   << " -o "
                   << io::appendDirFn(opts.logDir, dot_string + ".pdf") << " "
                   << io::appendDirFn(opts.logDir, dot_string + ".dot");
        std::string command = command_ss.str();
        int ran             = std::system(command.c_str());
        std::cout << command << " returned with status " << ran << std::endl;
      }
    }
  };

  // should only be true for debugging (on a machine with dot program)
  bool genPdf = false;
  test(genPdf);
}
