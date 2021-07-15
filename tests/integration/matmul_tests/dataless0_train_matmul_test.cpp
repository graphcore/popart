// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Train0MatmulTest

#include <../random_util.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

// Test:
// C = matmul (A, B) where both A and B are weight matrices,
// loss = lambda*|C|_1
BOOST_AUTO_TEST_CASE(DatalessTrainingMatmul) {

  // genPdf : generate of dot file pdf of the training computation
  auto test = [](bool genPdf) {
    using namespace popart;

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
    auto l1          = bder->aiGraphcoreOpset1().l1loss(
        {C_id}, lossLambda, ReductionType::Sum);

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
    auto dataFlow      = DataFlow(batchesPerStep,
                             {{C_id, art},
                              {reservedGradientPrefix() + A_id, art},
                              {reservedGradientPrefix() + B_id, art}});

    auto device = popart::createTestDevice(TEST_TARGET);

    auto opts            = SessionOptions();
    opts.enableOutlining = true;
    if (genPdf) {
      opts.firstDotOp = 0;
      opts.finalDotOp = 100;
      opts.dotChecks.insert(DotCheck::Final);
      opts.logDir = "./dotfiles";
      if (!boost::filesystem::exists(opts.logDir)) {
        boost::filesystem::create_directories(opts.logDir);
      }
    }

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

    // prepare the anchors. We have the output C,
    std::vector<float> raw_C_out(C_info.nelms());
    popart::NDArrayWrapper<float> C_wrapper(raw_C_out.data(), C_info.shape());

    // the gradient of A,
    std::vector<float> raw_A_grad_out(A_info.nelms());
    popart::NDArrayWrapper<float> A_grad_wrapper(raw_A_grad_out.data(),
                                                 A_info.shape());
    // and the gradient of B.
    std::vector<float> raw_B_grad_out(B_info.nelms());
    popart::NDArrayWrapper<float> B_grad_wrapper(raw_B_grad_out.data(),
                                                 B_info.shape());

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {C_id, C_wrapper},
        {reservedGradientPrefix() + A_id, A_grad_wrapper},
        {reservedGradientPrefix() + B_id, B_grad_wrapper}};

    session->prepareDevice();

    // inputs:
    popart::NDArrayWrapper<float> A_wrapper(v_A_init.data(), A_info);
    popart::NDArrayWrapper<float> B_wrapper(v_B_init.data(), B_info);
    std::map<popart::TensorId, popart::IArray &> inputs = {{A_id, A_wrapper},
                                                           {B_id, B_wrapper}};

    popart::StepIO stepio(inputs, anchors);

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
                   << io::appendDirFn(opts.logDir, dot_string + "_r.dot");
        std::string command = command_ss.str();
        int ran             = std::system(command.c_str());
        std::cout << command << " returned with status " << ran << std::endl;
        BOOST_CHECK(ran == 0);
        auto pdfFileNames =
            io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".pdf");
        BOOST_CHECK(pdfFileNames.size() == 1);
      }
    }
  };

  // should only be true for debugging (on a machine with dot program)
  bool genPdf = false;
  test(genPdf);
}
