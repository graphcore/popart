#define BOOST_TEST_MODULE SoftmaxGradDirectTest0

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(SoftmaxGradDirect0) {
  // (label), (probs) -> [NLLGrad]
  // [NllGrad] -> (d_probs)
  // (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts)
  //
  // should become
  // (label), (probs) -> [SoftmaxGradDirect] -> (d_acts)
  //
  // but ONLY if "they're" on the same IPUs
  //
  // If NlllWithSoftmaxGradDirect pattern is enabled,
  // and since the loss is anchored, this should become
  //
  // (label), (probs) -> [NlllWithSoftmaxGradDirect] -> (loss), (d_acts)

  auto test = [](bool sameIPU, bool enableNlllWithSoftmaxGradDirect) {
    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{2, 2}};
    TensorInfo labelInfo{"INT32", std::vector<int64_t>{2}};

    auto input1 = builder->addInputTensor(inInfo);

    // This tensor is NOT part of the ONNX model, losses
    // are kept separate. It's shape is provided with InputShapeInfo
    TensorId input2 = "labelId";

    auto identOut   = aiOnnx.identity({input1});
    auto softmaxOut = aiOnnx.softmax({identOut});

    if (sameIPU == false) {
      builder->virtualGraph(identOut, 2);
      builder->virtualGraph(softmaxOut, 2);
    }

    builder->addOutputTensor(softmaxOut);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    // Add the last tensor, and the 3rd tensor as anchors
    auto art      = AnchorReturnType("ALL");
    auto dataFlow = DataFlow(
        1, {{reservedGradientPrefix() + input1, art}, {"nllLossVal", art}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{
        new NllLoss(softmaxOut, input2, "nllLossVal", ReductionType::SUM)};

    if (sameIPU == false) {
      losses[0]->virtualGraph(1);
    }

    auto opts = SessionOptions();
    if (sameIPU == false) {
      opts.virtualGraphMode = VirtualGraphMode::Manual;
    }

    // No .dot files will be written
    opts.dotChecks = {};

    InputShapeInfo inputInfo{};
    inputInfo.add(input2, labelInfo);

    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    auto patterns = Patterns({PreAliasPatternType::PREUNIREPL,
                              PreAliasPatternType::SOFTMAXGRADDIRECT});
    patterns.enableNlllWithSoftMaxGradDirect(enableNlllWithSoftmaxGradDirect);

    Ir ir;
    ir.prepare({modelProto,
                inputInfo,
                dataFlow,
                losses,
                &optimizer,
                *cpuDevice,
                opts,
                patterns});

    // Check the ir

    // NllGradOp and SoftmaxGradOp should have been replaced with
    // SoftmaxGradDirectOp, but ONLY if same IPUs
    if (sameIPU == true && enableNlllWithSoftmaxGradDirect == false) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    } else if (sameIPU == false && enableNlllWithSoftmaxGradDirect == false) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    }

    // NllLoss, NllGradOp and SoftmaxGradOp should have been replaced with
    // NlllWithSoftmaxGradDirectOp, but ONLY if same IPUs
    if (sameIPU == true && enableNlllWithSoftmaxGradDirect == true) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 1);
    } else if (sameIPU == false && enableNlllWithSoftmaxGradDirect == true) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    }
  };

  test(true, false);
  test(false, false);

  test(true, true);
  test(false, true);
}
