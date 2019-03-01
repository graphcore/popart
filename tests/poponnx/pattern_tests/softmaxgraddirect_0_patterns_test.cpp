#define BOOST_TEST_MODULE SoftmaxGradDirectTest0

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(SoftmaxGradDirect0) {
  // (label), (probs) -> [NLLGrad]
  // [NllGrad] -> (d_probs)
  // (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts)
  //
  // should become
  // (label), (probs) -> [SoftmaxGradDirect] -> (d_acts)
  //
  // but ONLY if "they're" on the same IPUs

  auto test = [](bool sameIPU) {
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
    auto art       = AnchorReturnType("ALL");
    auto dataFlow  = DataFlow(1,
                             {{softmaxOut, art},
                              {reservedGradientPrefix() + input1, art},
                              {"nllLossVal", art}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{new NllLoss(softmaxOut, input2, "nllLossVal")};

    if (sameIPU == false) {
      losses[0]->virtualGraph(1);
    }

    auto opts = SessionOptions();
    // No .dot files will be written
    opts.dotChecks = {};

    InputShapeInfo inputInfo{};
    inputInfo.add(input2, labelInfo);

    Ir ir;
    ir.prepare({modelProto,
                inputInfo,
                dataFlow,
                losses,
                &optimizer,
                opts,
                Patterns({PreAliasPatternType::PREUNIREPL,
                          PreAliasPatternType::SOFTMAXGRADDIRECT})});

    // Check the ir
    // NllGradOp and SoftmaxGradOp should have been replaced with
    // SoftmaxGradDirectOp, but ONLY if different IPUs

    if (sameIPU == true) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          1);
    } else {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
    }
  };
  test(true);
  test(false);
}
