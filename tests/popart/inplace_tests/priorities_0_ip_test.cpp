#define BOOST_TEST_MODULE Priorities0InplaceTest

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

BOOST_AUTO_TEST_CASE(NegPriorities_concat0) {

  auto runTest = [](float priorityValue) {
    // Two Ops with their inplacing priorities either
    // set negative (so they should NOT be inplaced)
    // or kept positive (so they SHOULD be inplaced if possible)
    //
    // clang-format off
    //
    //
    //                          priority might be negative!
    // in0 -|                       ^
    //      |                       |
    // in1 -|- [Concat] - x1 --|    |
    //      |                  |    |
    // in2 -|--|               |- [Concat] - x3 - [Relu] - o1 -|
    //         |- [Concat] x2 -|                              /
    //         |                                             /
    //         |                                            /
    // in3 ----|                                          [Concat] ---- o2 
    //                                                    |              |
    // in4 ---------- [Relu] -- c3 -----------------------|         [ReduceSum]
    //                   ^                                               |
    //                   |                                              out
    //                   |
    //                 priority might be negative!
    //
    //
    // modification happens iff modifyPriorities is true
    //
    // clang-format on

    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo shape0{"FLOAT", std::vector<int64_t>{3}};
    auto in0 = builder->addInputTensor(shape0);
    auto in1 = builder->addInputTensor(shape0);
    auto in2 = builder->addInputTensor(shape0);
    auto in3 = builder->addInputTensor(shape0);

    auto x1  = aiOnnx.concat({in0, in1, in2}, 0);
    auto x2  = aiOnnx.concat({in2, in3}, 0);
    auto x3  = aiOnnx.concat({x1, x2}, 0);
    auto o1  = aiOnnx.relu({x3});
    auto in4 = builder->addInputTensor(shape0);
    auto c3  = aiOnnx.relu({in4});
    auto o2  = aiOnnx.concat({o1, c3}, 0);
    auto out = aiOnnx.reducesum({o2});
    builder->addOutputTensor(out);

    builder->setInplacePreferences(x3, {{"ConcatInplace", priorityValue}});
    // TODO if  these 2 priorities are large (>10.0f), this test fails. See task
    // T9423
    builder->setInplacePreferences(o2, {{"ConcatInplace", 9.0f}});
    builder->setInplacePreferences(o1, {{"ReluInplace", 9.5f}});

    builder->setInplacePreferences(c3, {{"ReluInplace", priorityValue}});

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{
        new L1Loss(out, "l1LossVal", 0.1, ReductionType::SUM)};
    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                &optimizer,
                *cpuDevice,
                {},
                Patterns(PatternsLevel::NONE).enableInPlace(true)});

    // .... Tests ....
    // irrespective of priority x1, x2, o2 should be inplace concats,
    // and x3 should only be inplace if priority > 0.
    std::vector<TensorId> inplaceConcatIds    = {x1, x2, o2};
    std::vector<TensorId> notInplaceConcatIds = {};
    // o1 should always be inplace, c3 only sometimes.
    std::vector<TensorId> inplaceReluIds    = {o1};
    std::vector<TensorId> notInplaceReluIds = {};

    if (priorityValue > 0) {
      inplaceConcatIds.push_back(x3);
      inplaceReluIds.push_back(c3);
    } else {
      notInplaceConcatIds.push_back(x3);
      notInplaceReluIds.push_back(c3);
    }

    for (TensorId id : inplaceConcatIds) {
      BOOST_CHECK(ir.getMainGraphTensors().get(id)->getProducer()->opid.type ==
                  "ConcatInplace");
    }

    for (TensorId id : inplaceReluIds) {
      BOOST_CHECK(ir.getMainGraphTensors().get(id)->getProducer()->opid.type ==
                  "ReluInplace");
    }

    for (TensorId id : notInplaceConcatIds) {
      BOOST_CHECK(ir.getMainGraphTensors().get(id)->getProducer()->opid.type ==
                  "Concat");
    }

    for (TensorId id : notInplaceReluIds) {
      BOOST_CHECK(ir.getMainGraphTensors().get(id)->getProducer()->opid.type ==
                  "Relu");
    }
  };

  // SHOULD all be inplaced:
  runTest(1e07f);
  runTest(1e-5f);

  // should NOT all be inplaced:
  runTest(-1.0f);
}
