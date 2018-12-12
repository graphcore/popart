#define BOOST_TEST_MODULE ViewChangingTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
// needed for getting model from string
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
// needed for ConstVoidData
#include <poponnx/tensordata.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(ViewChangingTest_Reshape0) {

  // We will reshape a tensor from rank-4:
  Shape inShape = {2, 5, 3, 4};
  // to rank-2:
  Shape outShape = {10, 12};
  // Note above that the total number elements of the tensor remains 120

  Shape outShapeSize = {static_cast<int64_t>(outShape.size())};
  TensorInfo inInfo{"FLOAT", inShape};
  ConstVoidData outShapeData = {outShape.data(), {"INT64", outShapeSize}};

  // Build an onnx model
  auto builder    = Builder::create();
  auto newShapeId = builder->addInitializedInputTensor(outShapeData);
  auto inId       = builder->addInputTensor(inInfo);
  auto outId      = builder->reshape({inId, newShapeId});
  builder->addOutputTensor(outId);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(outId, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {}, // no const tensors
              {}, // no SessionOptions
              Patterns({PatternType::POSTNREPL})});

  // Check the ir
  // 1) that the Reshape Op is present,
  BOOST_CHECK(ir.opsOfType(OpType::RESHAPE).size() == 1);
  // 2) that the shape of the output tensor is as specified.
  BOOST_CHECK(ir.getTensors().get(outId)->info.shape() == outShape);
}
