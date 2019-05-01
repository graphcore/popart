#define BOOST_TEST_MODULE PatternsTest

#include <boost/test/unit_test.hpp>
#include <vector>

#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/transforms/prune.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(PruneTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape{"FLOAT", std::vector<int64_t>{2}};

  auto i1 = builder->addInputTensor(shape);
  std::vector<TensorId> tensorIds{i1};
  // Create some ops which are used
  for (int i = 0; i < 6; i++) {
    auto x = aiOnnx.identity({tensorIds[tensorIds.size() - 1]});
    tensorIds.push_back(x);
  }
  builder->addOutputTensor(tensorIds.back());

  // Add some operations which should be pruned
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  aiOnnx.add({i1, i1});
  auto i2 = aiOnnx.add({i1, i1});
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2}, "test_add");
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2});
  aiOnnx.add({i2, i2});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow = DataFlow(1,
                           {{tensorIds.back(), AnchorReturnType("ALL")},
                            {tensorIds[2], AnchorReturnType("ALL")}});

  Ir ir;
  ir.setOnnxModel(modelProto);
  ir.setDataFlow(dataFlow);
  ir.registerInputTensors();
  ir.constructForwards();
  ir.applyTransform(Prune::id(), ir.getMainGraph());

  // All but the original 6 operations should be pruned
  BOOST_CHECK(ir.getMainGraphOps().size() == 6);
}
