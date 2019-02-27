#define BOOST_TEST_MODULE RemoveUnusedInputTest0

#include <boost/test/unit_test.hpp>
#include <onnx/onnx_pb.h>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

// confirm that when prepareNodesFroTraining,
// batch-normalization has 5 outputs
BOOST_AUTO_TEST_CASE(Transformation_PrepareTrain0) {

  auto test = [](bool prepareTraining) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo info_d{"FLOAT", std::vector<int64_t>{4, 3, 32, 32}};
    TensorInfo info_c{"FLOAT", std::vector<int64_t>{3}};

    auto in0 = builder->addInputTensor(info_d);

    auto a0 = builder->addInputTensor(info_c);
    auto a1 = builder->addInputTensor(info_c);
    auto a2 = builder->addInputTensor(info_c);
    auto a3 = builder->addInputTensor(info_c);

    // in both cases, we set the number of outputs to 1
    unsigned num_outputs = 1;
    float eps            = 0.2;
    float momen          = 0.3;
    auto outs            = aiOnnx.batchnormalization(
        {in0, a0, a1, a2, a3}, num_outputs, eps, momen);
    builder->addOutputTensor(outs[0]);

    std::string proto = builder->getModelProto();

    GraphTransformer gt(proto);

    if (prepareTraining) {
      gt.prepareNodesForTraining();
    }
    auto postProto = io::getModelFromString(gt.getModelProto());

    // cofirm that the batch normalization has 5
    // outputs when the transformation has been run
    int visited = 0;
    for (auto node : postProto.graph().node()) {
      if (node.op_type() == "BatchNormalization") {
        BOOST_CHECK(node.output_size() == prepareTraining ? 5 : 1);
        ++visited;
      }
    }
    BOOST_CHECK(visited == 1);
  };

  test(true);
  test(false);
}
