// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PrngTest

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <testdevice.hpp>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <poprand/RandomGen.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/datatype.hpp"
#include "popart/half.hpp"
#include "popart/iarray.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/vendored/any.hpp"
#include "popart/voiddata.hpp"
#include "random_util.hpp"

namespace popart {
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

using namespace popart;
using namespace popart::popx;

namespace {

static constexpr int seed = 11583;

// Helper function to get a float16 array as uint16_ts.
std::vector<uint16_t> getRandomFloat16Data(size_t numElems) {

  static DefaultRandomEngine eng(seed);
  static UniformRealDistribution<float> unitDist(-1.0f, 1.0f);

  std::vector<uint16_t> wDataVecAsUint16(numElems, 0);
  for (size_t i = 0; i < numElems; i++) {
    wDataVecAsUint16.at(i) = floatToHalf(unitDist(eng));
  }

  return wDataVecAsUint16;
}

// Helper class to pass float16 buffers.
class Float16Buffer : public IArray {
public:
  Float16Buffer(void *_data, Shape _shape)
      : IArray(), mdata(_data), mshape(_shape) {}

  virtual ~Float16Buffer() {}

  virtual void *data() { return mdata; }

  virtual DataType dataType() const { return DataType::FLOAT16; }

  virtual std::size_t rank() const { return mshape.size(); }

  virtual int64_t dim(size_t index) const { return mshape.at(index); }

  virtual std::size_t nelms() const {
    return std::accumulate(mshape.begin(),
                           mshape.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }

  virtual const Shape shape() const { return mshape; }

private:
  void *mdata;
  Shape mshape;
};

} // anonymous namespace

BOOST_AUTO_TEST_CASE(
    stochastic_rounding_determinism_across_replicas_weird_prng_op) {

  // Using custom ops which affect the PRNG state in a data-dependent way, this
  // integration test ensures that with stochastic rounding enabled a model
  // weights do not diverge on replicas when these custom ops are used.

  constexpr int numReplicas    = 2;
  constexpr int64_t inputSize  = 10;
  constexpr int64_t labelSize  = 5;
  constexpr int64_t numBatches = 1;

  static OperatorIdentifier weirdPrngOpOperatorIdentifier(
      "TestOps", "WeirdPrngOp", 1);

  // An op which puts input to output but with data-dependent PRNG behaviour.
  // It will ensure that, when given different data the PRNG state is affected
  // differently on each replica, which will result in drifting weights unless
  // the PRNG state is explicitly managed to avoid this in PopART.
  class WeirdPrngOp : public Op {
  public:
    WeirdPrngOp(const Op::Settings &settings)
        : Op(weirdPrngOpOperatorIdentifier, settings) {}

    void setup() override {
      // Straight through, input to output.
      for (const auto &entry : input->tensorIdMap()) {
        outInfo(entry.first) = inInfo(entry.first);
      }
    }

    std::unique_ptr<Op> clone() const override {
      return std::make_unique<WeirdPrngOp>(*this);
    }

    float getSubgraphValue() const override { return getLowSubgraphValue(); }
  };

  // ONNX op definition for WeirdPrngOp so we can use it in the ONNX builder.
  OpDefinition weirdPrngOpDef(
      {OpDefinition::Inputs({{"input", {DataType::FLOAT16}}}),
       OpDefinition::Outputs({{"output", {DataType::FLOAT16}}}),
       OpDefinition::Attributes({})});

  // Semantics for WeirdPrngOp.
  class WeirdPrngOpx : public PopOpx {
  public:
    WeirdPrngOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

    void grow(snap::program::Sequence &prog) const {

      // Do some PRNG stuff when input0's first element is 0.0. To do this,
      // first we need to isolate the first element of input0, then map it to a
      // boolean and put it in a condition tensor.
      BOOST_REQUIRE_MESSAGE(op_p->input->hasIndex(0),
                            "Expected WeirdPrngOpx to have input 0");
      BOOST_REQUIRE_MESSAGE(op_p->input->tensor(0)->info.dataType() ==
                                DataType::FLOAT16,
                            "Expected WeirdPrngOpx input 0 to be FLOAT16");

      const uint16_t zeroHalfValue = floatToHalf(0.0f);
      auto zeroHalfPoplarConst     = graph().getPoplarGraph().addConstantHalf(
          poplar::HALF, {}, zeroHalfValue);
      graph().getPoplarGraph().setTileMapping(zeroHalfPoplarConst, 0);

      auto in0 = getInTensor(0);
      std::vector<size_t> in0Begin(in0.rank(), 0);
      std::vector<size_t> in0End(in0.rank(), 1);
      auto in0FirstElem = in0.slice(in0Begin, in0End);

      auto condition = popops::eq(graph().getPoplarGraph(),
                                  in0FirstElem.getPoplarTensor(),
                                  zeroHalfPoplarConst,
                                  prog.getPoplarSequence());

      condition = condition.reshape({});

      poplar::program::Sequence thenProg({});
      poplar::program::Sequence elseProg({});

      // Doing some poprand stuff should mess with the PRNG state.
      poprand::uniform(graph().getPoplarGraph(),
                       nullptr,
                       0u,
                       in0.getPoplarTensor(),
                       poplar::FLOAT,
                       0.0f,
                       1.0f,
                       thenProg);

      prog.getPoplarSequence().add(
          poplar::program::If(condition, thenProg, elseProg));
      for (auto entry : op_p->output->tensorMap()) {
        setOutTensor(entry.first, getInTensor(entry.first));
      }
    }
  };

  // Register the Opx.
  OpxCreator<WeirdPrngOpx> weirdPrngOpxCreator(weirdPrngOpOperatorIdentifier);

  auto builder = Builder::create();

  OpManager::registerOp({{"TestOps", "WeirdPrngOp", 1},
                         false,
                         weirdPrngOpDef,
                         [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
                           return std::unique_ptr<WeirdPrngOp>(
                               new WeirdPrngOp(info.settings));
                         }});

  // Add weight tensor (as halfs).
  TensorInfo wInfo{"FLOAT16", std::vector<int64_t>{1, inputSize, labelSize}};
  std::vector<uint16_t> wDataVecAsUint16 = getRandomFloat16Data(wInfo.nelms());
  ConstVoidData wData{reinterpret_cast<void *>(wDataVecAsUint16.data()), wInfo};
  auto w = builder->addInitializedInputTensor(wData);

  // Add input tensor for data input.
  TensorInfo inputsInfo{"FLOAT16", std::vector<int64_t>{1, 1, inputSize}};
  std::vector<uint16_t> inputsDataVecAsUint16 =
      getRandomFloat16Data(numBatches * numReplicas * inputsInfo.nelms());

  // IMPORTANT: We change the first element of each 'inputs' tensor to match
  // the half value of the replica index. This way, replica 0 will trigger
  // a poprand call and other replicas will not.

  for (size_t b = 0; b < numBatches; ++b) {
    for (size_t r = 0; r < numReplicas; ++r) {
      inputsDataVecAsUint16.at((b * numReplicas + r) * inputsInfo.nelms()) =
          floatToHalf((float)r);
    }
  }

  ConstVoidData inputsData{reinterpret_cast<void *>(wDataVecAsUint16.data()),
                           inputsInfo};
  auto inputs = builder->addInputTensor(inputsInfo);

  // Add input tensor for labels input.
  TensorInfo labelsInfo{"FLOAT16", std::vector<int64_t>{labelSize}};
  std::vector<uint16_t> labelsDataVecAsUint16 =
      getRandomFloat16Data(numBatches * numReplicas * labelsInfo.nelms());
  ConstVoidData labelsData{reinterpret_cast<void *>(wDataVecAsUint16.data()),
                           labelsInfo};
  auto labels = builder->addInputTensor(labelsInfo);

  // Build the fwd graph.
  auto fwd1 =
      builder->customOp({"TestOps", "WeirdPrngOp", 1}, 1, {inputs}, 1, {})[0];
  auto fwd2 = builder->aiOnnxOpset11().matmul({fwd1, w});

  // Calculate the loss.
  auto err  = builder->aiOnnxOpset11().sub({fwd2, labels});
  auto loss = builder->aiGraphcoreOpset1().l1loss({err}, 0.1);

  // Run it, getting the weights from each replica after the final step.
  auto anchorMap = AnchorReturnTypeMap{{loss, AnchorReturnType("Final")},
                                       {w, AnchorReturnType("Final")}};

  DataFlow dataFlow(numBatches, anchorMap);
  auto optimizer = ConstSGD(0.1);
  auto device    = createTestDevice(TEST_TARGET, numReplicas);

  SessionOptions opts;
  opts.enableStochasticRounding = true;
  opts.enableReplicatedGraphs   = true;
  opts.replicatedGraphCount     = numReplicas;

  // TODO (T48752): Remove.
  opts._enableRngStateManagement = true;

  // Uncomment when profiling (see ~T32263~).
  // opts.engineOptions["target.syncReplicasIndependently"] = "true";

  auto session = TrainingSession::createFromOnnxModel(
      builder->getModelProto(),
      dataFlow,
      loss,
      optimizer,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();
  session->weightsFromHost();
  session->setRandomSeed(seed);

  std::vector<uint16_t> lossAnchorData(1, 0);
  Float16Buffer lossAnchorIArray(
      reinterpret_cast<void *>(lossAnchorData.data()), {numReplicas});

  std::vector<uint16_t> wAnchorData(inputSize * labelSize * numReplicas, 0);
  Float16Buffer wAnchorIArray(reinterpret_cast<void *>(wAnchorData.data()),
                              {numReplicas, wInfo.nelms()});

  Float16Buffer inputsIArray(
      reinterpret_cast<void *>(inputsDataVecAsUint16.data()),
      {numBatches, numReplicas, inputsInfo.nelms()});
  Float16Buffer labelsIArray(
      reinterpret_cast<void *>(labelsDataVecAsUint16.data()),
      {numBatches, numReplicas, labelsInfo.nelms()});

  StepIO stepio{{{inputs, inputsIArray}, {labels, labelsIArray}},
                {{loss, lossAnchorIArray}, {w, wAnchorIArray}}};

  session->run(stepio);

  // Now check if all replicas agree on all weight values.
  for (int replica_i = 0; replica_i < numReplicas; ++replica_i) {
    for (int replica_j = replica_i + 1; replica_j < numReplicas; ++replica_j) {

      for (int m = 0; m < wInfo.nelms(); ++m) {
        auto data_i = wAnchorData[replica_i * wInfo.nelms() + m];
        auto data_j = wAnchorData[replica_j * wInfo.nelms() + m];
        std::stringstream ss;
        ss << "Replica " << replica_i << " and " << replica_j
           << " unexpectedly disagree on weight ('" << w << "') at index " << m
           << " after " << numBatches << " batches (" << halfToFloat(data_i)
           << " != " << halfToFloat(data_j) << ")\n\n";

        ss << "  with replica " << replica_i << " weights being\n ";
        for (size_t i = 0; i < wInfo.nelms(); ++i) {
          ss << halfToFloat(wAnchorData[replica_i * wInfo.nelms() + i]) << " ";
        }
        ss << "\n\n";

        ss << "  with replica " << replica_j << " weights being\n ";
        for (size_t i = 0; i < wInfo.nelms(); ++i) {
          ss << halfToFloat(wAnchorData[replica_j * wInfo.nelms() + i]) << " ";
        }
        ss << "\n";

        BOOST_REQUIRE_MESSAGE(data_i == data_j, ss.str());
      }
    }
  }
}
