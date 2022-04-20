// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE HostOpTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popart/builder.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "popart/attributes.hpp"
#include "popart/dataflow.hpp"
#include "popart/datatype.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/any.hpp"

namespace popart {
class IArray;

namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace {

struct FunctionData {
  std::vector<const popart::TensorInfo *> inputs, outputs;
  std::string functionHandle;

  void asAttribute(popart::Attributes &attrs) {
    auto i = reinterpret_cast<popart::Attributes::Int>(this);
    attrs.setAttribute("function", i);
  }

  void asAttribute(std::map<std::string, popart::any> &attrs) {
    auto i = reinterpret_cast<popart::Attributes::Int>(this);
    attrs.emplace("function", i);
  }

  static FunctionData *getFrom(const popart::Attributes &attrs) {
    auto i = attrs.getAttribute<popart::Attributes::Int>("function");
    FunctionData *data = reinterpret_cast<FunctionData *>(i);
    return data;
  }
};

/*
  Popart custom op which uses the metadata gathered by the compiler to setup
  poplar tensors and copy into/from them from/to host.
*/
class HostOp final : public popart::Op {
public:
  HostOp(const popart::OperatorIdentifier &_opid,
         const popart::Op::Settings &settings_,
         FunctionData *info)
      : popart::Op(_opid, settings_), data(info) {}

  static std::unique_ptr<HostOp> create(const popart::OpCreatorInfo &info);

  // Configure the output popart Tensor
  void setup() override {
    // Tell popart what the output should look like.
    for (std::uint32_t i = 0; i < data->outputs.size(); ++i) {
      outInfo(i) = *data->outputs[i];
    }
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<HostOp>(*this);
  }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }
  bool hasSideEffect() const override { return true; }

  FunctionData *data;

  static const popart::OperatorIdentifier identifier;
  static const popart::OpCreator<HostOp> creator;
};

std::unique_ptr<HostOp> HostOp::create(const popart::OpCreatorInfo &info) {
  FunctionData *data = FunctionData::getFrom(info.attributes);
  return std::make_unique<HostOp>(info.opid, info.settings, data);
}

class HostOpx : public popart::popx::Opx {
public:
  HostOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<HostOp>(op, HostOp::identifier);

    data = dynamic_cast<HostOp *>(op)->data;
  }

  void grow(poplar::program::Sequence &sequence) const override {
    poplar::Graph &graph = this->graph();

    // Get basic op info from metadata.
    const std::uint32_t num_inputs  = data->inputs.size();
    const std::uint32_t num_outputs = data->outputs.size();

    // For each input create the FIFO and copy from it into the poplar tensor
    // popart has already created/
    std::vector<poplar::Graph::HostFunctionArgument> input_args;
    std::vector<poplar::Tensor> inputs;
    inputs.reserve(num_inputs);
    input_args.reserve(num_inputs);
    for (std::uint32_t input_index = 0; input_index < num_inputs;
         ++input_index) {
      // poplar::Tensor from popart.
      poplar::Tensor input_tensor = getInTensor(input_index);
      inputs.push_back(input_tensor);
      input_args.emplace_back(input_tensor.elementType(),
                              input_tensor.numElements());
    }

    std::vector<poplar::Graph::HostFunctionArgument> output_args;
    std::vector<poplar::Tensor> outputs;
    outputs.reserve(num_outputs);
    output_args.reserve(num_outputs);
    for (std::uint32_t output = 0; output < num_outputs; ++output) {
      const poplar::Type type = popart::popx::popType(*data->outputs[output]);

      std::vector<size_t> shape = data->outputs[output]->shape_szt();

      // Add the poplar tensor.
      poplar::Tensor output_tensor = graph.addVariable(
          type,
          shape,
          poplar::VariableMappingMethod::LINEAR,
          data->functionHandle + "::out" + std::to_string(output));

      outputs.push_back(output_tensor);
      output_args.emplace_back(output_tensor.elementType(),
                               output_tensor.numElements());

      // Tell popart this is the output.
      setOutTensor(output, output_tensor);
    }

    poplar::HostFunction hf =
        graph.addHostFunction(data->functionHandle, input_args, output_args);
    sequence.add(poplar::program::Call(hf, inputs, outputs));
  }

  FunctionData *data;

  static const popart::popx::OpxCreator<HostOpx> creator;
};

const popart::OperatorIdentifier HostOp::identifier{"test.custom_ops",
                                                    "HostOp",
                                                    popart::OpVersion{1},
                                                    popart::NumInputs{1, 64},
                                                    1};

const popart::OpCreator<HostOp> HostOp::creator{{{HostOp::identifier, {}}},
                                                &HostOp::create,
                                                true};

const popart::popx::OpxCreator<HostOpx> HostOpx::creator(HostOp::identifier);

static popart::RegisterShapeInferenceFunction hostOpShapeInference(
    HostOp::identifier,
    [](popart::ShapeInferenceContext &ctx) {
      // Get the stream info from the attribute map we passed to create the op.
      auto *data = FunctionData::getFrom(ctx.getAttributes());

      // Tell popart what the output should look like.
      for (std::uint32_t i = 0; i < data->outputs.size(); ++i) {
        ctx.outInfo(i) = *data->outputs[i];
      }
    });

} // namespace

BOOST_AUTO_TEST_CASE(CPUCallback_test) {
  using namespace popart;
  auto builder = Builder::create();

  TensorInfo fpArray{DataType::FLOAT, Shape{10}};

  auto a = builder->addInputTensor(fpArray),
       b = builder->addInputTensor(fpArray);

  FunctionData data;
  data.inputs  = {&fpArray, &fpArray};
  data.outputs = {&fpArray};

  std::map<std::string, popart::any> attributes_map;
  data.asAttribute(attributes_map);

  auto c =
      builder->customOp(HostOp::identifier, 2, {a, b}, 1, attributes_map)[0];
  builder->addOutputTensor(c);

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  // inputs:
  std::vector<float> in_a(10, 1.0f), in_b(10, 2.0f), out_c(10);
  popart::NDArrayWrapper<float> a_wrapper{in_a.data(), fpArray},
      b_wrapper{in_b.data(), fpArray}, c_wrapper{out_c.data(), fpArray};

  std::map<popart::TensorId, popart::IArray &> inputs{{a, a_wrapper},
                                                      {b, b_wrapper}},
      outputs{{c, c_wrapper}};

  auto device  = createTestDevice(TEST_TARGET);
  auto session = InferenceSession::createFromOnnxModel(
      builder->getModelProto(),
      DataFlow(1, {{c, AnchorReturnType("All")}}),
      device,
      InputShapeInfo(),
      SessionOptions(),
      Patterns(PatternsLevel::Default));
  session->prepareDevice();

  session->connectHostFunction(data.functionHandle,
                               [](const void *const *ins,
                                  size_t numIns,
                                  void *const *outs,
                                  size_t numOuts) {
                                 auto *a = static_cast<const float *>(ins[0]),
                                      *b = static_cast<const float *>(ins[1]);
                                 auto *c = static_cast<float *>(outs[0]);
                                 for (unsigned i = 0; i < 10; i++) {
                                   c[i] = a[i] + b[i];
                                 }
                               });

  StepIO stepio(inputs, outputs);
  session->run(stepio);

  std::vector<float> expected(in_a.size(), in_a.front() + in_b.front());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected.begin(), expected.end(), out_c.begin(), out_c.end());
}
