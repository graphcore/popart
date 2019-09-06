#include <popart/opmanager.hpp>
#include <popart/transforms/transformbuilder.hpp>

namespace popart {

std::unique_ptr<Op>
TransformBuilder::createOp(const OperatorIdentifier &opid,
                           std::map<std::string, boost::any> attributes,
                           const std::string debugPrefix) {

  Attributes attr;
  for (auto attribute : attributes) {
    const std::type_info &tinfo = attribute.second.type();
    if (tinfo == typeid(Attributes::Int)) {
      auto value = boost::any_cast<Attributes::Int>(attribute.second);
      attr.setAttribute(attribute.first, value);
    } else if (tinfo == typeid(Attributes::Ints)) {
      auto value = boost::any_cast<Attributes::Ints>(attribute.second);
      attr.setAttribute(attribute.first, value);
    } else {
      throw error("Unsupported attribute value type {}", tinfo.name());
    }
  }
  return OpManager::createOp(opid, graph, debugPrefix, attr);
}

TensorId TransformBuilder::op(const OperatorIdentifier &_opid,
                              std::vector<TensorId> &inputs,
                              std::map<std::string, boost::any> attributes,
                              boost::optional<int64_t> virtualGraphId,
                              const std::string debugPrefix) {

  auto op = createOp(_opid, attributes, debugPrefix);

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->createAndConnectOutTensor(0, debugPrefix + "/out");

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }

  op->setup();
  auto _op = op.get();
  graph.moveIntoGraph(std::move(op));
  return _op->outTensor(0)->id;
}

void TransformBuilder::opWithOutput(
    const OperatorIdentifier &_opid,
    std::vector<TensorId> &inputs,
    std::map<std::string, boost::any> attributes,
    TensorId &out,
    boost::optional<int64_t> virtualGraphId,
    const std::string debugPrefix) {

  auto op = createOp(_opid, attributes, debugPrefix);

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->connectOutTensor(0, out);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }

  op->setup();
  graph.moveIntoGraph(std::move(op));
}

TransformBuilder::TransformBuilder(Graph &graph_) : graph(graph_) {}

TensorId TransformBuilder::concat(std::vector<TensorId> &inputs,
                                  boost::optional<int64_t> virtualGraphId,
                                  const std::string debugPrefix) {
  return op(Onnx::Operators::Concat_1,
            inputs,
            {{"axis", static_cast<int64_t>(0)}},
            virtualGraphId,
            debugPrefix + "/concat");
}

TensorId TransformBuilder::matmul(TensorId lhs,
                                  TensorId rhs,
                                  boost::optional<int64_t> virtualGraphId,
                                  const std::string debugPrefix) {
  std::vector<TensorId> inputs = {lhs, rhs};
  return op(Onnx::Operators::MatMul_1,
            inputs,
            {},
            virtualGraphId,
            debugPrefix + "/matmul");
}

TensorId TransformBuilder::slice(TensorId in,
                                 const Shape &starts,
                                 const Shape &ends,
                                 const Shape &axes,
                                 boost::optional<int64_t> virtualGraphId,
                                 const std::string debugPrefix) {
  std::vector<TensorId> inputs = {in};

  return op(Onnx::Operators::Slice_1,
            inputs,
            {{"starts", starts}, {"ends", ends}, {"axes", axes}},
            virtualGraphId,
            debugPrefix + "/slice");
}

void TransformBuilder::slice(TensorId in,
                             const Shape &starts,
                             const Shape &ends,
                             const Shape &axes,
                             TensorId out,
                             boost::optional<int64_t> virtualGraphId,
                             const std::string debugPrefix) {
  std::vector<TensorId> inputs = {in};

  opWithOutput(Onnx::Operators::Slice_1,
               inputs,
               {{"starts", starts}, {"ends", ends}, {"axes", axes}},
               out,
               virtualGraphId,
               debugPrefix + "/slice");
}

TensorId TransformBuilder::transpose(TensorId in,
                                     Shape perm,
                                     boost::optional<int64_t> virtualGraphId,
                                     const std::string debugPrefix) {
  std::vector<TensorId> inputs = {in};
  return op(Onnx::Operators::Transpose_1,
            inputs,
            {{"perm", perm}},
            virtualGraphId,
            debugPrefix + "/transpose");
}

void TransformBuilder::transpose(TensorId in,
                                 Shape perm,
                                 TensorId out,
                                 boost::optional<int64_t> virtualGraphId,
                                 const std::string debugPrefix) {
  std::vector<TensorId> inputs = {in};
  opWithOutput(Onnx::Operators::Transpose_1,
               inputs,
               {{"perm", perm}},
               out,
               virtualGraphId,
               debugPrefix + "/transpose");
}

} // namespace popart
