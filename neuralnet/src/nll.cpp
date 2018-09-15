#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>
#include <neuralnet/nll.hpp>



namespace neuralnet {

onnx::OpSchema createNegLogLikeOpSchema() {
  auto schema = onnx::OpSchema();
  // This is the first version of this opset
  schema.SinceVersion(1);
  // exactly two inputs required
  schema.NumInputs({2});
  // exactly two inputs produced
  schema.NumOutputs({2});
  schema.SetDoc("This is the doc string from operator NegLogLike");
  schema.SetName("NegLogLike");
  schema.SetDomain("gnilwen.semaj");
  // This schema has no attributes
  schema.Input(0, "X", "Tensor pre-log-soft-max", "T", onnx::OpSchema::Single);
  schema.Input(1, "Y", "Label", "I", onnx::OpSchema::Single);
  schema.Output(
      0, "dX", "Gradient of X w.r.t. Loss", "T", onnx::OpSchema::Single);
  schema.Output(1, "Loss", "The Loss", "float", onnx::OpSchema::Optional);
  // Note : this MUST be set otherwise there is a segfault from
  // ONNX in Finalize(). A ticket should be openened with team ONNX.
  schema.TypeConstraint(
      "T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "bla1");
  schema.TypeConstraint("I", {"tensor(int32)"}, "bla2");
  schema.Finalize();
  return schema;
}

const onnx::OpSchema & getNegLogLikeOpSchema(){
  const static onnx::OpSchema schema = createNegLogLikeOpSchema();
  return schema;
}


TensorId getUniqueOutId(const onnx::ModelProto &m) {
  auto nOuts = m.graph().output_size();
  if (nOuts != 1) {
    throw error("cannot create NegLogLikeLoss from onnx Graph with " +
                std::to_string(nOuts) + " outputs");
  }
  return m.graph().output(0).name();
}

NegLogLikeLoss::NegLogLikeLoss(const onnx::ModelProto &m, TensorId id1)
    : NegLogLikeLoss(getUniqueOutId(m), id1) {}

std::vector<TensorId> NegLogLikeLoss::getStreamTensorNames() const {
  return {Y};
}

NegLogLikeLoss::NegLogLikeLoss(TensorId id0, TensorId id1)
    : X(id0), Y(id1) {
}

NegLogLikeOp::NegLogLikeOp(OpId opId,
                           const onnx::NodeProto &node,
                           Graph *pgraph)
    : Op(opId, node, pgraph) {}

void NegLogLikeOp::setup() {
  // dX has same info as X
  output.tensor(0)->info = input.tensor(0)->info;
  // Loss, always float32, rank 1 tensor of size batchsize
  output.tensor(1)->info = {TP::FLOAT, {input.tensor(0)->info.dim(0)}};
}

TensorId NegLogLikeLoss::getLossId() { return "NLLloss"; }

std::vector<std::unique_ptr<Node>> NegLogLikeLoss::getNodes() const {
  std::unique_ptr<Node> NLLnode (new Node);
  auto && schema = getNegLogLikeOpSchema();

  // inputs are the
  //     index 0 : the tensor to have soft max is applied to qne
  //     index 1 : the correct class
  //     we confirm the above, checking the schema has not changed
  if (!(schema.inputs().at(0).GetName() == "X")){
    throw error("NegLogLike Schema does not have input 0 as X");
  }
  if (!(schema.inputs().at(1).GetName() == "Y")){
    throw error("NegLogLike Schema does not have input 1 as Y");
  }
  NLLnode->add_input(X);
  NLLnode->add_input(Y);

  // outputs are
  //     index 0 : the gradient of input 0 w.r.t. the loss
  //     index 1 : the loss
  //     we just cofirm the above
  if (!(schema.outputs().at(0).GetName() == "dX")){
    throw error("NegLogLike Schema does not have output 0 as dX");
  }
  if (!(schema.outputs().at(1).GetName() == "Loss")){
    throw error("NegLogLike Schema does not have output 1 as Loss");
  }
  NLLnode->add_output(getGradId(X));
  NLLnode->add_output(getLossId());
  
  NLLnode->set_op_type("NegLogLike");
  NLLnode->set_domain("gnilwen.semaj");

  // NLLnode has no name
  // NLLnode has no attributes
  // NLLnode has no doc_string
  
  std::vector<std::unique_ptr<Node>> nodes;
  nodes.emplace_back(std::move(NLLnode));
  return nodes;
}

} // namespace neuralnet
