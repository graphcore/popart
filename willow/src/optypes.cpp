#include <string>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/optypes.hpp>

namespace willow {

static const char *ai_onnx      = "ai.onnx";
static const char *ai_graphcore = "ai.graphcore";

OpTypes::OpTypes() {

  opTypes_ = {
      {std::make_pair("Add", ai_onnx), std::make_pair(OpType::ADD, true)},
      {std::make_pair("AddArg0Grad", ai_graphcore),
       std::make_pair(OpType::ADDARG0GRAD, false)},
      {std::make_pair("AddArg1Grad", ai_graphcore),
       std::make_pair(OpType::ADDARG1GRAD, false)},
      {std::make_pair("AddBias", ai_graphcore),
       std::make_pair(OpType::ADDBIAS, false)},
      {std::make_pair("AddBiasDataGrad", ai_graphcore),
       std::make_pair(OpType::ADDBIASDATAGRAD, false)},
      {std::make_pair("AddBiasBiasGrad", ai_graphcore),
       std::make_pair(OpType::ADDBIASBIASGRAD, false)},
      {std::make_pair("AveragePool", ai_onnx),
       std::make_pair(OpType::AVERAGEPOOL, true)},
      {std::make_pair("AveragePoolGrad", ai_graphcore),
       std::make_pair(OpType::AVERAGEPOOLGRAD, false)},
      {std::make_pair("Constant", ai_onnx),
       std::make_pair(OpType::CONSTANT, true)},
      {std::make_pair("Conv", ai_onnx), std::make_pair(OpType::CONV, true)},
      {std::make_pair("ConvDataGrad", ai_graphcore),
       std::make_pair(OpType::CONVDATAGRAD, false)},
      {std::make_pair("ConvWeightsGrad", ai_graphcore),
       std::make_pair(OpType::CONVWEIGHTSGRAD, false)},
      {std::make_pair("Identity", ai_onnx),
       std::make_pair(OpType::IDENTITY, true)},
      {std::make_pair("IdentityGrad", ai_graphcore),
       std::make_pair(OpType::IDENTITYGRAD, false)},
      {std::make_pair("L1", ai_graphcore), std::make_pair(OpType::L1, false)},
      {std::make_pair("L1Grad", ai_graphcore),
       std::make_pair(OpType::L1GRAD, false)},
      {std::make_pair("Softmax", ai_onnx),
       std::make_pair(OpType::SOFTMAX, true)},
      {std::make_pair("SoftmaxGrad", ai_graphcore),
       std::make_pair(OpType::SOFTMAXGRAD, false)},
      {std::make_pair("SoftmaxGradDirect", ai_graphcore),
       std::make_pair(OpType::SOFTMAXGRADDIRECT, false)},
      {std::make_pair("Negate", ai_onnx), std::make_pair(OpType::NEGATE, true)},
      {std::make_pair("NegateGrad", ai_graphcore),
       std::make_pair(OpType::NEGATEGRAD, false)},
      {std::make_pair("Nll", ai_graphcore), std::make_pair(OpType::NLL, false)},
      {std::make_pair("NllGrad", ai_graphcore),
       std::make_pair(OpType::NLLGRAD, false)},
      {std::make_pair("MatMul", ai_onnx), std::make_pair(OpType::MATMUL, true)},
      {std::make_pair("MatMulLhsGrad", ai_graphcore),
       std::make_pair(OpType::MATMULLHSGRAD, false)},
      {std::make_pair("MatMulRhsGrad", ai_graphcore),
       std::make_pair(OpType::MATMULRHSGRAD, false)},
      {std::make_pair("MaxPool", ai_onnx),
       std::make_pair(OpType::MAXPOOL, true)},
      {std::make_pair("MaxPoolGrad", ai_graphcore),
       std::make_pair(OpType::MAXPOOLGRAD, false)},
      {std::make_pair("Pad", ai_onnx), std::make_pair(OpType::PAD, true)},
      {std::make_pair("ReduceSum", ai_onnx),
       std::make_pair(OpType::REDUCESUM, true)},
      {std::make_pair("ReduceSumGrad", ai_graphcore),
       std::make_pair(OpType::REDUCESUMGRAD, false)},
      {std::make_pair("Relu", ai_onnx), std::make_pair(OpType::RELU, true)},
      {std::make_pair("ReluGrad", ai_graphcore),
       std::make_pair(OpType::RELUGRAD, false)},
      {std::make_pair("Sub", ai_onnx), std::make_pair(OpType::SUBTRACT, true)},
      {std::make_pair("SubtractArg0Grad", ai_graphcore),
       std::make_pair(OpType::SUBTRACTARG0GRAD, false)},
      {std::make_pair("SubtractArg1Grad", ai_graphcore),
       std::make_pair(OpType::SUBTRACTARG1GRAD, false)},
      {std::make_pair("Sum", ai_onnx), std::make_pair(OpType::SUM, true)},
      {std::make_pair("Squeeze", ai_onnx),
       std::make_pair(OpType::SQUEEZE, true)},
      {std::make_pair("SqueezeGrad", ai_graphcore),
       std::make_pair(OpType::SQUEEZEGRAD, false)},
      {std::make_pair("SGDVarUpdate", ai_graphcore),
       std::make_pair(OpType::SGDVARUPDATE, false)},
      {std::make_pair("ConstSGDVarUpdate", ai_graphcore),
       std::make_pair(OpType::CONSTSGDVARUPDATE, false)}};

  for (auto &x : opTypes_) {
    strings_[x.second.first] = x.first;
  }
}

const OpType &OpTypes::get(std::string op_type, std::string op_domain) const {
  std::string domain = op_domain;
  if (domain == "") {
    domain = "ai.onnx";
  }

  auto found = opTypes_.find(std::make_pair(op_type, domain));
  if (found == opTypes_.end()) {
    throw error("No OpType found for `" + domain + ":" + op_type + "'");
  }
  return found->second.first;
}

const std::string &OpTypes::getName(OpType opType) const {
  return strings_.at(opType).first;
}

const std::string &OpTypes::getDomain(OpType opType) const {
  return strings_.at(opType).second;
}

const OpTypeMap &OpTypes::getMap() const { return opTypes_; };

OpTypes initOpTypes() { return OpTypes(); }

const OpTypes &getOpTypes() {
  const static OpTypes X = initOpTypes();
  return X;
}

std::vector<std::pair<std::string, std::string>>
getSupportedOperations(bool includePrivate) {
  std::vector<std::pair<std::string, std::string>> list;
  for (auto op_type : getOpTypes().getMap()) {
    if (op_type.second.second || includePrivate) {
      list.push_back(op_type.first);
    }
  }
  return list;
}

std::string getOnnxDomain() { return ai_onnx; }
std::string getPoponnxDomain() { return ai_graphcore; }

} // namespace willow
