#include <string>
#include <vector>

#include <poponnx/error.hpp>
#include <poponnx/optypes.hpp>

namespace willow {

static const char *ai_onnx      = "ai.onnx";
static const char *ai_graphcore = "ai.graphcore";

OpTypes::OpTypes() {

  opTypes_ = {
      {std::make_pair("Add", ai_onnx), OpTypeInfo(OpType::ADD, true)},
      {std::make_pair("AddArg0Grad", ai_graphcore),
       OpTypeInfo(OpType::ADDARG0GRAD, false)},
      {std::make_pair("AddArg1Grad", ai_graphcore),
       OpTypeInfo(OpType::ADDARG1GRAD, false)},
      {std::make_pair("AddBias", ai_graphcore),
       OpTypeInfo(OpType::ADDBIAS, false)},
      {std::make_pair("AddBiasDataGrad", ai_graphcore),
       OpTypeInfo(OpType::ADDBIASDATAGRAD, false)},
      {std::make_pair("AddBiasBiasGrad", ai_graphcore),
       OpTypeInfo(OpType::ADDBIASBIASGRAD, false)},
      {std::make_pair("AveragePool", ai_onnx),
       OpTypeInfo(OpType::AVERAGEPOOL, true)},
      {std::make_pair("AveragePoolGrad", ai_graphcore),
       OpTypeInfo(OpType::AVERAGEPOOLGRAD, false)},
      {std::make_pair("Constant", ai_onnx), OpTypeInfo(OpType::CONSTANT, true)},
      {std::make_pair("Conv", ai_onnx), OpTypeInfo(OpType::CONV, true)},
      {std::make_pair("ConvDataGrad", ai_graphcore),
       OpTypeInfo(OpType::CONVDATAGRAD, false)},
      {std::make_pair("ConvWeightsGrad", ai_graphcore),
       OpTypeInfo(OpType::CONVWEIGHTSGRAD, false)},
      {std::make_pair("Identity", ai_onnx), OpTypeInfo(OpType::IDENTITY, true)},
      {std::make_pair("IdentityGrad", ai_graphcore),
       OpTypeInfo(OpType::IDENTITYGRAD, false)},
      {std::make_pair("L1", ai_graphcore), OpTypeInfo(OpType::L1, false)},
      {std::make_pair("L1Grad", ai_graphcore),
       OpTypeInfo(OpType::L1GRAD, false)},
      {std::make_pair("Softmax", ai_onnx), OpTypeInfo(OpType::SOFTMAX, true)},
      {std::make_pair("SoftmaxGrad", ai_graphcore),
       OpTypeInfo(OpType::SOFTMAXGRAD, false)},
      {std::make_pair("SoftmaxGradDirect", ai_graphcore),
       OpTypeInfo(OpType::SOFTMAXGRADDIRECT, false)},
      {std::make_pair("Negate", ai_onnx), OpTypeInfo(OpType::NEGATE, true)},
      {std::make_pair("NegateGrad", ai_graphcore),
       OpTypeInfo(OpType::NEGATEGRAD, false)},
      {std::make_pair("Nll", ai_graphcore), OpTypeInfo(OpType::NLL, false)},
      {std::make_pair("NllGrad", ai_graphcore),
       OpTypeInfo(OpType::NLLGRAD, false)},
      {std::make_pair("MatMul", ai_onnx), OpTypeInfo(OpType::MATMUL, true)},
      {std::make_pair("MatMulLhsGrad", ai_graphcore),
       OpTypeInfo(OpType::MATMULLHSGRAD, false)},
      {std::make_pair("MatMulRhsGrad", ai_graphcore),
       OpTypeInfo(OpType::MATMULRHSGRAD, false)},
      {std::make_pair("MaxPool", ai_onnx), OpTypeInfo(OpType::MAXPOOL, true)},
      {std::make_pair("MaxPoolGrad", ai_graphcore),
       OpTypeInfo(OpType::MAXPOOLGRAD, false)},
      {std::make_pair("Pad", ai_onnx), OpTypeInfo(OpType::PAD, true)},
      {std::make_pair("ReduceSum", ai_onnx),
       OpTypeInfo(OpType::REDUCESUM, true)},
      {std::make_pair("ReduceSumGrad", ai_graphcore),
       OpTypeInfo(OpType::REDUCESUMGRAD, false)},
      {std::make_pair("Relu", ai_onnx), OpTypeInfo(OpType::RELU, true)},
      {std::make_pair("ReluGrad", ai_graphcore),
       OpTypeInfo(OpType::RELUGRAD, false)},
      {std::make_pair("Sub", ai_onnx), OpTypeInfo(OpType::SUBTRACT, true)},
      {std::make_pair("SubtractArg0Grad", ai_graphcore),
       OpTypeInfo(OpType::SUBTRACTARG0GRAD, false)},
      {std::make_pair("SubtractArg1Grad", ai_graphcore),
       OpTypeInfo(OpType::SUBTRACTARG1GRAD, false)},
      {std::make_pair("Sum", ai_onnx), OpTypeInfo(OpType::SUM, true)},
      {std::make_pair("Squeeze", ai_onnx), OpTypeInfo(OpType::SQUEEZE, true)},
      {std::make_pair("SqueezeGrad", ai_graphcore),
       OpTypeInfo(OpType::SQUEEZEGRAD, false)},
      {std::make_pair("SGDVarUpdate", ai_graphcore),
       OpTypeInfo(OpType::SGDVARUPDATE, false)},
      {std::make_pair("ConstSGDVarUpdate", ai_graphcore),
       OpTypeInfo(OpType::CONSTSGDVARUPDATE, false)}};

  for (auto &x : opTypes_) {
    strings_[x.second.type] = x.first;
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
  return found->second.type;
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
    if (op_type.second.is_public || includePrivate) {
      list.push_back(op_type.first);
    }
  }
  return list;
}

std::string getOnnxDomain() { return ai_onnx; }
std::string getPoponnxDomain() { return ai_graphcore; }

} // namespace willow
