#ifndef GUARD_NEURALNET_OP_TYPES_HPP
#define GUARD_NEURALNET_OP_TYPES_HPP

#include <map>

#include <poponnx/names.hpp>

namespace poponnx {

enum class OpType {
  ADD = 0,
  ADDARG0GRAD,
  ADDARG1GRAD,
  ADDBIAS,
  ADDBIASDATAGRAD,
  ADDBIASBIASGRAD,
  AVERAGEPOOL,
  AVERAGEPOOLGRAD,
  CONSTANT,
  CONSTSGDVARUPDATE,
  CONV,
  CONVDATAGRAD,
  CONVWEIGHTSGRAD,
  IDENTITY,
  IDENTITYGRAD,
  L1,
  L1GRAD,
  SOFTMAX,
  SOFTMAXGRAD,
  SOFTMAXGRADDIRECT,
  NEGATE,
  NEGATEGRAD,
  NLL,
  NLLGRAD,
  MATMUL,
  MATMULLHSGRAD,
  MATMULRHSGRAD,
  MAXPOOL,
  MAXPOOLGRAD,
  MUL,
  MULARG0GRAD,
  MULARG1GRAD,
  PAD,
  REDUCESUM,
  REDUCESUMGRAD,
  RELU,
  RELUGRAD,
  RELUINPLACE,
  SGDVARUPDATE,
  SQUEEZE,
  SQUEEZEGRAD,
  SUBTRACT,
  SUBTRACTARG0GRAD,
  SUBTRACTARG1GRAD,
  SUM
};

struct OpTypeInfo {
  OpTypeInfo(OpType t, bool p) : type(t), is_public(p) {}

  OpType type;
  bool is_public;
};

using OpTypeMap = std::map<std::pair<OpName, OpDomain>, OpTypeInfo>;

class OpTypes {
public:
  OpTypes();
  const OpType &get(OpName op_name, OpDomain op_domain) const;
  const OpName &getName(OpType opType) const;
  const OpDomain &getDomain(OpType opType) const;
  const OpTypeMap &getMap() const;

private:
  OpTypeMap opTypes_;
  std::map<OpType, std::pair<std::string, std::string>> strings_;
};

OpTypes initOpTypes();
const OpTypes &getOpTypes();

std::vector<std::pair<OpName, OpDomain>>
getSupportedOperations(bool includePrivate);

OpDomain getOnnxDomain();
OpDomain getPoponnxDomain();

} // namespace poponnx

#endif
