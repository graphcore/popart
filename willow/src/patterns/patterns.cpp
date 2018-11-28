#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

class SoftmaxGradDirectOp;

PatternTypes initPatternTypes() { return PatternTypes(); }

const PatternTypes &getPatternTypes() {
  const static PatternTypes X = initPatternTypes();
  return X;
}

bool Pattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->pir->isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
};

PatternTypes::PatternTypes() {

  opTypes_ = {{"PostNRepl", PatternType::POSTNREPL},
              {"PreUniRepl", PatternType::PREUNIREPL},
              {"SoftmaxGradDirect", PatternType::SOFTMAXGRADDIRECT},
              {"SplitConvBias", PatternType::SPLITCONVBIAS},
              {"OpToIdentity", PatternType::OPTOIDENTITY},
              {"SubtractArg1GradOp", PatternType::SUBTRACTARG1GRADOP},
              {"MulArgGradOp", PatternType::MULARGGRADOP},
              {"Inplace0", PatternType::INPLACE0}};

  std::vector<std::string> opTypeKeys;
  opTypeKeys.reserve(opTypes_.size());
  for (auto &x : opTypes_) {
    strings_[x.second] = x.first;
  }
}

const PatternType &PatternTypes::get(std::string op_type) const {
  auto found = opTypes_.find(op_type);
  if (found == opTypes_.end()) {
    std::vector<std::string> opTypeNames;
    opTypeNames.reserve(opTypes_.size());
    for (auto &name_type : opTypes_) {
      opTypeNames.push_back(name_type.first);
    }
    std::stringstream errm;
    errm << "No PatternType found for " << op_type << ". Options are ";
    appendSequence(errm, opTypeNames);
    throw error(errm.str());
  }

  return found->second;
}

const std::string &PatternTypes::get(PatternType opType) const {
  return strings_.at(opType);
}

} // namespace willow
