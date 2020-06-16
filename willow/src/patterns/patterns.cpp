// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/logging.hpp>
#include <popart/patterns/patterns.hpp>

#include <popart/op/abs.hpp>
#include <popart/op/cos.hpp>
#include <popart/op/sign.hpp>
#include <popart/op/sin.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/patterns/convbias.hpp>
#include <popart/patterns/convdatagrad.hpp>
#include <popart/patterns/cosgradoppattern.hpp>
#include <popart/patterns/coshoppattern.hpp>
#include <popart/patterns/divarg0gradoppattern.hpp>
#include <popart/patterns/divarg1gradoppattern.hpp>
#include <popart/patterns/elementwisegradoppattern.hpp>
#include <popart/patterns/expgradoppattern.hpp>
#include <popart/patterns/gemmdecompositionpattern.hpp>
#include <popart/patterns/initaccumulatepattern.hpp>
#include <popart/patterns/inplace.hpp>
#include <popart/patterns/loggradoppattern.hpp>
#include <popart/patterns/logsoftmaxoppattern.hpp>
#include <popart/patterns/lstmoppattern.hpp>
#include <popart/patterns/matmulgradpattern.hpp>
#include <popart/patterns/mularggradoppattern.hpp>
#include <popart/patterns/negativeonescalepattern.hpp>
#include <popart/patterns/nlllwithsoftmaxgraddirect.hpp>
#include <popart/patterns/optoidentitypattern.hpp>
#include <popart/patterns/optoreshapepattern.hpp>
#include <popart/patterns/padsum.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/postnrepl.hpp>
#include <popart/patterns/powarg0gradoppattern.hpp>
#include <popart/patterns/powarg1gradoppattern.hpp>
#include <popart/patterns/preunirepl.hpp>
#include <popart/patterns/reciprocalgradoppattern.hpp>
#include <popart/patterns/sequenceexpander.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/patterns/softmaxgraddirect.hpp>
#include <popart/patterns/splitgather.hpp>
#include <popart/patterns/splitgradoptoconcatpattern.hpp>
#include <popart/patterns/splitoppattern.hpp>
#include <popart/patterns/sqrtgradoppattern.hpp>
#include <popart/patterns/subtractarg1gradoppattern.hpp>
#include <popart/patterns/sumtoaddpattern.hpp>
#include <popart/patterns/tantosinovercospattern.hpp>
#include <popart/patterns/updateinplaceprioritiesforipu.hpp>

namespace popart {

void PatternNames::addName(const std::type_info &patternInfo,
                           const std::string &name) {
  auto &instance = getInstance();
  auto found     = instance.names.find(std::type_index(patternInfo));
  if (found == instance.names.end()) {
    instance.names[std::type_index(patternInfo)] = name;
  } else {
    throw error("A name has already been added for pattern class {}",
                patternInfo.name());
  }
}

const std::string &PatternNames::getName(const std::type_info &patternInfo) {
  auto &instance = getInstance();
  auto found     = instance.names.find(std::type_index(patternInfo));
  if (found != instance.names.end()) {
    return found->second;
  } else {
    throw error("Could not return name for pattern {}", patternInfo.name());
  }
}

bool PatternNames::contains(const std::string &name) {
  auto &instance = getInstance();
  for (auto &info_name : instance.names) {
    if (name == info_name.second) {
      return true;
    }
  }
  return false;
}

PreAliasPatternManager &PreAliasPatternManager::getInstance() {
  static PreAliasPatternManager instance;
  return instance;
}

Patterns::Patterns(PatternsLevel level) {
  switch (level) {

  // add the default patterns
  case PatternsLevel::Default: {
    for (auto &ti_info : PreAliasPatternManager::getPatternInfos()) {
      auto &ti   = ti_info.first;
      auto &info = ti_info.second;
      settings.insert({ti, info.enabledByDefault});
    }
    inplaceEnabled = true;
    break;
  }

  // add all of the patterns
  case PatternsLevel::All: {
    for (auto &ti_info : PreAliasPatternManager::getPatternInfos()) {
      auto &ti = ti_info.first;
      settings.insert({ti, true});
    }
    inplaceEnabled = true;
    break;
  }

  // add none of the patterns
  case PatternsLevel::NoPatterns: {
    break;
  }
  };
}

Patterns::Patterns(std::vector<PreAliasPatternType> types) {
  logging::pattern::warn(
      "`Patterns::Patterns(std::vector<PreAliasPatternType> types)' "
      "constructor is deprecated and will be removed in a future release. "
      "Please use `static Patterns Patterns::create(std::vector<std::string> "
      "patterns)' instead");

  for (auto type : types) {
    auto ti = PreAliasPatternManager::getTypeIndex(type);
    settings.insert({ti, true});
  }
}

Patterns Patterns::create(std::vector<std::string> strings) {
  Patterns patterns(PatternsLevel::NoPatterns);

  for (auto p : strings) {
    if (p == "InPlace") {
      patterns.enableInPlace(true);
    } else {
      auto ti = PreAliasPatternManager::tryGetTypeIndex(p);
      if (ti) {
        patterns.settings.insert({*ti, true});
      } else {
        if (p == "Inplace") {
          throw error("Unknown pattern {}, did you mean InPlace?", p);
        } else {
          throw error("Unknown pattern {}", p);
        }
      }
    }
  }

  return patterns;
}

template <typename PATTERN> bool Patterns::isPatternEnabled() {
  auto ti = std::type_index(typeid(PATTERN));
  return isPatternEnabled(ti);
}

template <typename PATTERN> Patterns &Patterns::enablePattern(bool v) {
  auto ti = std::type_index(typeid(PATTERN));
  return enablePattern(ti, v);
}

bool Patterns::isPatternEnabled(const std::type_index &t) {
  auto it = settings.find(t);
  if (it != settings.end()) {
    return it->second;
  }

  return false;
}

bool Patterns::isPatternEnabled(const std::string &t) {
  auto ti = PreAliasPatternManager::getTypeIndex(t);
  return isPatternEnabled(ti);
}

bool Patterns::isPatternEnabled(PreAliasPatternType t) {
  logging::pattern::warn(
      "`bool Patterns::isPatternEnabled(PreAliasPatternType t)' is deprecated "
      "and will be removed in a future release. Please use `bool "
      "Patterns::isPatternEnabled(const std::string &)' instead");

  auto ti = PreAliasPatternManager::getTypeIndex(t);
  return isPatternEnabled(ti);
}

bool Patterns::isInitAccumulateEnabled() {
  return isPatternEnabled<InitAccumulatePattern>();
}

bool Patterns::isPreUniReplEnabled() { return isPatternEnabled<PreUniRepl>(); }

bool Patterns::isPostNReplEnabled() { return isPatternEnabled<PostNRepl>(); }

bool Patterns::isSoftMaxGradDirectEnabled() {
  return isPatternEnabled<SoftmaxGradDirect>();
}

bool Patterns::isNlllWithSoftMaxGradDirectEnabled() {
  return isPatternEnabled<NlllWithSoftmaxGradDirect>();
}

bool Patterns::isSplitConvBiasEnabled() {
  return isPatternEnabled<ConvBiasPattern>();
}

bool Patterns::isSplitGatherEnabled() {
  return isPatternEnabled<SplitGatherPattern>();
}

bool Patterns::isOpToIdentityEnabled() {
  return isPatternEnabled<OpToIdentityPattern>();
}

bool Patterns::isOpToReshapeEnabled() {
  return isPatternEnabled<OpToReshapePattern>();
}

bool Patterns::isSubtractArg1GradOpEnabled() {
  return isPatternEnabled<SubtractArg1GradOpPattern>();
}

bool Patterns::isMulArgGradOpEnabled() {
  return isPatternEnabled<MulArgGradOpPattern>();
}

bool Patterns::isReciprocalGradOpEnabled() {
  return isPatternEnabled<ReciprocalGradOpPattern>();
}

bool Patterns::isDivArg0GradOpEnabled() {
  return isPatternEnabled<DivArg0GradOpPattern>();
}

bool Patterns::isDivArg1GradOpEnabled() {
  return isPatternEnabled<DivArg1GradOpPattern>();
}

bool Patterns::isPowArg0GradOpEnabled() {
  return isPatternEnabled<PowArg0GradOpPattern>();
}

bool Patterns::isPowArg1GradOpEnabled() {
  return isPatternEnabled<PowArg1GradOpPattern>();
}

bool Patterns::isSinGradOpEnabled() {
  return isPatternEnabled<ElementWiseGradOpPattern<SinGradOp, CosOp>>();
}

bool Patterns::isCosGradOpEnabled() {
  return isPatternEnabled<CosGradOpPattern>();
}

bool Patterns::isTanToSinOverCosEnabled() {
  return isPatternEnabled<TanToSinOverCosPattern>();
}

bool Patterns::isSqrtGradOpEnabled() {
  return isPatternEnabled<SqrtGradOpPattern>();
}

bool Patterns::isExpGradOpEnabled() {
  return isPatternEnabled<ExpGradOpPattern>();
}

bool Patterns::isLogGradOpEnabled() {
  return isPatternEnabled<LogGradOpPattern>();
}

bool Patterns::isLogSoftmaxOpEnabled() {
  return isPatternEnabled<LogSoftmaxOpPattern>();
}

bool Patterns::isGemmDecompositionEnabled() {
  return isPatternEnabled<GemmDecompositionPattern>();
}

bool Patterns::isNegativeOneScaleEnabled() {
  return isPatternEnabled<NegativeOneScalePattern>();
}

bool Patterns::isMatMulOpEnabled() { return isPatternEnabled<MatMulPattern>(); }

bool Patterns::isMatMulLhsGradOpEnabled() {
  return isPatternEnabled<MatMulLhsGradPattern>();
}

bool Patterns::isMatMulRhsGradOpEnabled() {
  return isPatternEnabled<MatMulRhsGradPattern>();
}

Patterns &Patterns::enableInitAccumulate(bool v) {
  return enablePattern<InitAccumulatePattern>(v);
}

Patterns &Patterns::enablePreUniRepl(bool v) {
  return enablePattern<PreUniRepl>(v);
}

Patterns &Patterns::enablePostNRepl(bool v) {
  return enablePattern<PostNRepl>(v);
}

Patterns &Patterns::enableSoftMaxGradDirect(bool v) {
  return enablePattern<SoftmaxGradDirect>(v);
}

Patterns &Patterns::enableNlllWithSoftMaxGradDirect(bool v) {
  return enablePattern<NlllWithSoftmaxGradDirect>(v);
}

Patterns &Patterns::enableOpToReshape(bool v) {
  return enablePattern<OpToReshapePattern>(v);
}

Patterns &Patterns::enableSplitConvBias(bool v) {
  return enablePattern<ConvBiasPattern>(v);
}

Patterns &Patterns::enableSplitGather(bool v) {
  return enablePattern<SplitGatherPattern>(v);
}

Patterns &Patterns::enableOpToIdentity(bool v) {
  return enablePattern<OpToIdentityPattern>(v);
}

Patterns &Patterns::enableSubtractArg1GradOp(bool v) {
  return enablePattern<SubtractArg1GradOpPattern>(v);
}

Patterns &Patterns::enableMulArgGradOp(bool v) {
  return enablePattern<MulArgGradOpPattern>(v);
}

Patterns &Patterns::enableReciprocalGradOp(bool v) {
  return enablePattern<ReciprocalGradOpPattern>(v);
}

Patterns &Patterns::enableDivArg0GradOp(bool v) {
  return enablePattern<DivArg0GradOpPattern>(v);
}

Patterns &Patterns::enableDivArg1GradOp(bool v) {
  return enablePattern<DivArg1GradOpPattern>(v);
}

Patterns &Patterns::enablePowArg0GradOp(bool v) {
  return enablePattern<PowArg0GradOpPattern>(v);
}

Patterns &Patterns::enablePowArg1GradOp(bool v) {
  return enablePattern<PowArg1GradOpPattern>(v);
}

Patterns &Patterns::enableSinGradOp(bool v) {
  return enablePattern<ElementWiseGradOpPattern<SinGradOp, CosOp>>(v);
}

Patterns &Patterns::enableCosGradOp(bool v) {
  return enablePattern<CosGradOpPattern>(v);
}

Patterns &Patterns::enableTanToSinOverCos(bool v) {
  return enablePattern<TanToSinOverCosPattern>(v);
}

Patterns &Patterns::enableSqrtGradOp(bool v) {
  return enablePattern<SqrtGradOpPattern>(v);
}

Patterns &Patterns::enableExpGradOp(bool v) {
  return enablePattern<ExpGradOpPattern>(v);
}

Patterns &Patterns::enableLogGradOp(bool v) {
  return enablePattern<LogGradOpPattern>(v);
}

Patterns &Patterns::enableLogSoftmaxOp(bool v) {
  return enablePattern<LogSoftmaxOpPattern>(v);
}

Patterns &Patterns::enableGemmDecomposition(bool v) {
  return enablePattern<GemmDecompositionPattern>(v);
}

Patterns &Patterns::enableNegativeOneScale(bool v) {
  return enablePattern<NegativeOneScalePattern>(v);
}

Patterns &Patterns::enableMatMulOp(bool v) {
  return enablePattern<MatMulPattern>(v);
}

Patterns &Patterns::enableMatMulLhsGradOp(bool v) {
  return enablePattern<MatMulLhsGradPattern>(v);
}

Patterns &Patterns::enableMatMulRhsGradOp(bool v) {
  return enablePattern<MatMulRhsGradPattern>(v);
}

Patterns &Patterns::enablePattern(const std::type_index &t, bool v) {
  logging::pattern::warn(
      "Pattern {} {}", PreAliasPatternManager::getPatternName(t), v);
  settings[t] = v;
  return *this;
}

Patterns &Patterns::enablePattern(const std::string &t, bool v) {
  auto ti = PreAliasPatternManager::getTypeIndex(t);
  return enablePattern(ti, v);
}

Patterns &Patterns::enablePattern(PreAliasPatternType t, bool v) {
  logging::pattern::warn(
      "`Patterns &Patterns::enablePattern(PreAliasPatternType t, bool v)' is "
      "deprecated and will be removed in a future release. Please use "
      "`Patterns &Patterns::enablePattern(const std::string &)' instead");
  auto ti = PreAliasPatternManager::getTypeIndex(t);
  return enablePattern(ti, v);
}

std::vector<std::unique_ptr<PreAliasPattern>> Patterns::getPreAliasList() {
  static std::map<std::type_index, float> patternPriority{
      {std::type_index(typeid(InitAccumulatePattern)), 37},
      {std::type_index(typeid(PreUniRepl)), 36},
      {std::type_index(typeid(PostNRepl)), 35},
      {std::type_index(typeid(SoftmaxGradDirect)), 34},
      {std::type_index(typeid(NlllWithSoftmaxGradDirect)), 33},
      {std::type_index(typeid(ConvBiasPattern)), 32},
      {std::type_index(typeid(OpToIdentityPattern)), 31},
      {std::type_index(typeid(SubtractArg1GradOpPattern)), 30},
      {std::type_index(typeid(MulArgGradOpPattern)), 29},
      {std::type_index(typeid(ReciprocalGradOpPattern)), 28},
      {std::type_index(typeid(DivArg0GradOpPattern)), 27},
      {std::type_index(typeid(DivArg1GradOpPattern)), 26},
      {std::type_index(typeid(ElementWiseGradOpPattern<SinGradOp, CosOp>)), 25},
      {std::type_index(typeid(CosGradOpPattern)), 24},
      {std::type_index(typeid(TanToSinOverCosPattern)), 23},
      {std::type_index(typeid(SqrtGradOpPattern)), 22},
      {std::type_index(typeid(ExpGradOpPattern)), 21},
      {std::type_index(typeid(LogGradOpPattern)), 20},
      {std::type_index(typeid(CoshOpPattern)), 19},
      {std::type_index(typeid(LogSoftmaxOpPattern)), 18},
      {std::type_index(typeid(GemmDecompositionPattern)), 17},
      {std::type_index(typeid(NegativeOneScalePattern)), 16},
      {std::type_index(typeid(PadSumPattern)), 15},
      {std::type_index(typeid(ElementWiseGradOpPattern<AbsGradOp, SignOp>)),
       14},
      {std::type_index(typeid(SplitGatherPattern)), 13},
      {std::type_index(typeid(ConvDataGradPattern)), 12},
      {std::type_index(typeid(SumToAddPattern)), 11},
      {std::type_index(typeid(OpToReshapePattern)), 10},
      {std::type_index(typeid(SplitGradOpToConcatPattern)), 9},
      {std::type_index(typeid(SplitOpPattern)), 8},
      {std::type_index(typeid(PowArg0GradOpPattern)), 7},
      {std::type_index(typeid(PowArg1GradOpPattern)), 6},
      {std::type_index(typeid(ContiguateIpuCopyIndicesPattern)), 5},
      {std::type_index(typeid(MatMulPattern)), 4},
      {std::type_index(typeid(MatMulLhsGradPattern)), 3},
      {std::type_index(typeid(MatMulRhsGradPattern)), 2},
      {std::type_index(typeid(SGD1Decompose)), 1},
      {std::type_index(typeid(LSTMPattern)), 0}};

  std::vector<std::type_index> patternTypes;
  for (auto &typeIndex_enabled : settings) {
    auto &typeIndex = typeIndex_enabled.first;
    auto enabled    = typeIndex_enabled.second;
    if (enabled) {
      patternTypes.push_back(typeIndex);
    }
  }

  // Deterministic pattern order is important. Sort the vector in order of
  // descending priority, using the entries in `patternPriority`. Any
  // patterns not in `patternPriority` will come after the patterns that are in
  // `patternPriority`, and will be sorted using their names.
  std::sort(
      patternTypes.begin(), patternTypes.end(), [](auto lhs, auto rhs) -> bool {
        auto lhsPriority = patternPriority.find(lhs);
        auto rhsPriority = patternPriority.find(rhs);

        // If both have a priority, sort by descending priority.
        // If neither have a priority, sort by name.
        // If only one has a priority, it should the first of the two.
        if (lhsPriority != patternPriority.end() &&
            rhsPriority != patternPriority.end()) {
          return lhsPriority->second > rhsPriority->second;
        } else if (lhsPriority == patternPriority.end() &&
                   rhsPriority == patternPriority.end()) {
          auto lhsName = PreAliasPatternManager::getPatternName(lhs);
          auto rhsName = PreAliasPatternManager::getPatternName(rhs);
          return lhsName < rhsName;
        } else {
          return lhsPriority != patternPriority.end();
        }
      });

  std::vector<std::unique_ptr<PreAliasPattern>> patterns;
  for (auto ptype : patternTypes) {
    patterns.emplace_back(PreAliasPatternManager::createPattern(ptype));
  }

  return patterns;
}

std::ostream &operator<<(std::ostream &os, const Patterns &patterns) {

  for (auto setting : patterns.settings) {
    os << PreAliasPatternManager::getPatternName(setting.first) << " ";
  }

  return os;
}

bool Patterns::operator==(const Patterns &p) const {
  if (p.settings != this->settings) {
    return false;
  }
  if (this->inplaceEnabled != p.inplaceEnabled) {
    return false;
  }

  return true;
}

} // namespace popart
