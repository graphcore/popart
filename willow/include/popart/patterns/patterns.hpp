// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PATTERNS_HPP
#define GUARD_NEURALNET_PATTERNS_HPP

#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <popart/logging.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/util.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

enum class PatternsLevel { NoPatterns, Minimal, Default, All };

class PatternNames {
public:
  static void addName(const std::type_info &patternInfo,
                      const std::string &name);

  static const std::string &getName(const std::type_info &patternInfo);

  template <typename PATTERN> static const std::string &getName() {
    return getName(typeid(PATTERN));
  }

  static bool contains(const std::string &);

private:
  std::unordered_map<std::type_index, std::string> names;

  static PatternNames &getInstance() {
    static PatternNames instance;
    return instance;
  }
};

template <class PATTERN> class AddPatternName {
public:
  AddPatternName(std::string name) {
    PatternNames::addName(typeid(PATTERN), name);
  }
};

// This is a factory class which the patterns are registered with
class PreAliasPatternManager {
private:
  struct PreAliasPatternInfo {
    bool enabledByDefault;
    bool mandatory;
    std::string name;
    std::function<std::unique_ptr<PreAliasPattern>()> factory;
  };

  PreAliasPatternManager() = default;

  std::map<PreAliasPatternType, std::type_index> patternTypeToTypeIndex;
  std::map<std::type_index, PreAliasPatternInfo> patternInfos;

  // Singleton
  static PreAliasPatternManager &getInstance();

public:
  // could add another parameter to set which level this pattern is in.
  static void
  registerPattern(PreAliasPatternType type,
                  const std::type_index &ti,
                  std::string name,
                  bool enabled,
                  bool mandatory,
                  std::function<std::unique_ptr<PreAliasPattern>()> func) {
    getInstance().patternInfos.insert(
        {ti, PreAliasPatternInfo{enabled, mandatory, name, func}});
    getInstance().patternTypeToTypeIndex.insert({type, ti});
  }

  static void
  registerPattern(const std::type_index &ti,
                  std::string name,
                  bool enabled,
                  bool mandatory,
                  std::function<std::unique_ptr<PreAliasPattern>()> func) {
    getInstance().patternInfos.insert(
        {ti, PreAliasPatternInfo{enabled, mandatory, name, func}});
  }

  static const std::map<std::type_index, PreAliasPatternInfo> &
  getPatternInfos() {
    return getInstance().patternInfos;
  }

  static std::type_index getTypeIndex(PreAliasPatternType type) {
    return getInstance().patternTypeToTypeIndex.at(type);
  }

  static const PreAliasPatternInfo &getInfo(const std::type_index &ti) {
    return getInstance().patternInfos.at(ti);
  }

  static std::string opReplacementPattern(Op *op_) {
    for (auto &ti_info : getInstance().patternInfos) {
      auto &info   = ti_info.second;
      auto pattern = info.factory();
      if (pattern->matches(op_)) {
        auto name = pattern->getPatternName();
        return name;
      }
    }
    return "";
  }

  static std::string getPatternName(const std::type_index &ti) {
    return getInfo(ti).name;
  }

  static nonstd::optional<std::type_index> tryGetTypeIndex(std::string s) {
    for (auto &ti_info : getInstance().patternInfos) {
      auto &ti   = ti_info.first;
      auto &info = ti_info.second;
      if (info.name == s) {
        return ti;
      }
    }
    return nonstd::nullopt;
  }

  static std::type_index getTypeIndex(const std::string &s) {
    auto ti = tryGetTypeIndex(s);
    if (ti) {
      return *ti;
    } else {
      throw error("Unknown pattern {}.", s);
    }
  }

  static std::unique_ptr<PreAliasPattern>
  createPattern(const std::type_index &ti) {
    return getInfo(ti).factory();
  }
};

// This class registers a lambda function to create a pattern with the
// PreAliasPatternManager
template <class PATTERN> class PatternCreator {
public:
  PatternCreator(PreAliasPatternType type,
                 std::string name,
                 bool enabled   = true,
                 bool mandatory = false) {
    auto ti = std::type_index(typeid(PATTERN));
    PreAliasPatternManager::registerPattern(
        type,
        ti,
        name,
        enabled,
        mandatory,
        [name]() -> std::unique_ptr<PreAliasPattern> {
          return std::unique_ptr<PATTERN>(new PATTERN());
        });
    AddPatternName<PATTERN> registerName(name);
  }

  PatternCreator(std::string name,
                 bool enabled   = true,
                 bool mandatory = false) {
    auto ti = std::type_index(typeid(PATTERN));
    PreAliasPatternManager::registerPattern(
        ti,
        name,
        enabled,
        mandatory,
        [name]() -> std::unique_ptr<PreAliasPattern> {
          return std::unique_ptr<PATTERN>(new PATTERN());
        });
    AddPatternName<PATTERN> registerName(name);
  }
};

// A class to hold which patterns are enabled/disabled
class Patterns {
private:
  template <typename PATTERN> bool isPatternEnabled();

  template <typename PATTERN> Patterns &enablePattern(bool v);

public:
  Patterns(PatternsLevel level);
  Patterns() : Patterns(PatternsLevel::Default) {}
  Patterns(std::vector<std::string> patterns);

  static Patterns create(std::vector<std::string> patterns);

  bool isPatternEnabled(const std::type_index &t);
  bool isPatternEnabled(const std::string &t);

  Patterns &enablePattern(const std::type_index &t, bool v);
  Patterns &enablePattern(const std::string &t, bool v);

  bool isInitAccumulateEnabled();
  bool isPreUniReplEnabled();
  bool isPostNReplEnabled();
  bool isSoftMaxGradDirectEnabled();
  bool isNlllWithSoftMaxGradDirectEnabled();
  bool isSplitGatherEnabled();
  bool isOpToIdentityEnabled();
  bool isUpsampleToResizeEnabled();
  bool isSubtractArg1GradOpEnabled();
  bool isMulArgGradOpEnabled();
  bool isReciprocalGradOpEnabled();
  bool isAtan2Arg0GradOpEnabled();
  bool isAtan2Arg1GradOpEnabled();
  bool isDivArg0GradOpEnabled();
  bool isDivArg1GradOpEnabled();
  bool isPowArg0GradOpEnabled();
  bool isPowArg1GradOpEnabled();
  bool isSinGradOpEnabled();
  bool isCosGradOpEnabled();
  bool isInPlaceEnabled() { return inplaceEnabled; }
  bool isUpdateInplacePrioritiesForIpuEnabled() {
    return updateInplacePrioritiesForIpuEnabled;
  }
  bool isSqrtGradOpEnabled();
  bool isExpGradOpEnabled();
  bool isExpm1GradOpEnabled();
  bool isLog1pGradOpEnabled();
  bool isLogGradOpEnabled();
  bool isNegativeOneScaleEnabled();
  bool isMatMulOpEnabled();
  bool isMatMulLhsGradOpEnabled();
  bool isMatMulRhsGradOpEnabled();
  bool isRandomNormalLikeOpPatternEnabled();
  bool isRandomUniformLikeOpPatternEnabled();
  bool isZerosLikeOpPatternEnabled();
  bool isDecomposeBinaryConstScalarEnabled();
  bool isSparseAccumulateEnabled();
  bool isFmodArg0GradOpEnabled();
  bool isLambSerialisedWeightEnabled();

  // The following methods are fluent allow you to
  // Pattens().enableInPlace0(false).
  //           enablePreUniRepl(true);
  Patterns &enableInitAccumulate(bool v);
  Patterns &enablePreUniRepl(bool v);
  Patterns &enablePostNRepl(bool v);
  Patterns &enableSoftMaxGradDirect(bool v);
  Patterns &enableNlllWithSoftMaxGradDirect(bool v);
  Patterns &enableSplitGather(bool v);
  Patterns &enableOpToIdentity(bool v);
  Patterns &enableUpsampleToResize(bool v);
  Patterns &enableSubtractArg1GradOp(bool v);
  Patterns &enableMulArgGradOp(bool v);
  Patterns &enableReciprocalGradOp(bool v);
  Patterns &enableAtan2Arg0GradOp(bool v);
  Patterns &enableAtan2Arg1GradOp(bool v);
  Patterns &enableDivArg0GradOp(bool v);
  Patterns &enableDivArg1GradOp(bool v);
  Patterns &enablePowArg0GradOp(bool v);
  Patterns &enablePowArg1GradOp(bool v);
  Patterns &enableSinGradOp(bool v);
  Patterns &enableCosGradOp(bool v);
  Patterns &enableInPlace(bool v) {
    inplaceEnabled = v;
    return *this;
  }
  Patterns &enableUpdateInplacePrioritiesForIpu(bool v) {
    updateInplacePrioritiesForIpuEnabled = v;
    return *this;
  }
  Patterns &enableSqrtGradOp(bool v);
  Patterns &enableExpGradOp(bool v);
  Patterns &enableExpm1GradOp(bool v);
  Patterns &enableLog1pGradOp(bool v);
  Patterns &enableLogGradOp(bool v);
  Patterns &enableNegativeOneScale(bool v);
  Patterns &enableMatMulOp(bool v);
  Patterns &enableMatMulLhsGradOp(bool v);
  Patterns &enableMatMulRhsGradOp(bool v);
  Patterns &enableRandomNormalLikeOpPattern(bool v);
  Patterns &enableRandomUniformLikeOpPattern(bool v);
  Patterns &enableZerosLikeOpPattern(bool v);
  Patterns &enableSparseAccumulate(bool v);
  Patterns &enableDecomposeBinaryConstScalar(bool v);
  Patterns &enableLambSerialisedWeight(bool v);

  Patterns &enableRuntimeAsserts(bool b) {
    runtimeAssertsOn = b;
    return *this;
  }
  std::vector<std::unique_ptr<PreAliasPattern>> getPreAliasList();

  bool operator==(const Patterns &p) const;
  friend std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

  const std::map<std::type_index, bool> &getSettings() const {
    return settings;
  }
  bool getInplaceEnabled() const { return inplaceEnabled; }
  bool getUpdateInplacePrioritiesForIpuEnabled() const {
    return updateInplacePrioritiesForIpuEnabled;
  }
  bool getRuntimeAssertsOn() const { return runtimeAssertsOn; }

private:
  void ensureAllMandatoryPreAliasPatternsAreEnabled() const;

  // Map of which settings are enabled, indexed by value of std::type_index
  std::map<std::type_index, bool> settings;

  // the one pattern which is not a PreAliasPattern
  bool inplaceEnabled{false};
  bool updateInplacePrioritiesForIpuEnabled{false};

  // If set we throw an error if a mandatory pattern is disabled.
  bool runtimeAssertsOn{true};
};

std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

} // namespace popart

namespace std {
template <> struct hash<popart::Patterns> {
  std::size_t operator()(const popart::Patterns &patterns) const;
};
} // namespace std

namespace popart {
inline std::size_t hash_value(const Patterns &patterns) {
  return std::hash<Patterns>()(patterns);
}
} // namespace popart

#endif
