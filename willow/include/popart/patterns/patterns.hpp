// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERNS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERNS_HPP_

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popart/logging.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/error.hpp"

namespace popart {
class Op;

/// Class representing the pattern set to run.
enum class PatternsLevel {
  /// Do not run any patterns.
  NoPatterns,
  /// Run all mandatory patterns.
  Minimal,
  /// Run the default set of patterns, which includes all mandatory patterns as
  /// well as a few additional patterns.
  Default,
  /// Run all patterns.
  All,
};

/// Class to manage pattern names.
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

// Factory class for the registration of patterns.
class PreAliasPatternManager {
private:
  struct PreAliasPatternInfo {
    bool enabledByDefault;
    bool mandatory;
    std::string name;
    std::function<std::unique_ptr<PreAliasPattern>()> factory;
  };

  PreAliasPatternManager() = default;

  std::map<std::type_index, PreAliasPatternInfo> patternInfos;

  // Singleton
  static PreAliasPatternManager &getInstance();

public:
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

/// A class to hold which patterns are enabled and disabled.
class Patterns {
private:
  template <typename PATTERN> bool isPatternEnabled();

  template <typename PATTERN> Patterns &enablePattern(bool v);

public:
  /**
   * Constructor for the Patterns class.
   *
   * \param level The pattern set to run.
   */
  Patterns(PatternsLevel level);

  /**
   * Default constructor for the Patterns class.
   *
   * The pattern set to run is set to PatternsLevel::Default.
   */
  Patterns() : Patterns(PatternsLevel::Default) {}

  /**
   * Constructor for the Patterns class.
   *
   * \param patterns A vector of pattern names of patterns to be run.
   */
  Patterns(std::vector<std::string> patterns);

  /**
   * Create a set of pattern to be run.
   *
   * \param patterns A vector of pattern names of patterns to be run.
   */
  static Patterns create(std::vector<std::string> patterns);

  /**
   * Check if a pattern (of class PreAliasPattern) is enabled.
   *
   * \param t The pattern to check.
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPatternEnabled(const std::type_index &t);

  /**
   * Check if pattern (not of class PreAliasPattern) is enabled.
   *
   * \param t The name of the pattern to check.
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPatternEnabled(const std::string &t);

  /**
   * Enable a pattern of class PreAliasPattern.
   *
   * \param t The pattern to enable.
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   * \returns Pattern.
   */
  Patterns &enablePattern(const std::type_index &t, bool v);

  /**
   * Enable a pattern not of class PreAliasPattern.
   *
   * \param t The pattern to enable.
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   * \returns Pattern.
   */
  Patterns &enablePattern(const std::string &t, bool v);

  /**
   * Get the names of all patterns of class PreAliasPattern, using the same
   * order as getPreAliasList().
   *
   * \returns A vector of the names of all patterns of class PreAliasPattern.
   */
  static std::vector<std::string> getAllPreAliasPatternNames();

  /**
   * Check if a pattern is mandatory.
   *
   * Mandatory patterns must be enabled and must be run.
   *
   * This method throws an error at runtime if the pattern is
   * disabled and if enableRuntimeAsserts() is `true`.
   *
   * \param pattern The pattern to check.
   * \returns If `true` then pattern is mandatory. If `false` then pattern is
   *      not mandatory.
   */
  static bool isMandatory(Pattern &pattern);

  /**
   * Check if a pattern is mandatory.
   *
   * Mandatory patterns must be enabled and must be run.
   *
   * This method throws an error at runtime if the pattern is
   * disabled and if enableRuntimeAsserts() is `true`.
   *
   * \param patternName The name of the pattern to check.
   * \returns If `true` then pattern is mandatory. If `false` then pattern is
   *      not mandatory.
   */
  static bool isMandatory(std::string &patternName);

  /**
   * Check if InitAccumulatePattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isInitAccumulateEnabled();
  /**
   * Check if PreUniRepl is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPreUniReplEnabled();
  /**
   * Check if PostNRepl is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPostNReplEnabled();
  /**
   * Check if SoftMaxGradDirect is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isSoftMaxGradDirectEnabled();
  /**
   * Check if NlllWithSoftMaxGradDirect is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isNlllWithSoftMaxGradDirectEnabled();
  /**
   * Check if SplitGatherPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isSplitGatherEnabled();
  /**
   * Check if OpToIdentityPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isOpToIdentityEnabled();
  /**
   * Check if UpsampleToResizePattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isUpsampleToResizeEnabled();
  /**
   * Check if SubtractArg1GradOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isSubtractArg1GradOpEnabled();
  /**
   * Check if MulArgGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isMulArgGradOpEnabled();
  /**
   * Check if ReciprocalGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isReciprocalGradOpEnabled();
  /**
   * Check if Atan2Arg0GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isAtan2Arg0GradOpEnabled();
  /**
   * Check if Atan2Arg1GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isAtan2Arg1GradOpEnabled();
  /**
   * Check if DivArg0GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isDivArg0GradOpEnabled();
  /**
   * Check if DivArg1GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isDivArg1GradOpEnabled();
  /**
   * Check if PowArg0GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPowArg0GradOpEnabled();
  /**
   * Check if PowArg1GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isPowArg1GradOpEnabled();
  /**
   * Check if SinGradOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isSinGradOpEnabled();
  /**
   * Check if CosGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isCosGradOpEnabled();
  /**
   * Check if InPlace is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isInPlaceEnabled() { return inplaceEnabled; }

  /**
   * Check if UpdateInplacePrioritiesForIpu is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isUpdateInplacePrioritiesForIpuEnabled() {
    return updateInplacePrioritiesForIpuEnabled;
  }
  /**
   * Check if SqrtGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isSqrtGradOpEnabled();
  /**
   * Check if ConvFlipWeightsDoubleFlipPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isConvFlipWeightsDoubleFlipEnabled();
  /**
   * Check if ConvFlipWeightsGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isConvFlipWeightsGradOpEnabled();
  /**
   * Check if ExpandCastPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isExpandCastEnabled();
  /**
   * Check if ExpGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isExpGradOpEnabled();
  /**
   * Check if Expm1GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isExpm1GradOpEnabled();
  /**
   * Check if Log1pGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isLog1pGradOpEnabled();
  /**
   * Check if LogGradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isLogGradOpEnabled();
  /**
   * Check if NegativeOneScalePattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isNegativeOneScaleEnabled();
  /**
   * Check if MatMulOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isMatMulOpEnabled();
  /**
   * Check if MatMulLhsGradOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isMatMulLhsGradOpEnabled();
  /**
   * Check if MatMulRhsGradOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isMatMulRhsGradOpEnabled();
  /**
   * Check if RandomNormalLikeOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isRandomNormalLikeOpPatternEnabled();
  /**
   * Check if RandomUniformLikeOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isRandomUniformLikeOpPatternEnabled();
  /**
   * Check if ZerosLikeOp is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isZerosLikeOpPatternEnabled();
  /**
   * Check if DecomposeBinaryConstScalar is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isDecomposeBinaryConstScalarEnabled();
  /**
   * Check if FmodArg0GradOpPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isFmodArg0GradOpEnabled();
  /**
   * Check if LambSerialisedWeightPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isLambSerialisedWeightEnabled();
  /**
   * Check if TiedGatherPattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isTiedGatherEnabled();
  /**
   * Check if TiedGatherAccumulatePattern is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool isTiedGatherAccumulateEnabled();

  // The following methods are fluent allow you to
  // Patterns().enableInPlace0(false).
  //           enablePreUniRepl(true);

  /**
   * Enable or disable InitAccumulatePattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableInitAccumulate(bool v);

  /**
   * Enable or disable PreUniRepl.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enablePreUniRepl(bool v);

  /**
   * Enable or disable PostNRepl.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enablePostNRepl(bool v);

  /**
   * Enable or disable SoftMaxGradDirect.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableSoftMaxGradDirect(bool v);

  /**
   * Enable or disable NlllWithSoftMaxGradDirect.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableNlllWithSoftMaxGradDirect(bool v);

  /**
   * Enable or disable SplitGatherPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableSplitGather(bool v);

  /**
   * Enable or disable OpToIdentityPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableOpToIdentity(bool v);

  /**
   * Enable or disable UpsampleToResizePattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableUpsampleToResize(bool v);

  /**
   * Enable or disable SubtractArg1GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableSubtractArg1GradOp(bool v);

  /**
   * Enable or disable MulArgGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableMulArgGradOp(bool v);

  /**
   * Enable or disable ReciprocalGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableReciprocalGradOp(bool v);

  /**
   * Enable or disable Atan2Arg0GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableAtan2Arg0GradOp(bool v);

  /**
   * Enable or disable Atan2Arg1GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableAtan2Arg1GradOp(bool v);

  /**
   * Enable or disable DivArg0GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableDivArg0GradOp(bool v);

  /**
   * Enable or disable DivArg1GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableDivArg1GradOp(bool v);

  /**
   * Enable or disable PowArg0GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enablePowArg0GradOp(bool v);

  /**
   * Enable or disable PowArg1GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enablePowArg1GradOp(bool v);

  /**
   * Enable or disable SinGradOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableSinGradOp(bool v);

  /**
   * Enable or disable CosGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableCosGradOp(bool v);

  /**
   * Enable or disable InPlace.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableInPlace(bool v) {
    inplaceEnabled = v;
    return *this;
  }

  /**
   * Enable or disable UpdateInplacePrioritiesForIpu.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableUpdateInplacePrioritiesForIpu(bool v) {
    updateInplacePrioritiesForIpuEnabled = v;
    return *this;
  }

  /**
   * Enable or disable SqrtGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableSqrtGradOp(bool v);

  /**
   * Enable or disable ConvFlipWeightsDoubleFlipPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableConvFlipWeightsDoubleFlip(bool v);

  /**
   * Enable or disable ConvFlipWeightsGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableConvFlipWeightsGradOp(bool v);

  /**
   * Enable or disable ExpGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableExpGradOp(bool v);

  /**
   * Enable or disable Expm1GradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableExpm1GradOp(bool v);

  /**
   * Enable or disable Log1pGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableLog1pGradOp(bool v);

  /**
   * Enable or disable LogGradOpPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableLogGradOp(bool v);

  /**
   * Enable or disable NegativeOneScalePattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableNegativeOneScale(bool v);

  /**
   * Enable or disable MatMulOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableMatMulOp(bool v);

  /**
   * Enable or disable MatMulLhsGradOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableMatMulLhsGradOp(bool v);

  /**
   * Enable or disable MatMulRhsGradOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableMatMulRhsGradOp(bool v);

  /**
   * Enable or disable RandomNormalLikeOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableRandomNormalLikeOpPattern(bool v);

  /**
   * Enable or disable RandomUniformLikeOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableRandomUniformLikeOpPattern(bool v);

  /**
   * Enable or disable ZerosLikeOp.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableZerosLikeOpPattern(bool v);

  /**
   * Enable or disable DecomposeBinaryConstScalar.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableDecomposeBinaryConstScalar(bool v);

  /**
   * Enable or disable LambSerialisedWeightPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableLambSerialisedWeight(bool v);

  /**
   * Enable or disable TiedGatherPattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableTiedGather(bool v);

  /**
   * Enable or disable TiedGatherAccumulatePattern.
   *
   * \param v If `true` then enable pattern. If `false` then disable pattern.
   */
  Patterns &enableTiedGatherAccumulate(bool v);

  /**
   * Enable or disable runtime asserts.
   *
   * If runtime asserts are enabled, then a check is performed to confirm that
   * all mandatory patterns are enabled.
   *
   * \param v If `true` then enable runtime asserts. If `false` then disable run
   * time asserts.
   */
  Patterns &enableRuntimeAsserts(bool b) {
    runtimeAssertsOn = b;
    return *this;
  }

  /**
   * Get list of patterns to be run before aliasing.
   *
   * \returns A vector of pointers to patterns of class PreAliasPattern.
   */
  std::vector<std::unique_ptr<PreAliasPattern>> getPreAliasList();

  /**
   * Equality operator.
   *
   * \param p Pattern to compare to.
   *
   * \returns `true` if patterns are equal; `false` otherwise.
   */
  bool operator==(const Patterns &p) const;

  /**
   * Write a string representation of patterns to an output stream.
   *
   * \param os An output stream that the the string representation should be
   * written to. \param patterns The patterns for which the string
   * representation is created.
   *
   * \returns An output stream containing the string representation of the
   * patterns.
   */
  friend std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

  /**
   * Get the settings (enabled or disabled) for patterns.
   *
   * \returns Map of which patterns are enabled or disabled, indexed by value of
   * std::type_index.
   */
  const std::map<std::type_index, bool> &getSettings() const {
    return settings;
  }

  /**
   * Check if the pattern InPlace is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool getInplaceEnabled() const { return inplaceEnabled; }

  /**
   * Check if the pattern UpdateInplacePrioritiesForIpu is enabled.
   *
   * \returns `true` if pattern is enabled; `false` otherwise.
   */
  bool getUpdateInplacePrioritiesForIpuEnabled() const {
    return updateInplacePrioritiesForIpuEnabled;
  }

  /**
   * Check if runtime asserts are enabled.
   *
   * If runtime asserts are enabled, then a check is performed to confirm that
   * all mandatory patterns are enabled.
   *
   * \returns `true` if runtime asserts are enabled; `false` otherwise.
   */
  bool getRuntimeAssertsOn() const { return runtimeAssertsOn; }

private:
  void ensureAllMandatoryPreAliasPatternsAreEnabled() const;

  // Map of which patterns are enabled, indexed by value of std::type_index
  std::map<std::type_index, bool> settings;

  // the patterns which are not instances of PreAliasPattern
  bool inplaceEnabled{false};
  bool updateInplacePrioritiesForIpuEnabled{false};

  // If set, we throw an error if a mandatory pattern is disabled.
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

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_PATTERNS_HPP_
