#ifndef GUARD_NEURALNET_PATTERNS_HPP
#define GUARD_NEURALNET_PATTERNS_HPP

#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include <poponnx/logging.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/util.hpp>

#include <boost/optional.hpp>

namespace poponnx {

// FFS : add gcc like levels O2, O3, OS etc
enum class PatternsLevel { NONE, DEFAULT, ALL };

// This is a factory class which the ptatterns are registered with
class PatternManager {

  PatternManager() = default;

  // List of all registed pattern types
  std::vector<PatternType> patterns;

  // Used to convert a string to a pattern type
  std::map<std::string, PatternType> stringToPatternTypeMapping;

  // Used to construct the Patterns
  std::map<PatternType, std::function<std::unique_ptr<Pattern>()>> factory;

  // Singleton
  static PatternManager &getInstance() {
    static PatternManager instance;
    return instance;
  }

public:
  // could add another parameter to set which level this pattern is in.
  static void registerPattern(PatternType type,
                              std::string name,
                              std::function<std::unique_ptr<Pattern>()> func) {
    getInstance().patterns.push_back(type);
    getInstance().stringToPatternTypeMapping.insert(
        std::pair<std::string, PatternType>(name, type));
    getInstance().factory.insert(
        std::pair<PatternType, std::function<std::unique_ptr<Pattern>()>>(
            type, func));
  }

  static const std::vector<PatternType> &getPatternList() {
    return getInstance().patterns;
  }

  static std::string getPatternName(PatternType type) {
    for (auto i : getInstance().stringToPatternTypeMapping) {
      if (i.second == type)
        return i.first;
    }
    return "unknown";
  }

  static boost::optional<PatternType> convertPatternType(std::string s) {
    auto it = getInstance().stringToPatternTypeMapping.find(s);
    return boost::optional<PatternType>{
        it != getInstance().stringToPatternTypeMapping.end(), it->second};
  }

  static std::unique_ptr<Pattern> createPattern(PatternType type) {
    auto it = getInstance().factory.find(type);
    if (it != getInstance().factory.end()) {
      return it->second();
    } else
      return nullptr;
  }
};

// This class registers a lambda function to cerate a pattern with the
// PatternManager
template <class PATTERN> class PatternCreator {
public:
  PatternCreator(PatternType type, std::string name) {
    PatternManager::registerPattern(
        type, name, []() -> std::unique_ptr<Pattern> {
          return std::unique_ptr<PATTERN>(new PATTERN());
        });
  }
};

// A class to hold which patterns are enabled/disabled
// FFS : Should this be renamed PatternSettings. i.e. it represents the graph
// 'optimization' settings
class Patterns {

public:
  Patterns(std::vector<PatternType> types);

  Patterns(PatternsLevel level = PatternsLevel::DEFAULT);

  static Patterns create(std::vector<std::string> patterns);

  bool isPatternEnabled(PatternType t);
  Patterns &enablePattern(PatternType t, bool v);

  bool isPreUniReplEnabled() {
    return isPatternEnabled(PatternType::PREUNIREPL);
  }
  bool isPostNReplEnabled() { return isPatternEnabled(PatternType::POSTNREPL); }
  bool isSoftMaxGradDirectEnabled() {
    return isPatternEnabled(PatternType::SOFTMAXGRADDIRECT);
  }
  bool isSplitConvBiasEnabled() {
    return isPatternEnabled(PatternType::SPLITCONVBIAS);
  }
  bool isOpToIdentityEnabled() {
    return isPatternEnabled(PatternType::OPTOIDENTITY);
  }
  bool isSubtractArg1GradOpEnabled() {
    return isPatternEnabled(PatternType::SUBTRACTARG1GRADOP);
  }
  bool isMulArgGradOpEnabled() {
    return isPatternEnabled(PatternType::MULARGGRADOP);
  }
  bool isReciprocalGradOpEnabled() {
    return isPatternEnabled(PatternType::RECIPROCALGRADOP);
  }
  bool isDivArg0GradOpEnabled() {
    return isPatternEnabled(PatternType::DIVARG0GRADOP);
  }
  bool isDivArg1GradOpEnabled() {
    return isPatternEnabled(PatternType::DIVARG1GRADOP);
  }
  bool isSinGradOpEnabled() { return isPatternEnabled(PatternType::SINGRADOP); }
  bool isCosGradOpEnabled() { return isPatternEnabled(PatternType::COSGRADOP); }
  bool isTanOpEnabled() { return isPatternEnabled(PatternType::TANOP); }
  bool isInPlace0Enabled() { return isPatternEnabled(PatternType::INPLACE0); }

  // The following methods are fluent allow you to
  // Pattens().enableInPlace0(false).
  //           enablePreUniRepl(true);
  Patterns &enablePreUniRepl(bool v) {
    return enablePattern(PatternType::PREUNIREPL, v);
  }
  Patterns &enablePostNRepl(bool v) {
    return enablePattern(PatternType::POSTNREPL, v);
  }
  Patterns &enableSoftMaxGradDirect(bool v) {
    return enablePattern(PatternType::SOFTMAXGRADDIRECT, v);
  }
  Patterns &enableSplitConvBias(bool v) {
    return enablePattern(PatternType::SPLITCONVBIAS, v);
  }
  Patterns &enableOpToIdentity(bool v) {
    return enablePattern(PatternType::OPTOIDENTITY, v);
  }
  Patterns &enableSubtractArg1GradOp(bool v) {
    return enablePattern(PatternType::SUBTRACTARG1GRADOP, v);
  }
  Patterns &enableMulArgGradOp(bool v) {
    return enablePattern(PatternType::MULARGGRADOP, v);
  }
  Patterns &enableReciprocalGradOp(bool v) {
    return enablePattern(PatternType::RECIPROCALGRADOP, v);
  }
  Patterns &enableDivArg0GradOp(bool v) {
    return enablePattern(PatternType::DIVARG0GRADOP, v);
  }
  Patterns &enableDivArg1GradOp(bool v) {
    return enablePattern(PatternType::DIVARG1GRADOP, v);
  }
  Patterns &enableSinGradOp(bool v) {
    return enablePattern(PatternType::SINGRADOP, v);
  }
  Patterns &enableCosGradOp(bool v) {
    return enablePattern(PatternType::COSGRADOP, v);
  }
  Patterns &enableTanOp(bool v) { return enablePattern(PatternType::TANOP, v); }
  Patterns &enableInPlace0(bool v) {
    return enablePattern(PatternType::INPLACE0, v);
  }

  std::vector<std::unique_ptr<Pattern>> getPatternList();

  friend std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

private:
  // Map of which settings are enabled, indexed by value of PatternType
  std::map<PatternType, bool> settings;
};

std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

} // namespace poponnx

#endif
