#ifndef GUARD_NEURALNET_PATTERNS_HPP
#define GUARD_NEURALNET_PATTERNS_HPP

#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include <popart/logging.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/util.hpp>

#include <boost/optional.hpp>

namespace popart {

// FFS : add gcc like levels O2, O3, OS etc
enum class PatternsLevel { NONE, DEFAULT, ALL };

// This is a factory class which the patterns are registered with
class PreAliasPatternManager {

  PreAliasPatternManager() = default;

  // List of all registered pattern types
  std::vector<PreAliasPatternType> patterns;

  // Used to convert a string to a PreAliasPatternType (an enum class)
  std::map<std::string, PreAliasPatternType> stringToPreAliasPatternTypeMapping;

  // Used to construct the Patterns
  std::map<PreAliasPatternType,
           std::function<std::unique_ptr<PreAliasPattern>()>>
      factory;

  // Singleton
  static PreAliasPatternManager &getInstance() {
    static PreAliasPatternManager instance;
    return instance;
  }

public:
  // could add another parameter to set which level this pattern is in.
  static void
  registerPattern(PreAliasPatternType type,
                  std::string name,
                  std::function<std::unique_ptr<PreAliasPattern>()> func) {
    getInstance().patterns.push_back(type);
    getInstance().stringToPreAliasPatternTypeMapping.insert(
        std::pair<std::string, PreAliasPatternType>(name, type));
    getInstance().factory.insert(
        std::pair<PreAliasPatternType,
                  std::function<std::unique_ptr<PreAliasPattern>()>>(type,
                                                                     func));
  }

  static const std::vector<PreAliasPatternType> &getPatternList() {
    return getInstance().patterns;
  }

  static std::string getPatternName(PreAliasPatternType type) {
    // as there is no reverse mapping, we use the O(N) linear search here
    for (auto i : getInstance().stringToPreAliasPatternTypeMapping) {
      if (i.second == type) {
        return i.first;
      }
    }
    throw error("No `name' string for PatterType provided in getPatternName");
  }

  static boost::optional<PreAliasPatternType>
  convertPreAliasPatternType(std::string s) {
    auto it = getInstance().stringToPreAliasPatternTypeMapping.find(s);
    return boost::optional<PreAliasPatternType>{
        it != getInstance().stringToPreAliasPatternTypeMapping.end(),
        it->second};
  }

  static std::unique_ptr<PreAliasPattern>
  createPattern(PreAliasPatternType type) {
    auto it = getInstance().factory.find(type);
    if (it != getInstance().factory.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

// This class registers a lambda function to create a pattern with the
// PreAliasPatternManager
template <class PATTERN> class PatternCreator {
public:
  PatternCreator(PreAliasPatternType type, std::string name) {
    PreAliasPatternManager::registerPattern(
        type, name, [name]() -> std::unique_ptr<PreAliasPattern> {
          auto pattern = std::unique_ptr<PATTERN>(new PATTERN());
          pattern->initialise(name);
          return std::move(pattern);
        });
  }
};

// A class to hold which patterns are enabled/disabled
// FFS : Should this be renamed PatternSettings. i.e. it represents the graph
// 'optimization' settings
class Patterns {

public:
  Patterns(std::vector<PreAliasPatternType> types);

  Patterns(PatternsLevel level = PatternsLevel::DEFAULT);

  static Patterns create(std::vector<std::string> patterns);

  bool isPatternEnabled(PreAliasPatternType t);
  Patterns &enablePattern(PreAliasPatternType t, bool v);

  bool isPreUniReplEnabled() {
    return isPatternEnabled(PreAliasPatternType::PREUNIREPL);
  }
  bool isPostNReplEnabled() {
    return isPatternEnabled(PreAliasPatternType::POSTNREPL);
  }
  bool isSoftMaxGradDirectEnabled() {
    return isPatternEnabled(PreAliasPatternType::SOFTMAXGRADDIRECT);
  }
  bool isNlllWithSoftMaxGradDirectEnabled() {
    return isPatternEnabled(PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT);
  }
  bool isSplitConvBiasEnabled() {
    return isPatternEnabled(PreAliasPatternType::SPLITCONVBIAS);
  }
  bool isSplitGatherEnabled() {
    return isPatternEnabled(PreAliasPatternType::SPLITGATHER);
  }
  bool isOpToIdentityEnabled() {
    return isPatternEnabled(PreAliasPatternType::OPTOIDENTITY);
  }
  bool isSubtractArg1GradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::SUBTRACTARG1GRADOP);
  }
  bool isMulArgGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::MULARGGRADOP);
  }
  bool isReciprocalGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::RECIPROCALGRADOP);
  }
  bool isDivArg0GradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::DIVARG0GRADOP);
  }
  bool isDivArg1GradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::DIVARG1GRADOP);
  }
  bool isPowArg0GradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::POWARG0GRADOP);
  }
  bool isPowArg1GradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::POWARG1GRADOP);
  }
  bool isSinGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::SINGRADOP);
  }
  bool isCosGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::COSGRADOP);
  }
  bool isTanToSinOverCosEnabled() {
    return isPatternEnabled(PreAliasPatternType::TANTOSINOVERCOS);
  }
  bool isInPlaceEnabled() { return inplaceEnabled; }
  bool isUpdateInplacePrioritiesForIpuEnabled() {
    return updateInplacePrioritiesForIpuEnabled;
  }
  bool isSqrtGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::SQRTGRADOP);
  }
  bool isExpGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::EXPGRADOP);
  }
  bool isLogGradOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::LOGGRADOP);
  }
  bool isLogSoftmaxOpEnabled() {
    return isPatternEnabled(PreAliasPatternType::LOGSOFTMAXOP);
  }
  bool isGemmDecompositionEnabled() {
    return isPatternEnabled(PreAliasPatternType::GEMMDECOMPOSITION);
  }
  bool isNegativeOneScaleEnabled() {
    return isPatternEnabled(PreAliasPatternType::NEGATIVEONESCALE);
  }

  // The following methods are fluent allow you to
  // Pattens().enableInPlace0(false).
  //           enablePreUniRepl(true);
  Patterns &enablePreUniRepl(bool v) {
    return enablePattern(PreAliasPatternType::PREUNIREPL, v);
  }
  Patterns &enablePostNRepl(bool v) {
    return enablePattern(PreAliasPatternType::POSTNREPL, v);
  }
  Patterns &enableSoftMaxGradDirect(bool v) {
    return enablePattern(PreAliasPatternType::SOFTMAXGRADDIRECT, v);
  }
  Patterns &enableNlllWithSoftMaxGradDirect(bool v) {
    return enablePattern(PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT, v);
  }
  Patterns &enableSplitConvBias(bool v) {
    return enablePattern(PreAliasPatternType::SPLITCONVBIAS, v);
  }
  Patterns &enableSplitGather(bool v) {
    return enablePattern(PreAliasPatternType::SPLITGATHER, v);
  }
  Patterns &enableOpToIdentity(bool v) {
    return enablePattern(PreAliasPatternType::OPTOIDENTITY, v);
  }
  Patterns &enableSubtractArg1GradOp(bool v) {
    return enablePattern(PreAliasPatternType::SUBTRACTARG1GRADOP, v);
  }
  Patterns &enableMulArgGradOp(bool v) {
    return enablePattern(PreAliasPatternType::MULARGGRADOP, v);
  }
  Patterns &enableReciprocalGradOp(bool v) {
    return enablePattern(PreAliasPatternType::RECIPROCALGRADOP, v);
  }
  Patterns &enableDivArg0GradOp(bool v) {
    return enablePattern(PreAliasPatternType::DIVARG0GRADOP, v);
  }
  Patterns &enableDivArg1GradOp(bool v) {
    return enablePattern(PreAliasPatternType::DIVARG1GRADOP, v);
  }
  Patterns &enablePowArg0GradOp(bool v) {
    return enablePattern(PreAliasPatternType::POWARG0GRADOP, v);
  }
  Patterns &enablePowArg1GradOp(bool v) {
    return enablePattern(PreAliasPatternType::POWARG1GRADOP, v);
  }
  Patterns &enableSinGradOp(bool v) {
    return enablePattern(PreAliasPatternType::SINGRADOP, v);
  }
  Patterns &enableCosGradOp(bool v) {
    return enablePattern(PreAliasPatternType::COSGRADOP, v);
  }
  Patterns &enableTanToSinOverCos(bool v) {
    return enablePattern(PreAliasPatternType::TANTOSINOVERCOS, v);
  }
  Patterns &enableInPlace(bool v) {
    inplaceEnabled = v;
    return *this;
  }
  Patterns &enableUpdateInplacePrioritiesForIpu(bool v) {
    updateInplacePrioritiesForIpuEnabled = v;
    return *this;
  }
  Patterns &enableSqrtGradOp(bool v) {
    return enablePattern(PreAliasPatternType::SQRTGRADOP, v);
  }
  Patterns &enableExpGradOp(bool v) {
    return enablePattern(PreAliasPatternType::EXPGRADOP, v);
  }
  Patterns &enableLogGradOp(bool v) {
    return enablePattern(PreAliasPatternType::LOGGRADOP, v);
  }
  Patterns &enableLogSoftmaxOp(bool v) {
    return enablePattern(PreAliasPatternType::LOGSOFTMAXOP, v);
  }
  Patterns &enableGemmDecomposition(bool v) {
    return enablePattern(PreAliasPatternType::GEMMDECOMPOSITION, v);
  }
  Patterns &enableNegativeOneScale(bool v) {
    return enablePattern(PreAliasPatternType::NEGATIVEONESCALE, v);
  }

  std::vector<std::unique_ptr<PreAliasPattern>> getPreAliasList();

  friend std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

private:
  // Map of which settings are enabled, indexed by value of PreAliasPatternType
  std::map<PreAliasPatternType, bool> settings;

  // the one pattern which is not a PreAliasPattern
  bool inplaceEnabled{false};
  bool updateInplacePrioritiesForIpuEnabled{false};
};

std::ostream &operator<<(std::ostream &os, const Patterns &patterns);

} // namespace popart

#endif
