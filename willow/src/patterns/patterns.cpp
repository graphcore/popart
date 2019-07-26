#include <popart/logging.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

Patterns::Patterns(PatternsLevel level) {

  // step 1 : adding patterns to run
  switch (level) {

  // add all of the patterns
  case PatternsLevel::DEFAULT:
  case PatternsLevel::ALL: {
    auto patternList = PreAliasPatternManager::getPatternList();
    for (auto pattern : patternList) {
      settings.insert(std::pair<PreAliasPatternType, bool>(pattern, true));
    }
    inplaceEnabled = true;
    break;
  }

  // add none of the patterns
  case PatternsLevel::NONE: {
    break;
  }
  }

  // step 2 : removing patterns to run (turning off logging)
  auto prevLogLevel = getLogLevel(logging::Module::pattern);
  setLogLevel(logging::Module::pattern, logging::Level::Off);
  switch (level) {
  case PatternsLevel::DEFAULT: {
    enableSplitGather(false);
    break;
  }

  case PatternsLevel::ALL:
  case PatternsLevel::NONE: {
    break;
  }
  }
  setLogLevel(logging::Module::pattern, prevLogLevel);
}

Patterns::Patterns(std::vector<PreAliasPatternType> types) {

  for (auto type : types) {
    settings.insert(std::pair<PreAliasPatternType, bool>(type, true));
  }
}

Patterns Patterns::create(std::vector<std::string> strings) {
  Patterns patterns(PatternsLevel::NONE);

  for (auto p : strings) {
    if (p == "InPlace") {
      patterns.enableInPlace(true);
    } else {
      auto type = PreAliasPatternManager::convertPreAliasPatternType(p);
      if (type) {
        patterns.settings.insert(
            std::pair<PreAliasPatternType, bool>(*type, true));
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

bool Patterns::isPatternEnabled(PreAliasPatternType t) {

  auto it = settings.find(t);
  if (it != settings.end()) {
    return it->second;
  }

  return false;
}

Patterns &Patterns::enablePattern(PreAliasPatternType t, bool v) {
  logging::pattern::warn("Pattern {} {}", static_cast<int>(t), v);
  settings[t] = v;
  return *this;
}

std::vector<std::unique_ptr<PreAliasPattern>> Patterns::getPreAliasList() {

  std::vector<std::unique_ptr<PreAliasPattern>> patterns;

  for (auto p : settings) {
    if (p.second) {
      patterns.emplace_back(PreAliasPatternManager::createPattern(p.first));
    }
  }

  return patterns;
}

std::ostream &operator<<(std::ostream &os, const Patterns &patterns) {

  for (auto setting : patterns.settings) {
    os << PreAliasPatternManager::getPatternName(setting.first) << " ";
  }

  return os;
}

} // namespace popart
