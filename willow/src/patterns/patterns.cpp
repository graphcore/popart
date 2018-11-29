#include <poponnx/logging.hpp>
#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

Patterns::Patterns(PatternsLevel level) {

  switch (level) {
  case PatternsLevel::NONE: {
  } break;

  // The default set of patterns
  case PatternsLevel::DEFAULT:
  case PatternsLevel::ALL: {
    // right now we will enable all the options, maybe later there will be a
    // subset
    auto patternList = PatternManager::getPatternList();
    for (auto pattern : patternList) {
      settings.insert(std::pair<PatternType, bool>(pattern, true));
    }
  } break;
  }
}

Patterns::Patterns(std::vector<PatternType> types) {

  for (auto type : types) {
    settings.insert(std::pair<PatternType, bool>(type, true));
  }
}

Patterns Patterns::create(std::vector<std::string> strings) {
  Patterns patterns(PatternsLevel::NONE);

  for (auto p : strings) {
    auto type = PatternManager::convertPatternType(p);
    if (type) {
      patterns.settings.insert(std::pair<PatternType, bool>(*type, true));
    } else
      logging::ir::warn("Unknown pattern {}", p);
  }

  return patterns;
}

bool Patterns::isPatternEnabled(PatternType t) {

  auto it = settings.find(t);
  if (it != settings.end()) {
    return it->second;
  }

  return false;
}

Patterns &Patterns::enablePattern(PatternType t, bool v) {
  logging::ir::warn("Pattern {} {}", static_cast<int>(t), v);
  settings[t] = v;
  return *this;
}

std::vector<std::unique_ptr<Pattern>> Patterns::getPatternList() {

  std::vector<std::unique_ptr<Pattern>> patterns;

  for (auto p : settings) {
    if (p.second) {
      patterns.emplace_back(PatternManager::createPattern(p.first));
    }
  }

  return patterns;
}

std::ostream &operator<<(std::ostream &os, const Patterns &patterns) {

  for (auto setting : patterns.settings) {
    os << PatternManager::getPatternName(setting.first) << " ";
  }

  return os;
}

} // namespace poponnx
