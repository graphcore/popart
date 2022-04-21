// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <parsedtensorid.hpp>
#include <regex>
#include <string>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/scope.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

namespace popart {

void ParsedTensorId::parse() {
  parseScopes();
  parsePrefixes();
  parseName();
  generateId();
}

void ParsedTensorId::parseScopes() {
  auto scopesVector = findMatches(inputTId, irScopes);
  scopes            = {scopesVector.begin(), scopesVector.end()};
}

void ParsedTensorId::parsePrefixes() {
  prefixes = findMatches(inputTId, reservedPrefixes());
}

void ParsedTensorId::parseName() {
  std::size_t pos;
  name = inputTId;
  if (!scopes.empty()) {
    // Remove the scopes
    auto lastScope = scopes.back();
    pos            = name.rfind(lastScope);
    if (pos != std::string::npos) {
      name.erase(
          0, pos + std::string(sNameDelimiter).length() + lastScope.length());
    }
  }

  if (!prefixes.empty()) {
    // Remove the prefixes
    auto lastPrefix = prefixes.back();
    pos             = name.rfind(lastPrefix);
    if (pos != std::string::npos) {
      name.erase(0, pos + lastPrefix.length());
    }
  }
}

TensorId ParsedTensorId::addPrefix(const std::string &prefix) {
  prefixes.push_back(prefix);
  generateId();
  return tId;
}

TensorId ParsedTensorId::removePrefixIfExist(const std::string &prefix) {
  if (!prefixes.empty()) {
    auto it = std::find(prefixes.begin(), prefixes.end(), prefix);
    if (it != prefixes.end()) {
      prefixes.erase(it);
    } else {
      logging::warn("{} not found in {}", prefix, tId);
    }
  }
  generateId();

  return tId;
}

TensorId ParsedTensorId::addScope(const Scope &s) {
  if (!s.empty()) {
    // Extract the scopes
    auto foundScopes = findMatches(s.str(), irScopes);
    if (foundScopes.empty()) {
      throw error("Cannot add scope {} as it matches no Graphs in the Ir",
                  s.str());
    }
    // Insert scopes
    foundScopes.insert(foundScopes.end(), scopes.begin(), scopes.end());
    scopes = {foundScopes.begin(), foundScopes.end()};
  }

  generateId();
  return tId;
}

TensorId ParsedTensorId::removeScope(const Scope &s) {
  if (!s.empty()) {
    auto foundScopes = findMatches(s.str(), irScopes);
    if (foundScopes.empty()) {
      throw error("Cannot remove scope {} as it matches no Graphs in the Ir",
                  s.str());
    }

    auto sScopeIt        = foundScopes.begin();
    const auto sScopeEnd = foundScopes.end();

    auto tScopeIt        = scopes.begin();
    const auto tScopeEnd = scopes.end();

    for (; sScopeIt != sScopeEnd; ++sScopeIt, ++tScopeIt) {
      if (tScopeIt == tScopeEnd || *sScopeIt != *tScopeIt) {
        throw error(
            "Cannot remove scope from {} as it does not start with scope {}",
            tId,
            *sScopeIt);
      } else {
        scopes.pop_front();
      }
    }
  }

  generateId();
  return tId;
}

bool ParsedTensorId::scopeExist(const Scope &s) {
  if (!s.empty()) {
    auto sVec        = findMatches(s.str(), irScopes);
    auto foundScopes = findMatches(tId, sVec);

    if (!foundScopes.empty()) {
      for (auto const &fs : foundScopes) {
        if (std::find(scopes.begin(), scopes.end(), fs) == scopes.end()) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

bool ParsedTensorId::prefixExist(const std::string &p) {
  return std::find(prefixes.begin(), prefixes.end(), p) != prefixes.end();
}

void ParsedTensorId::setIrScopes(const Ir &ir) {
  for (const auto &gIdAndGPointer : ir.getGraphs()) {
    auto const gId = gIdAndGPointer.first.str();
    if (!gId.empty()) {
      irScopes.push_back(gIdAndGPointer.first.str());
    }
  }
}

std::vector<std::string>
ParsedTensorId::findMatches(const std::string &s,
                            const std::vector<std::string> &potentialMatches) {
  std::vector<std::string> matches;
  // Store begin and length of the match in s
  std::map<std::size_t, std::size_t> strBeginAndStrLengths;
  for (const auto &pm : potentialMatches) {
    // Find all matches
    auto pos = s.find(pm);
    while (pos != std::string::npos) {
      // If the pos is already present in the map we will choose the longest
      auto it = strBeginAndStrLengths.find(pos);
      if (it == strBeginAndStrLengths.end()) {
        strBeginAndStrLengths[pos] = pm.length();
      } else {
        auto length = it->second > pm.length() ? it->second : pm.length();
        strBeginAndStrLengths[pos] = length;
      }
      pos = s.find(pm, pos + 1);
    }
  }

  pruneOverlappedMatches(strBeginAndStrLengths);

  // Extract in correct order
  for (const auto &beginAndLen : strBeginAndStrLengths) {
    matches.push_back(s.substr(beginAndLen.first, beginAndLen.second));
  }

  return matches;
}

void ParsedTensorId::generateId() {
  tId = "";
  // Add the scopes
  for (const auto &scope : scopes) {
    tId += scope + sNameDelimiter;
  }
  // Add the prefixes
  for (const auto &prefix : prefixes) {
    tId += prefix;
  }
  // Add the name
  tId += name;
}

void pruneOverlappedMatches(
    std::map<std::size_t, std::size_t> &strBeginAndStrLengths) {
  // Populate the absolute end point
  std::vector<std::size_t> strEnd;
  for (const auto &sl : strBeginAndStrLengths) {
    strEnd.push_back(sl.first + sl.second);
  }
  // Check if there is an overlap
  std::set<std::size_t> keysToPrune;
  // As map keys are sorted ascending order by default, that means that an
  // overlap must come after the element it overlaps. Hence we can iterate
  // strBeginAndStrLengths and strEnd in reverse order to find the overlaps We
  // will compare the current match begin with the previous match end
  auto matchBeginLRIt = strBeginAndStrLengths.rbegin();
  auto matchEndRIt    = strEnd.rbegin() + 1;
  auto itEnd          = strEnd.rend();

  if (matchBeginLRIt != strBeginAndStrLengths.rend()) {
    while (matchEndRIt != itEnd) {
      // In human words:
      // For two substrings A and B, where B starts before A
      // if A.start < B.end:
      //   there is overlap, prune A as it is fully contained in B
      if (matchBeginLRIt->first < *matchEndRIt) {
        keysToPrune.insert(matchBeginLRIt->first);
      }
      ++matchBeginLRIt;
      ++matchEndRIt;
    }
  }

  // Erase keys
  for (const auto &keyToPrune : keysToPrune) {
    auto it = strBeginAndStrLengths.find(keyToPrune);
    strBeginAndStrLengths.erase(it);
  }
}

} // namespace popart
