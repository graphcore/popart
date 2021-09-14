// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <parsedtensorid.hpp>
#include <regex>
#include <popart/error.hpp>
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
  auto splittedString = splitString(inputTId, sNameDelimiter);
  splittedString.pop_back();
  scopes = {splittedString.begin(), splittedString.end()};
}

void ParsedTensorId::parsePrefixes() {
  std::string idWithoutScopes = inputTId;
  if (!scopes.empty()) {
    auto lastScope = scopes.back();
    auto pos       = idWithoutScopes.find(lastScope);
    if (pos != std::string::npos) {
      idWithoutScopes.erase(0, pos + lastScope.length());
    }
  }

  // Static lambda initialization - r will only be initialized once
  const static std::regex r = []() {
    std::string re;
    for (auto prefix : reservedPrefixes()) {
      re += prefix + "|";
    }
    std::regex r{re};
    return r;
  }();

  for (auto it = std::sregex_iterator(
           idWithoutScopes.begin(), idWithoutScopes.end(), r);
       it != std::sregex_iterator();
       ++it) {
    std::smatch sm = *it;
    if (sm.length() > 0) {
      prefixes.push_back(sm.str());
    }
  }
}

void ParsedTensorId::parseName() {
  std::size_t pos;
  name = inputTId;
  if (!scopes.empty()) {
    // Remove the scopes
    auto lastScope = scopes.back();
    pos            = name.find(lastScope);
    if (pos != std::string::npos) {
      name.erase(
          0, pos + std::string(sNameDelimiter).length() + lastScope.length());
    }
  }

  if (!prefixes.empty()) {
    // Remove the prefixes
    auto lastPrefix = prefixes.back();
    pos             = name.find(lastPrefix);
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
    auto foundScopes = splitString(s.str(), sNameDelimiter);

    // Insert scopes
    foundScopes.insert(foundScopes.end(), scopes.begin(), scopes.end());
    scopes = {foundScopes.begin(), foundScopes.end()};
  }

  generateId();
  return tId;
}

TensorId ParsedTensorId::removeScope(const Scope &s) {
  auto foundScopes = splitString(s.str(), sNameDelimiter);

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

  generateId();
  return tId;
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

} // namespace popart
