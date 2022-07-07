// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <numeric>
#include <parsedtensorid.hpp>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/scope.hpp>
#include <popart/tensornames.hpp>

#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include <vector>

namespace popart {

ParsedTensorId::ParsedTensorId(const TensorId &tId, const Ir &ir)
    : parsingTId(tId), tId_(tId) {

  auto lastScopeAndPrefixPos = extractElements(ir);

  // Sort tIdVec based on positionVec
  std::vector<std::size_t> indexVec(positionVec.size());
  std::iota(indexVec.begin(), indexVec.end(), 0);
  std::sort(indexVec.begin(),
            indexVec.end(),
            [this](const std::size_t &first, const std::size_t &second) {
              return positionVec[first] < positionVec[second];
            });
  std::vector<std::string> tmpVec;
  for (const auto &i : indexVec) {
    tmpVec.push_back(tIdVec.at(i));
  }
  tIdVec = std::move(tmpVec);
  // Sort positionVec as we want to figure out the tIdVec index of the last
  // prefix
  std::sort(positionVec.begin(), positionVec.end());

  // Set lastPrefixIndex
  std::size_t lastScopeIndex;
  auto lastScopePosIt = std::find(
      positionVec.cbegin(), positionVec.cend(), lastScopeAndPrefixPos.first);
  lastScopeIndex       = lastScopePosIt - positionVec.cbegin();
  auto lastPrefixPosIt = std::find(
      positionVec.cbegin(), positionVec.cend(), lastScopeAndPrefixPos.second);
  nextPrefixIndex = lastPrefixPosIt - positionVec.cbegin();

  if (nextPrefixIndex == 0) {
    // If we have scopes tIdVec.size() will be larger than 1
    // As we always add prefixes after scopes, we need to account for this
    if ((tIdVec.size() > 1)) {
      nextPrefixIndex = lastScopeIndex + 1;
    } else {
      nextPrefixIndex = 0;
    }
  } else {
    ++nextPrefixIndex;
  }

  // Generate the Id
  generateId();

  // No more need for these variables, so we free them
  positionVec.clear();
}

TensorId ParsedTensorId::addPrefix(const std::string &prefix) {
  // Add the prefix after the last prefix index
  auto tIdVecIt = std::next(tIdVec.begin(), nextPrefixIndex);
  tIdVec.insert(tIdVecIt, prefix);
  // Increment the lastPrefixIndex
  ++nextPrefixIndex;
  generateId();
  return tId_;
}

TensorId ParsedTensorId::removePrefixIfExist(const std::string &prefix) {
  auto tIdVecIt = std::find(tIdVec.cbegin(), tIdVec.cend(), prefix);
  if (tIdVecIt == tIdVec.cend()) {
    logging::warn("{} not found in {}", prefix, tId_);
  } else {
    tIdVec.erase(tIdVecIt);
    --nextPrefixIndex;
    generateId();
  }

  return tId_;
}

TensorId ParsedTensorId::addScope(const Scope &s) {
  if (!s.empty()) {
    if (!scopeExistInParsedTensorIdIr(s)) {
      throw error("Cannot add scope {} as it matches no Graphs in the Ir",
                  s.str());
    }
    // Insert the scopes in reverse order
    auto scopeNames = s.getScopeNames();
    for (auto it = scopeNames.rbegin(); it != scopeNames.rend(); ++it) {
      tIdVec.insert(tIdVec.cbegin(), *it + sNameDelimiter);
      ++nextPrefixIndex;
    }

    generateId();
  }

  return tId_;
}

TensorId ParsedTensorId::removeScope(const Scope &s) {
  if (!s.empty()) {
    if (!scopeExistInParsedTensorIdIr(s)) {
      throw error("Cannot remove scope {} as it matches no Graphs in the Ir",
                  s.str());
    }
    if (!scopeExistInParsedTensorId(s)) {
      throw error("Cannot remove scope {} as it matches no scopes in {}",
                  s.str(),
                  tId_);
    }

    for (const auto &scope : s.getScopeNames()) {
      auto hitIt =
          std::find(tIdVec.cbegin(), tIdVec.cend(), scope + sNameDelimiter);
      auto scopeIndex = hitIt - tIdVec.cbegin();
      if (scopeIndex != 0) {
        throw error(
            "Cannot remove scope {} from {} as it does not start with scope {}",
            *hitIt,
            tId_,
            *hitIt);
      } else {
        tIdVec.erase(hitIt);
        --nextPrefixIndex;
      }
    }
    generateId();
  }

  return tId_;
}

bool ParsedTensorId::scopeExistInParsedTensorId(const Scope &s) {
  if (!s.empty()) {
    // Search by string as the scope may not have it's own vector
    auto pos = tId_.find(s.str() + sNameDelimiter);
    if (pos == std::string::npos) {
      return false;
    } else {
      return true;
    }
  }
  return false;
}

bool ParsedTensorId::scopeExistInParsedTensorIdIr(const Scope &s) {
  bool scopeInIr = true;
  for (const auto &scope : s.getScopeNames()) {
    if (std::find(sortedIrScopesWithDelimiter.cbegin(),
                  sortedIrScopesWithDelimiter.cend(),
                  scope + sNameDelimiter) ==
        sortedIrScopesWithDelimiter.cend()) {
      scopeInIr = false;
    }
  }
  return scopeInIr;
}

bool ParsedTensorId::prefixExist(const std::string &p) {
  return std::find(tIdVec.cbegin(), tIdVec.cend(), p) != tIdVec.cend();
}

TensorId ParsedTensorId::getId() { return tId_; }

void ParsedTensorId::generateId() {
  if (tIdVec.size() == 1) {
    // From the constructor, we are guaranteed to have at least one element
    tId_ = tIdVec.at(0);
  } else {
    // We generate the id by accumulating the string of tIdVec
    tId_ = std::accumulate(
        std::next(tIdVec.begin()), // This will be the first b below
        tIdVec.end(),
        tIdVec.at(0), // start with first element
        [](std::string a, std::string b) { return a + b; });
  }
}

void ParsedTensorId::setSortedIrScopesWithDelimiter(const Ir &ir) {
  for (const auto &gIdAndGPointer : ir.getGraphs()) {
    auto const gId = gIdAndGPointer.first.str();
    if (!gId.empty()) {
      sortedIrScopesWithDelimiter.push_back(gIdAndGPointer.first.str() + "/");
    }
  }
  std::sort(sortedIrScopesWithDelimiter.begin(),
            sortedIrScopesWithDelimiter.end(),
            [](const std::string &first, const std::string &second) -> bool {
              return first.size() > second.size();
            });
}

std::pair<std::size_t, std::size_t>
ParsedTensorId::extractElements(const Ir &ir) {
  for (std::size_t i = 0; i < tId_.size(); ++i) {
    positionMap[i] = i;
  }

  // Sort the reserved prefixes
  std::vector<std::string> sortedPrefixes = reservedPrefixes();
  std::sort(sortedPrefixes.begin(),
            sortedPrefixes.end(),
            [](const std::string &first, const std::string &second) -> bool {
              return first.size() > second.size();
            });

  // Set the sorted IrScopes
  setSortedIrScopesWithDelimiter(ir);

  // Extract elements
  auto lastPositionOfScopeExtraction =
      extractElementsFromVector(sortedIrScopesWithDelimiter);
  auto lastPositionOfPrefixExtraction =
      extractElementsFromVector(sortedPrefixes);
  // We have now extracted the scope and prefixes
  // We will now extract the part of parsingTId which is neither scope nor
  // prefix We will do this by looping through the rest of the parsingTId string
  // and capture substrings which are consecutive in the input tId
  std::size_t startOfSubString  = 0;
  std::size_t consecutiveLength = 0;
  std::size_t currentIndex      = 0;
  if (!positionMap.empty()) {
    std::size_t previousIndex =
        positionMap.at(0) - 1; // Start at -1 to easen looping
    for (std::size_t i = 0; i < positionMap.size(); ++i) {
      currentIndex = positionMap.at(i);
      if (currentIndex != previousIndex + 1) {
        // Guard as we don't want empty strings in tVecInd
        if (!(consecutiveLength == 0)) {
          tIdVec.push_back(
              parsingTId.substr(startOfSubString, consecutiveLength));
          positionVec.push_back(positionMap.at(startOfSubString));
          startOfSubString  = startOfSubString + 1;
          consecutiveLength = 1;
        }
      } else {
        ++consecutiveLength;
      }
      previousIndex = currentIndex;
    }
    // Capture the last element
    tIdVec.push_back(parsingTId.substr(startOfSubString, consecutiveLength));
    positionVec.push_back(positionMap.at(startOfSubString));
  } else {
    // Special case if the input tId = ""
    tIdVec.push_back("");
    positionVec.push_back(0);
  }

  // No need for the parsingTId any longer
  parsingTId.clear();
  return std::make_pair(lastPositionOfScopeExtraction,
                        lastPositionOfPrefixExtraction);
}

std::size_t ParsedTensorId::extractElementsFromVector(
    const std::vector<std::string> &sortedElementsToExtract) {
  // The character position of tId where the last extraction occurred
  std::size_t lastPositionOfExtraction = 0;

  for (const auto &element : sortedElementsToExtract) {
    // Find the position
    auto pos = parsingTId.find(element);
    // We assume that an element only appears once
    if (pos != std::string::npos) {
      // Update the highest position
      if (positionMap.at(pos) > lastPositionOfExtraction) {
        lastPositionOfExtraction = positionMap.at(pos);
      }

      // Add it to tIdVec
      tIdVec.push_back(parsingTId.substr(pos, element.size()));
      positionVec.push_back(positionMap.at(pos));
      // Remove scope from parsingTId to speed-up searching
      parsingTId.erase(pos, element.size());

      // Update the position map
      // First erase the keys of the removed sequence
      for (size_t i = pos; i < pos + element.size(); ++i) {
        positionMap.erase(i);
      }
      // Then update the remaining keys after the removal
      if (!positionMap.empty()) {
        // We loop from the last removal point to the highest index in the
        // positionMap
        for (size_t i = pos + element.size();
             i < positionMap.rbegin()->first + 1;
             ++i) {
          auto it = positionMap.find(i);
          // Swap the value form the old key to a newly constructed one
          std::swap(positionMap[i - element.size()], it->second);
          // Delete the old key
          positionMap.erase(it);
        }
      }
    }
  }
  return lastPositionOfExtraction;
}

} // namespace popart
