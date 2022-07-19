// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_PARSEDTENSORID_HPP_
#define POPART_WILLOW_SRC_PARSEDTENSORID_HPP_

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class Ir;
class Scope;

/**
 * Parse a TensorId into scopes prefixes and names
 *
 * Scopes and prefixes can be added and removed to the object
 *
 * Example:
 *    Ir ir;
 *    ir.createGraph({"g1"});
 *    ir.createGraph({"g2"});
 *    ir.createGraph({"g3"});
 *
 *    TensorId tId = "g1/g2/Step___Accl___name";
 *    ParsedTensorId pTId(tId);
 *
 *    Scope graphScope1;
 *    graphScope1 = graphScope1 / "g1";
 *    Scope newGraphScope;
 *    newGraphScope = newGraphScope / "g3";
 *    pTId.removeScope(graphScope1);
 *    pTId.addScope(newGraphScope);
 *
 *    pTId.removePrefixIfExist(reservedAcclPrefix());
 *    pTId.addPrefix(reservedGradientPrefix());
 *
 *    pTId.getId()  // "g3/g2/Step___Gradient___name";
 */
class ParsedTensorId {
public:
  /**
   * Parse a \a TensorId to find the location of scopes and prefixes
   * The \a scopes will be identified by the graphs in Ir
   * The \a prefixes will be identified by the prefixes in
   * \see reservedPrefixes()
   *
   * Any TensorId will be parsed, however, the TensorIds are expected to be
   * on the form
   *
   * ``(<scope>/)*(<prefix>)*\w*``
   *
   * Limitation (by design):
   * - Scopes cannot be removed unless the come first in the \a TensorId
   *
   * \param tId The TensorId to be parsed
   * \param ir The ir to check for scopes
   */
  ParsedTensorId(const TensorId &tId, const Ir &ir);

  /**
   * Add a prefix to the back off other prefixes and regenerate \a tId_
   *
   * \param prefix The prefix to be parsed
   * \return The TensorId with the added prefix
   */
  TensorId addPrefix(const std::string &prefix);
  /**
   * Remove a prefix if it's found and regenerate \a tId_
   *
   * A warning is given if the prefix is not found
   *
   * \param prefix The prefix to remove
   * \return The TensorId with the possibly removed prefix
   */
  TensorId removePrefixIfExist(const std::string &prefix);

  /**
   * Add a \a Scope to the beginning of \a scopes and regenerate \a tId_
   *
   * \param s The Scope to be added
   * \return The TensorId with the added scope
   */
  TensorId addScope(const Scope &s);
  /**
   * Remove a \a Scope from the beginning of Scopes and regenerate \a tId_
   *
   * An error is thrown if the \a Scope to be removed does not match the start
   * of the \a tId_
   *
   * Note: This does not require \a s to be in the Ir
   *
   * \param s The Scope to be removed
   * \return The TensorId with the removed scope
   */
  TensorId removeScope(const Scope &s);

  /**
   * Return whether or not a scope is present in the ParsedTensorId
   *
   * Note: This does not require that \a s is in the Ir
   *
   * \param s The scope to check
   * \returns True if the scope is found
   */
  bool scopeExistInParsedTensorId(const Scope &s);
  /**
   * Return whether or not a scope is present in the ir of ParsedTensorId
   *
   * \param s The scope to check
   * \returns True if the scope is found
   */
  bool scopeExistInParsedTensorIdIr(const Scope &s);
  /**
   * Return whether or not a prefix is present in the ParsedTensorId
   *
   * \param p The prefix to check
   * \returns True if the prefix is found
   */
  bool prefixExist(const std::string &p);

  /**
   * Return the \a TensorId on the form <scopes><prefixes><name>
   *
   * \returns The TensorId
   */
  TensorId getId();

private:
  /// Construct \a tId_ from \a tIdVec
  void generateId();

  /**
   * Set the scopes from the ir, and store it to \a sortedIrScopesWithDelimiter
   *
   * \param ir The Ir to search for the scopes in
   */
  void setSortedIrScopesWithDelimiter(const Ir &ir);

  /**
   * Extract scopes and prefixes, and populate
   *
   * \param ir The Ir to search for the scopes in
   * \returns Pair containing the character position of the extraction of
   *  the last scope and the last prefix
   */
  std::pair<std::size_t, std::size_t> extractElements(const Ir &ir);

  /**
   * Extract elements from parsingTId matching input vector.
   *
   * The elements will be cut from parsingTId and stored in tIdVec.
   *
   * \param sortedElementsToExtract A vector sorted from the longest to the
   *  shortest which will be used to matches in parsingTId
   * \returns The last position of the original tIdVec where an element has
   *  been extracted
   */
  std::size_t extractElementsFromVector(
      const std::vector<std::string> &sortedElementsToExtract);

  /**
   * Map between character position of parsingTId and tId_
   *
   * This is needed as we "eat" the characters of parsingTId, and we need
   * to keep track of character position of tId_
   */
  std::map<std::size_t, std::size_t> positionMap;
  /// The input \a TensorId
  TensorId parsingTId;
  /// The current \a TensorId
  TensorId tId_;

  /// Vector representation of tIdVec
  std::vector<std::string> tIdVec;
  /// Used to sort the tIdVec
  std::vector<std::size_t> positionVec;

  /// The position of the last prefix in tIdVec
  std::size_t nextPrefixIndex;

  /// Scopes present in the IR
  std::vector<std::string> sortedIrScopesWithDelimiter;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_PARSEDTENSORID_HPP_
