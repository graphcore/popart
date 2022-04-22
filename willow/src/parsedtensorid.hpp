// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_PARSEDTENSORID_HPP
#define GUARD_PARSEDTENSORID_HPP

#include <cstddef>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class Ir;
class Scope;

/**
 * Parse a TensorId into scopes prefixes and names
 *
 * Scopes and prefixes can be added and removed to the object
 * The resulting TensorId is on the format <scopes><prefixes><names>
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
   * Parse the \a TensorId into scopes, prefixes and names
   * The \a scopes will be identified by the graphs in Ir
   * The \a prefixes will be identified by the prefixes in \see
   * reservedPrefixes() The \a name is whatever remains of the string
   *
   * Limitations:
   * 1.
   * It's assumed that the scopes are in the beginning of the \a TensorId and
   * separated by sNameDelimiter.
   * An ill formatted \a TensorId like ``foo_myScope_bar`` will be returned as
   * ``myScope/foo__bar`` without throwing a warning.
   *
   * \param tId_ The TensorId to be parsed
   * \param ir The ir to check for scopes
   */
  ParsedTensorId(const TensorId &tId_, const Ir &ir) : inputTId(tId_) {
    setIrScopes(ir);
    parse();
  }

  /**
   * Add a prefix to the back off other prefixes and regenerates \a tId
   *
   * \param prefix The prefix to be parsed
   * \return The TensorId with the added prefix
   */
  TensorId addPrefix(const std::string &prefix);
  /**
   * Remove a prefix if it's found in the vector of prefixes and regenerates \a
   * tId A warning is given if none is found
   *
   * \param prefix The prefix to remove
   * \return The TensorId with the possibly removed prefix
   */
  TensorId removePrefixIfExist(const std::string &prefix);

  /**
   * Add a \a Scope to the beginning of \a scopes and regenerates \a tId
   *
   * Note: This does not require that \a s is in the Ir
   *
   * \param s The Scope to be added
   * \return The TensorId with the added scope
   */
  TensorId addScope(const Scope &s);
  /**
   * Remove a \a Scope from the beginning of Scopes and regenerates \a tId
   *
   * An error is thrown if the \a Scope to be removed does not match the start
   * of Scopes
   *
   * Note: This does not require that \a s is in the Ir
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
  bool scopeExist(const Scope &s);
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
  TensorId getId() { return tId; }

private:
  /**
   * Get the scopes from the ir, and stores it in \a irScopes
   *
   * \param ir The Ir to search for the scopes in
   */
  void setIrScopes(const Ir &ir);
  /**
   * Parse the \a TensorId into scopes, prefixes and names
   * The \a scopes will be identified by the graphs in \a Ir
   * The \a prefixes will be identified by the prefixes in \see
   * reservedPrefixes() The \a name is whatever remains of the string
   */
  void parse();
  //! Parse the \a scopes (identified by the graphs in the \a Ir)
  void parseScopes();
  //! Parse the \a prefixes (identified by \see reservedPrefixes())
  void parsePrefixes();
  //! Parse the \a name (what is neither scopes nor prefixes)
  void parseName();

  /**
   * Return a vector of matches of \a potentialMatches found in \a s
   *
   * The resulting vector will be sorted after where the match is found in a
   * string. Example: std::string s = "foo_baz:Bar"; std::vector<std::string>
   * v{"Bar", "foo", "baz"}; auto result = findMatches(s, v);
   *     // result now contains {"foo", "baz", "Bar"}
   *
   * \param s String to search for matches in
   * \param potentialMatches Vector to match against \a inputTId
   * \return foundMatches Vector to store the result in
   */
  std::vector<std::string>
  findMatches(const std::string &s,
              const std::vector<std::string> &potentialMatches);

  //! Construct \a tId on the form <scopes><prefixes><name>
  void generateId();

  //! The scopes of the \a TensorId
  std::deque<std::string> scopes;
  //! The scopes found in the ir
  std::vector<std::string> irScopes;
  //! The prefixes of the \a TensorId
  std::vector<std::string> prefixes;
  //! The name of the \a TensorId
  std::string name;
  //! The original input \a TensorId
  TensorId inputTId;
  //! The \a TensorId after parsing and possible manipulations
  TensorId tId;
};

/**
 * Prune the input for overlaps between a begin element and another
 * begin+length element.
 *
 * The algorithm assumes that the map is sorted in ascending order and that
 * there is only one match per position.
 * ParsedTensorId::findMatches takes care of the last assumption.
 *
 * Example:
 * Assume that we have the following columns
 * The first column represents the element in strBeginAndStrLengths, whereas the
 * second column is for illustration purpose only
 *
 * {Begin, Length}    {Begin, End}
 * { 0, 2}            { 0,  2}
 * { 3, 6}            { 3,  9}
 * { 4, 4}            { 4,  8} <- overlaps with {3, 9}
 * { 5, 1}            { 5,  6} <- overlaps with {3, 9} and {4, 8}
 * {11, 2}            {11, 13}
 * {14, 2}            {14, 16}
 * {15, 1}            {15, 16} <- overlaps with {14, 16}
 * {21, 5}            {21, 26}
 *
 * Calling pruneOverlappedMatches on the above will result in
 *
 * {Begin, Length}    {Begin, End}
 * { 0, 2}            { 0,  2}
 * { 3, 6}            { 3,  9}
 * {11, 2}            {11, 13}
 * {14, 2}            {14, 16}
 * {21, 5}            {21, 26}
 *
 * \param strBeginAndStrLengths Map containing begin and lengths of matches
 */
void pruneOverlappedMatches(
    std::map<std::size_t, std::size_t> &strBeginAndStrLengths);

} // namespace popart

#endif // GUARD_PARSEDTENSORID_HPP
