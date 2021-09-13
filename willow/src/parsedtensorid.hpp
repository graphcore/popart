// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_PARSEDTENSORID_HPP
#define GUARD_PARSEDTENSORID_HPP

#include <popart/names.hpp>

#include <deque>
#include <string>
#include <vector>

namespace popart {

/**
 * Parses a TensorId into scopes prefixes and names
 *
 * Scopes and prefixes can be added and removed to the object
 * The resulting TensorId is on the format <scoeps><prefixes><names>
 *
 * Example:
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
   * Parses the \a TensorId into scopes, prefixes and names
   * The \a scopes will be identified by \a sNameDelimiter
   * The \a prefixes will be identified by the prefixes in \see
   * reservedPrefixes() The \a name is whatever remains of the string \param
   * tId_ The TensorId to be parsed
   */
  ParsedTensorId(const TensorId &tId_) : inputTId(tId_) { parse(); }

  /**
   * Adds a prefix to the back off other prefixes and regenerates \a tId
   * \param prefix The prefix to be parsed
   */
  TensorId addPrefix(const std::string &prefix);
  /**
   * Removes a prefix if it's found in the vector of prefixes and regenerates \a
   * tId A warning is given if none is found \param prefix The prefix to remove
   */
  TensorId removePrefixIfExist(const std::string &prefix);

  /**
   * Adds a \a Scope to the beginning of \a scopes and regenerates \a tId
   * \param s The Scope to be added
   */
  TensorId addScope(const Scope &s);
  /**
   * Removes a \a Scope from the beginning of Scopes and regenerates \a tId
   * An error is thrown if the \a Scope to be removed does not match the start
   * of Scopes
   */
  TensorId removeScope(const Scope &s);

  /**
   * Returns the \a TensorId on the form <scopes><prefixes><name>
   * \returns The TensorId
   */
  TensorId getId() { return tId; }

private:
  //! Parses the TensorId into \a scopes, \a prefixes and \a name
  void parse();
  //! Parses the \a scopes (identified by \a sNameDelimiter)
  void parseScopes();
  //! Parses the \a prefixes (identified by \see reservedPrefixes())
  void parsePrefixes();
  //! Parses the \a name (what is neither scopes nor prefixes)
  void parseName();

  //! Constructs \a tId on the form <scopes><prefixes><name>
  void generateId();

  //! The scopes of the \a TensorId
  std::deque<std::string> scopes;
  //! The prefixes of the \a TensorId
  std::vector<std::string> prefixes;
  //! The name of the \a TensorId
  std::string name;

  //! The original input \a TensorId
  TensorId inputTId;
  //! The \a TensorId after parsing and possible manipulations
  TensorId tId;
};

} // namespace popart

#endif // GUARD_PARSEDTENSORID_HPP
