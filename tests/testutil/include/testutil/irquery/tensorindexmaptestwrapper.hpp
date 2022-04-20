// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_TENSOR_INDEX_MAP_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_TENSOR_INDEX_MAP_TEST_WRAPPER_HPP

#include "require.hpp"
#include "testutil/irquery/testwrapper.hpp"
#include <map>
#include <string>
#include <vector>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class Ir;
class Tensor;

namespace irquery {
class TensorIndexTestWrapper;
/**
 * Object that can be used to execute test queries over a mapping from indices
 * to tensor IDs (e.g. op inputs or outputs, or graph inputs or outputs).
 **/
class TensorIndexMapTestWrapper : public TestWrapper<std::map<int, Tensor *>> {
public:
  /**
   * Constructor.
   **/
  TensorIndexMapTestWrapper(Ir &ir,
                            const std::map<int, Tensor *> &tensorIndexMap,
                            const std::string &srcObjDescr,
                            const std::string &mapTypeDescrSingular,
                            const std::string &mapTypeDescrPlural);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Query whether a mapping from indices to tensor IDs contains a specific id.
   * \param map The source of tensor ids to check.
   * \param id The id to check.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A TensorIndexTestWrapper if the tensor index map has an entry with
   *     the specified id, else a default
   *     nonstd::optional<TensorIndexTestWrapper>.
   *     Note that if `testReq` is set to `Require::MustBeTrue` this function is
   *     guaranteed to either throw an exception or return a non-defaulted
   *     optional.
   **/
  nonstd::optional<TensorIndexTestWrapper>
  hasId(const TensorId &id, Require testReq = Require::Nothing);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Query whether a mapping from indices to tensor IDs contains has some id
   * at a specific index.
   * \param map The source of tensor ids to check.
   * \param index The index to check.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A TensorIndexTestWrapper if the tensor index map has an entry at
   *     the specified index, else a default
   *     nonstd::optional<TensorIndexTestWrapper>.
   *     Note that if `testReq` is set to `Require::MustBeTrue` this function is
   *     guaranteed to either throw an exception or return a non-defaulted
   *     optional.
   **/
  nonstd::optional<TensorIndexTestWrapper>
  hasIndex(int index, Require testReq = Require::Nothing);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Query whether a mapping from indices to tensor IDs contains has a specific
   * id at a specific index.
   * \param map The source of tensor ids to check.
   * \param index The index to check.
   * \param id The id to check.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A TensorIndexTestWrapper if the tensor index map has an entry at
   *     the specified index with the specified id, else a default
   *     nonstd::optional<TensorIndexTestWrapper>.
   *     Note that if `testReq` is set to `Require::MustBeTrue` this function is
   *     guaranteed to either throw an exception or return a non-defaulted
   *     optional.
   **/
  nonstd::optional<TensorIndexTestWrapper>
  hasIdAtIndex(int index,
               const TensorId &id,
               Require testReq = Require::Nothing);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Query whether a mapping from indices to tensor IDs contains a set of
   * tensor IDs (ignoring indices and order).
   * \param map The source of tensor ids to check.
   * \param ids The tensor IDs to match.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if there is an exact match between input tensor IDs of the op
   *     and inIds.
   **/
  bool containsIds(const std::vector<TensorId> &ids,
                   Require testReq = Require::Nothing);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Query whether a mapping from indices to tensor IDs exactly matches a set of
   * tensor IDs. This check ignores indices and order but takes into account
   * multiplicity -- if an ID appears twice in the map it should be included in
   * ids twice, too.
   * \param map The source of tensor ids to check.
   * \param ids The tensor IDs to match.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if there is an exact match between input tensor IDs of the op
   *     and inIds.
   **/
  bool hasExactIds(const std::vector<TensorId> &ids,
                   Require testReq = Require::Nothing);

private:
  // Value of Op->str() or Graph::getGraphString.
  std::string srcObjDescr;
  // Description of type of indices ("output", "input")
  std::string mapTypeDescrSingular;
  // Description of type of indices ("outputs", "inputs")
  std::string mapTypeDescrPlural;
};

} // namespace irquery
} // namespace popart

#endif
