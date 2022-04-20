// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_OPS_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_OPS_TEST_WRAPPER_HPP

#include "testutil/irquery/optestwrapper.hpp"
#include "testutil/irquery/require.hpp"
#include "testutil/irquery/testwrapper.hpp"
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace popart {
class Ir;

namespace irquery {

/**
 * This class wraps around a set of Ops so as to make testing of it easier.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add.
 */
class OpsTestWrapper : public TestWrapper<std::vector<Op *>> {
public:
  // Shorthands.
  template <typename OP = Op>
  using OpPred = std::function<bool(OpTestWrapper<OP> &op)>;

  /**
   * Constructor.
   **/
  OpsTestWrapper(Ir &ir,
                 const std::vector<Op *> &ops,
                 const std::string &srcObjDescr);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Determine if set of ops includes an op of template type OP (defaulted to
   * base class popart::Op). If so, return a test wrapper for said op, if not,
   * return nullptr.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A test wrapper for the op iff the op was found.
   *     Note that if `testReq` is set to `Require::MustBeTrue` this function is
   *     guaranteed to either throw an exception or return a non-defaulted
   *     optional.
   **/
  template <
      typename OP       = Op,
      typename enableif = std::enable_if_t<std::is_base_of<Op, OP>::value>>
  nonstd::optional<OpTestWrapper<OP, enableif>>
  hasOp(Require testReq = Require::Nothing);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Determine if set of ops includes an op of template type OP (defaulted to
   * base class popart::Op) for which opPred yields true. If so, return a test
   * wrapper for said op, if not, return nullptr.
   * \param opPred A predicate function mapping Op*s to bools.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A test wrapper for the op iff the op was found.
   *     Note that if `testReq` is set to `Require::MustBeTrue` this function is
   *     guaranteed to either throw an exception or return a non-defaulted
   *     optional.
   **/
  template <
      typename OP       = Op,
      typename enableif = std::enable_if_t<std::is_base_of<Op, OP>::value>>
  nonstd::optional<OpTestWrapper<OP, enableif>>
  hasOp(OpPred<OP> opPred, Require testReq = Require::Nothing);

private:
  // Description of whatever contains the op.
  std::string srcObjDescr;
};

} // namespace irquery
} // namespace popart

#endif
