#ifndef TEST_UTILS_IR_QUERY_HPP
#define TEST_UTILS_IR_QUERY_HPP

#include <functional>

#include <popart/graph.hpp>
#include <popart/op.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {
namespace testing {

enum class Require {
  // If the query result does not hold it constitutes a test failure.
  MustBeTrue = 0,
  // If the query result holds it constitutes a test failure.
  MustBeFalse = 1,
  // The test doesn't fail based on query result.
  Nothing = 2
};

/**
 * All query functions in this class take a testReq parameter. Leaving it
 * set to `Require::Nothing` will result a function that itself will never
 * trigger a test failure, simply returning a result.
 * ```
 * auto res = foo(Require::Nothing);
 * ```
 *
 * Calling `foo(Require::MustBeTrue)` is used for positive testing and is
 * functionally equivalent to (but has clearer failure messages than):
 * ```
 * auto res = foo(Require::Nothing);  // same as foo(Require::MustBeTrue);
 * BOOST_REQUIRE(res);
 * ```
 *
 * Calling `foo(Require::MustBeFalse)` is used for negative testing and is
 * functionally equivalent to (but has clearer failure messages than):
 * ```
 * auto res = foo(Require::Nothing); // same as foo(Require::MustBeFalse);
 * BOOST_REQUIRE(!res);
 * ```
 *
 * The reason to favour use of testReq arguments over explicit `BOOST_REQUIRE`
 * calls is that you get a more user-friendly failure message.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add (see unittest_irqueries.cpp).
 **/
class IrQuerier {
public:
  // Shorthands.
  using OpPred    = std::function<bool(Op *)>;
  using TensorIds = std::vector<TensorId>;

  /**
   * Determine if graph has an operation for which opPred yields true. If so,
   * return a pointer to said op, if not, return nullptr.
   * \param graph The graph to look for ops in.
   * \param opPred A predicate function mapping Op*s to bools.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return An Op* iff the op was found, else nullptr.
   **/
  Op *
  graphHasOp(Graph &graph, OpPred opPred, Require testReq = Require::Nothing);

  /**
   * Test if an op has an input at inIndex.
   * \param op The op to test.
   * \param inIndex The index to check.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if the op has an input at the specified index with the
   *specified id, else false.
   **/
  bool
  opHasInputAt(Op *op, InIndex inIndex, Require testReq = Require::Nothing);

  /**
   * Test if an op has an input with inId at index inIndex
   * \param op The op to test.
   * \param inIndex The index to check.
   * \param inId An optional TensorId value to match.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if the op has an input at the specified index with the
   *specified id, else false.
   **/
  bool opHasInputIdAt(Op *op,
                      InIndex inIndex,
                      TensorId inId,
                      Require testReq = Require::Nothing);

  /**
   * Test if an op's inputs contain set of tensor IDs (at any indices).
   * \param op The op to test.
   * \param inIds The tensor IDs to check for.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if there is an exact match between input tensor IDs of the op
   *     and inIds.
   **/
  bool
  opHasInputIds(Op *op, TensorIds inIds, Require testReq = Require::Nothing);

  /**
   * Test if an op has exactly a set of tensor IDs as input (at any indices). If
   * an ID appears twice it should be included in inIds twice.
   * \param op The op to test.
   * \param inIds The tensor IDs to check for.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if there is an exact match between input tensor IDs of the op
   *     and inIds.
   **/
  bool opHasExactInputIds(Op *op,
                          TensorIds inIds,
                          Require testReq = Require::Nothing);

  /**
   * Test if an op has an output at outIndex.
   * \param op The op to test.
   * \param outIndex The index to check.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if the op has an output at the specified outdex with the
   *    specified id, else false.
   **/
  bool
  opHasOutputAt(Op *op, OutIndex outIndex, Require testReq = Require::Nothing);

  /**
   * Test if an op has an output with outId at index outIndex.
   * \param op The op to test.
   * \param outIndex The index to check.
   * \param outId An optional TensorId value to match.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if the op has an output at the specified index with the
   *specified id, else false.
   **/
  bool opHasOutputIdAt(Op *op,
                       OutIndex outIndex,
                       TensorId outId,
                       Require testReq = Require::Nothing);

  /**
   * Test if an op's outputs contain set of tensor IDs (at any indices).
   * \param op The op to test.
   * \param outIds The tensor IDs to check for.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return True if there is an exact match between output tensor IDs of the op
   *     and outIds.
   **/
  bool
  opHasOutputIds(Op *op, TensorIds outIds, Require testReq = Require::Nothing);

  /**
   * Test if an op has exactly a set of tensor IDs as output (at any indices).
   *If an ID appears twice it should be included in outIds twice. \param op The
   *op to test. \param outIds The tensor IDs to check for. \param testReq A
   *parameter that defines when to generate a BOOST_REQUIRE failure, allowing
   *for positive and negative testing (and no testing). \return True if there is
   *an exact match between output tensor IDs of the op and outIds.
   **/
  bool opHasExactOutputIds(Op *op,
                           TensorIds outIds,
                           Require testReq = Require::Nothing);

protected:
  // Function that triggers a test failure with a message. It's basically a
  // wrapper around BOOST_REQUIRE that is exposed as protected so that this
  // class' behaviour can be tested.
  virtual void triggerTestFailure(std::string errorMsg);
};

} // namespace testing
} // namespace popart

#endif
