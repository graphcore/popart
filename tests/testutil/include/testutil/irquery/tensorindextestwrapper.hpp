// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORINDEXTESTWRAPPER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORINDEXTESTWRAPPER_HPP_

#include "testutil/irquery/testwrapper.hpp"
#include <string>
#include <utility>

#include "popart/tensordebuginfo.hpp"

namespace popart {
class Ir;
class Tensor;

namespace irquery {

// Forward declaration.
class TensorTestWrapper;

/**
 * Object that can be used to execute test queries over an index-tensor pair
 * (e.g. an op input or output, or graph input or output).
 **/
class TensorIndexTestWrapper : public TestWrapper<std::pair<int, Tensor *>> {
public:
  /**
   * Constructor.
   **/
  TensorIndexTestWrapper(Ir &ir,
                         const std::pair<int, Tensor *> &tensorIndex,
                         const std::string &srcObjDescr,
                         const std::string &descrSingular,
                         const std::string &descrPlural);
  /**
   * Shorthand for `unwrap()->first`.
   * \return The index.
   **/
  int index();

  /**
   * Shorthand for `unwrap()->second->id`.
   * \return The tensor id.
   **/
  TensorId id();

  /**
   * Get tensor test wrapper.
   * \return A TensorTestWrapper for the tensor.
   **/
  TensorTestWrapper tensor();

private:
  // Value of Op->str() or Graph::getGraphString.
  std::string srcObjDescr;
  // Description of type of indices ("output", "input")
  std::string descrSingular;
  // Description of type of indices ("outputs", "inputs")
  std::string descrPlural;
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORINDEXTESTWRAPPER_HPP_
