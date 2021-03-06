// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_TENSOR_INDEX_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_TENSOR_INDEX_TEST_WRAPPER_HPP

#include <functional>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>

#include <popart/vendored/optional.hpp>

#include <testutil/irquery/require.hpp>
#include <testutil/irquery/testwrapper.hpp>

namespace popart {
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

#endif
