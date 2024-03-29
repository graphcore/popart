// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODELGROWER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODELGROWER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "popart/names.hpp"

namespace popart {
class AliasModel;
class Graph;
class Op;
class Tensor;

/**
 * An enum type that determines whether topological constraints are added to
 * an alias model.
 **/
enum class DataDependenciesOnly {
  // Only add data constraints.
  Yes,
  // Add data constraints and additional topological constraints.
  No
};

/**
 * Class that contains some methods for creating `AliasModel` instances via a
 * `AliasModelGrowInterface`. It takes such an interface by reference and
 * grows it by calling, e.g., `Op::growAliasModel` on certain specific ops.
 **/
class AliasModelGrower final {
public:
  /**
   * Grow the default AliasModel.
   **/
  AliasModelGrower(AliasModel &aliasModel);

  /**
   * Get non-owning reference to grown AliasModel.
   **/
  AliasModel &getAliasModelRef();

  /**
   * Return owning AliasModel associated with this instance. Calling this
   * function leaves the grower without an AliasModel and growing further grow
   * functions without such a model will raise an exception.
   **/
  std::unique_ptr<AliasModel> getAliasModel();

  /**
   * Set the AliasModel we are growing.
   **/
  void setAliasModel(std::unique_ptr<AliasModel> aliasModel);

  /**
   * Grow an alias model that contains all tensors in a PopART Graph. This
   * mapping will include every PopART op and Tensor in the Graph.
   * \param graph The PopART Graph object to construct a mapping for.
   * \param dataDepsOnly Flag to indicate whether to add only data dependencies
   *     or whether to also add topocological constraints.
   **/
  void growFullGraph(const Graph &graph, DataDependenciesOnly dataDepsOnly);

  /**
   * Construct a mapping from tensors in a PopART Graph to an alias model that
   * is guaranteed to contain a mapping for any tensor that alias the
   * `tensorId` parameter (and ops that separate them) but may also contain
   * other tensors that do not alias it.
   *
   * The purpose of this function is to provide an alternative to
   * `getFullAliasModel` for when you do not require a whole mapping.
   *
   * \param graph The PopART Graph object to construct a mapping for.
   * \param tensorId The PopART Tensor used to determine which part of the
   *PopART graph to create a mapping for. \param dataDepsOnly Flag to indicate
   *whether to add only data dependencies or whether to also add topocological
   *constraints.
   **/
  void growPartialGraph(const Graph &graph,
                        const TensorId &tensorId,
                        DataDependenciesOnly dataDepsOnly);

  /**
   * Create an \c aliasModel for each graph and run the poprithms
   * ambiguity checker on it to see if there are any potential inplacing
   * ambiguities. See \see poprithms::memory::inplace::Graph::AmbiguityStatus
   * for more info.
   *
   * This can only detect if there is a *potential* ambiguity. For example:
   *
   * 1.
   *  \code
   * a <- init():
   * c <- init();
   * b <- a.add_(c);
   * c <- a.add_(c);
   * \endcode
   * if the last operation were a.mul_(c) there would be a genuine ambiguity as
   * `a + (b * c) != (a + b) * c`, but `a + (b + c) == (a + b) + c`. Poprithms
   * doesn't know if it's a mul_ or an add_ though (as poprithms will just see
   * it as a binary inplace modifying op), so it reports a potential ambiguity.
   *
   * 2.
   * \code
   * a <- init();
   * b <- a.relu_(); // relu inplace
   * c <- a.relu_(); // if this were sqrt_, there would be a genuine ambiguity.
   * \endcode
   * or more likely to come up in practise:
   *
   * 3.
   * \code
   * a <- init();
   * b <- init();
   * d <- a.add_(5); // a's value changes.
   * d <- a.copyFrom_(b); // copy from b to a.
   * e <- d.add(5);
   * \endcode
   *
   * The issue with this last case is that poprithms does not distinguish
   * between updates based on existing values, and updates to completely new
   * values.
   *
   * The general rule is as follows: If a tensor 'a' is consumed by an op 'm'
   * which modifies it, and 'a' is aliased to another tensor 'b' which is
   * consumed by an op 'c' which reads the value of 'b', then unless there
   * is a constraint between 'm' and 'c', the value of 'b' is ambiguous. By
   * 'reads' we include all ops which are not simply view-changers, or ops
   * like 'shape' which don't use the numerical values of the input.
   *
   * \returns true If a potential ambiguity is detected.
   * \returns false Otherwise.
   */
  bool containsAmbiguity() const;

  /**
   * Create an error string for the given AliasModel.
   *
   * \returns std::string The error string with info on the ops and tensors in
   * question.
   */
  std::string ambiguitySummary(const Graph &graph, AliasModel &) const;

private:
  /**
   * Data type that dictates whether we check at runtime whether a tensor that
   *is produced by an op may be added via `insertTensor`. When growing the alias
   * model for a full graph you would expect these tensors to be added by
   *growing their producer.
   **/
  enum class AllowInsertingProducedTensors {
    // Produced tensors may be added via `insertTensor`.
    Yes = 0,
    // Producer tensors must be added by their producer.
    No
  };

  /// Add a tensor to the AliasModel.
  void
  addTensor(const Graph &graph, Tensor *t, AllowInsertingProducedTensors allow);

  /// Add an Op to the AliasModel.
  void
  addAliaserOp(const Graph &graph, Op *op, AllowInsertingProducedTensors allow);

  /// Add topological constraints to the AliasModel.
  void addAliaserConstraints(const Graph &graph,
                             const std::vector<Op *> &opSubset);

  // The grow interface reference.
  std::reference_wrapper<AliasModel> aliasModel;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODELGROWER_HPP_
