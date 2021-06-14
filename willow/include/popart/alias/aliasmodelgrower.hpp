// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALIAS_ALIAS_MODEL_GROWER_HPP
#define GUARD_NEURALNET_ALIAS_ALIAS_MODEL_GROWER_HPP

#include <functional>
#include <memory>

#include <popart/alias/aliasmodel.hpp>

namespace popart {

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

#endif