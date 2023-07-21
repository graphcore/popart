// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODEL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODEL_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/inplace/graph.hpp>

#include "popart/error.hpp"
#include "popart/names.hpp"

namespace popart {

// Forward declaration.
class Op;
class Tensor;

/**
 * A container for the poprithms::memory::inplace::Graph which corresponds to a
 * PopART Graph. It contains the poprithms Graph, and mappings between PopART
 * Tensors and Ops, and their poprithms equivalents.
 * */
class AliasModel {
public:
  using PoprithmsTensorId = poprithms::memory::inplace::TensorId;
  using PoprithmsOpId     = poprithms::memory::inplace::OpId;

  AliasModel();

  ~AliasModel() = default;

  /**
   * load factor used for hash map containers
   */
  static constexpr int loadFactor = 0.5;

  /**
   * Set PopART graph
   */
  void setGraph(const popart::Graph *graph);

  /**
   * Register that a poprithms Tensor and a popart Tensor correspond to each
   * other. In addition to registering the Tensor correspondence, the Ops which
   * produce the respective Tensors are registered to be corresponding.
   *
   * \param poprithmsTensor The Tensor in the poprithms Graph.
   *
   * \param popartTensor The Tensor in the PopART Graph.
   * */
  void insertTensor(const PoprithmsTensorId &poprithmsTensor,
                    const Tensor &popartTensor);

  /**
   * Register that a poprithms Op and a popart Op correspond.
   *
   * Note that multiple poprithms Ops can correspond to a single popart Op.
   * */
  void insertOp(PoprithmsOpId, OpId);

  /**
   *
   * \param op A PopART Op, which might have multiple inputs, and whose output
   *           is a modifies alias of its input at index 0.
   *
   * This method performs the following steps:
   *
   * (1) inserts an aliasGate which is open at index 0
   * (2) appends a modify to the output aliasGate created in (1)
   * (3) registers that op.output(0) match the output of (2)
   * (4) registers that the poprithms ops created at (1) and (2) correspond to
   *     #op.
   *  */

  void insertUnaryModifier0(const Op &op);

  /**
   * As per insertUnaryModifier0, but the input index may be different from 0.
   * */
  void insertUnaryModifier(const Op &, InIndex);

  /**
   *
   * \param op A PopART Op with 2 inputs.
   *
   * This method performs the following steps:
   *
   * (1) inserts an aliasGate whose inputs are the 2 poprithms Tensors
   *     corresponding to the 2 inputs of #op. The alias gate is open at the
   *     index which #op aliases through, if any.
   *
   * (2) appends a modify to the output of the aliasGate created at (1)
   *
   * (3) registers that the poprithms ops (1) and (2) correspond to #op.
   *
   * Diagramatically, for the PopART Op:
   *
   *   input0 ... input1
   *       \      /
   *          op
   *          |
   *        output0
   *
   * This method creates the following poprithms subgraph:
   *
   *   input0 ... input1
   *       \       /
   *        aliasGate
   *           |
   *         modify
   *           |
   *        output0
   *
   *        */
  void insertBinaryModifier(const Op &op);

  /**
   * \param op A PopART Op with 2 or more inputs.
   * \param numInputs The number of inputs
   *
   * The method is the same as insertBinaryModifier except for allowing a
   * larger number of inputs than 2.
   * */
  void insertNG2aryModifier(const Op &op, unsigned int numInputs);

  /**
   * This method performs the following steps:
   *
   * (1) adds an aliasGate whose (unique) unput is viewChangeOut,
   *
   * (2) registers that the output of the aliasGate corresponds to the PopART
   *     Tensor #t.
   *
   * (3) registers that the creator of t (if there is any) corresponds to 2
   *     poprithms ops: the creator of viewChangeOut and the aliasGate created
   *     at (1).
   *
   * \param viewChangeOut This is a Tensor which is the output of a view
   *                      changing Op, such as reshape and dimShuffle.
   *
   * \param t This PopART Tensor is the output of the corresponding PopART view
   *          changing Op.
   *
   * \param isOutplace This boolean determines if the AliasGate created at (1)
   *                   should be open or closed. If isOutplace is true, then the
   *                   AliasGate will be closed.
   * */
  void insertViewChange(PoprithmsTensorId viewChangeOut,
                        const Tensor &t,
                        bool isOutplace);

  /**
   * Replace all appearances of #oldId in all maps between PopART and poprithms,
   * with #newId. This is useful when, for example, an Op is replaced in the
   * PopART Graph during the inplacing transformation.
   * */
  void update(OpId oldId, OpId newId);

  /**
   * \return The TensorId corresponding to a poprithms TensorId.
   * */
  TensorId getTensorId(const PoprithmsTensorId &id) const;
  bool contains(const PoprithmsTensorId &) const;

  /**
   * \return The poprithms TensorId corresponding to a TensorId.
   * */
  PoprithmsTensorId getPoprithmsTensorId(const TensorId &id) const;
  bool contains(const TensorId &) const;

  /**
   * \return The OpId corresponding to a poprithms OpId.
   * */
  OpId getOpId(PoprithmsOpId) const;
  bool contains(PoprithmsOpId) const;

  /**
   * \return The ID of the AliasGate in the poprithms Graph, which corresponds
   *         to the PopART Op #opId. If no such AliasGate exists, an error is
   *         thrown.
   * */
  PoprithmsOpId getGate(OpId opId) const;

  /**
   * \return The poprithms OpIds which correspond to a PopART OpId. It is
   *         possible for 1 PopART Op to correspond to multiple poprithms Ops.
   * */
  std::vector<PoprithmsOpId> getAll(OpId) const;
  bool contains(OpId) const;

  /**
   * Get all aliases for a tensor for this given model.
   *
   * Returned tensors include the argument #t, if it is non-empty.
   **/
  std::vector<Tensor *> allAliases(const Tensor &t) const;

  /**
   * \return true if all of the 'allocation' elements of \a sub and are also
   *         in \a super.
   * */
  bool contains(const Tensor &super, const Tensor &sub) const;

  /**
   * The poprithms Graph
   * */
  poprithms::memory::inplace::Graph g;

  /**
   * The PopART graph reference
   */
  popart::Graph *thisGraph = nullptr;

private:
  // using a simpler hasher which is faster than std::hash derivatives and that
  // is safe enough in this case.
  struct PoprithmsTensorIdSteadyHasher {
    size_t operator()(const PoprithmsTensorId &id) const {
      // cast poprithm typed integer (int64_t) to size_t
      if (id.opId().get() < 0) {
        throw error("Unexpected negative value!");
      }
      return static_cast<size_t>(id.opId().get());
    }
  };

  struct PoprithmsOpIdHasher {
    size_t operator()(const PoprithmsOpId &opid) const {
      // cast poprithm typed integer (int64_t) to size_t
      if (opid.get() < 0) {
        throw error("Unexpected negative value!");
      }
      return static_cast<size_t>(opid.get());
    }
  };

  // When dealing with medium size of or even more large models,
  // the number of tensors and ops grow fast. By using std::unordered_map
  // we can reduce of insertation, query complexities from O(logN)
  // (N is the number of tensors of ops to deal with) to O(1). The
  // problem of std::unordered_map is the capcity. The initial capacity of
  // the container is 8, and once it is at capacity, the container will
  // request almost doubled memory from kernel (very slow), and copy the old
  // data into the new memory (slow again). Frequent request memory in a loop
  // is time consuming. By predicting the memory to be used, we can achieve
  // exact O(1) complexity for faster compilation.
  template <size_t GROWTH_FACTOR, typename Key, typename Val, class Hasher>
  void
  preAllocAndRehash(typename std::unordered_map<Key, Val, Hasher> &hashTable,
                    size_t max_slots) {
    if (hashTable.bucket_count() < max_slots) {
      hashTable.reserve((hashTable.size() + max_slots) * GROWTH_FACTOR);
    }
    {
      float load_factor = hashTable.size() / hashTable.bucket_count();
      if (load_factor > AliasModel::loadFactor) {
        hashTable.rehash((hashTable.size() + max_slots) * GROWTH_FACTOR);
      }
    }
  }

  // Warning: do not iterate over these members as as iterating over unordered
  // containers introduces non-determinism.
  std::unordered_map<TensorId, PoprithmsTensorId> toTensor_;
  std::unordered_map<PoprithmsTensorId, TensorId, PoprithmsTensorIdSteadyHasher>
      fromTensor_;
  std::unordered_map<OpId, std::vector<PoprithmsOpId>> toOp_;
  std::unordered_map<PoprithmsOpId, OpId, PoprithmsOpIdHasher> fromOp_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ALIAS_ALIASMODEL_HPP_
