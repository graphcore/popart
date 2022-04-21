// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALIASMODEL_HPP
#define GUARD_NEURALNET_ALIASMODEL_HPP

#include <map>

#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/proposal.hpp>
#include <poprithms/memory/inplace/result.hpp>
#include <poprithms/memory/inplace/tensor.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>

#include <popart/operators.hpp>

namespace popart {

// Forward declaration.
class Op;

/**
 * A container for the poprithms::memory::inplace::Graph which corresponds to a
 * PopART Graph. It contains the poprithms Graph, and mappings between PopART
 * Tensors and Ops, and their poprithms equivalents.
 * */
class AliasModel {
public:
  using PoprithmsTensorId = poprithms::memory::inplace::TensorId;
  using PoprithmsOpId     = poprithms::memory::inplace::OpId;

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

private:
  std::map<TensorId, PoprithmsTensorId> toTensor_;
  std::map<PoprithmsTensorId, TensorId> fromTensor_;
  std::map<OpId, std::vector<poprithms::memory::inplace::OpId>> toOp_;
  std::map<poprithms::memory::inplace::OpId, OpId> fromOp_;
};

} // namespace popart

#endif
