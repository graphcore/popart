// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OPSHARDING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OPSHARDING_HPP_

#include <cstdint>
#include <vector>
#include <popart/names.hpp>
#include <popart/op/init.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Graph;
class Tensor;
class TensorInfo;

/**
 * Class that contains helper functions to add sharding related operations to
 * the graph.
 *
 * Static vs. dynamic methods:
 *
 * - Static methods can't be outlined since the slicing/concatenation index
 *   is fixed at compile time
 * - Static methods can be eliminated during compile time / IR lowering
 * - Static methods can be inplaced fully
 * - Dynamic methods can be outlined, the slicing/concatenation index is a
 *   tensor parameter
 * - Dynamic methods cannot be eliminated during compile time / IR lowering
 * - Dynamic methods can only be inplaced on the full tensor, while the slice
 *   tensors are outplace
 */
class ShardingHelper {
public:
  ShardingHelper(Graph *graph_);

  /// Concatenate tensors with a static ConcatOp (not outlineable).
  /// \param axis Tensor axis along which to concatenate.
  /// \param tensorIds Tensors to concatenate (must exist in the IR).
  /// \param concatId Output tensor name (must not exist in the IR).
  /// \param settings Op::Settings to apply to the ConcatOp.
  ///
  /// <pre>
  /// tensorIds(0) tensorIds(1) ... tensorIds(n)
  ///            \  \            /  /
  ///                  ConcatOp
  ///                     |
  ///                  concatId
  /// </pre>
  std::vector<Op *> staticConcat(int64_t axis,
                                 std::vector<TensorId> tensorIds,
                                 TensorId concatId,
                                 Op::Settings settings) const;

  /// Shard tensors with a set of SliceOps (not outlineable).
  /// \param axis Tensor axis along which to slice.
  /// \param tensorIds Slice tensors to generate (must not exist in the IR).
  /// \param concatId Input tensor name (must exist in the IR).
  /// \param settings Op::Settings to apply to the ConcatOp
  ///                 (1 setting in total or 1 setting per slice tensor).
  ///
  /// <pre>
  ///           concatId
  ///        /            \
  ///  SliceOp(0) ... SliceOp(n)
  ///       |              |
  /// tensorIds(0)   tensorIds(n)
  /// </pre>
  std::vector<Op *> staticShard(int64_t axis,
                                std::vector<TensorId> tensorIds,
                                TensorId concatId,
                                std::vector<Op::Settings> settings) const;

  /// Create an AddOps to create an offset version of the input tensor.
  /// \param tensorId Input tensor name (must exist in the IR).
  /// \param offsetId Offset tensor name (must exist in the IR).
  /// \param settings Op::Settings to apply to the AddOp.
  /// <pre>
  /// concatId  offsetId
  ///       |  /
  ///     AddOp
  ///       |
  /// tensorId
  /// </pre>
  std::vector<Op *> offset(TensorId tensorId,
                           TensorId concatId,
                           TensorId offsetId,
                           Op::Settings settings) const;

  /// Shard tensors with a set of AddOps to create offset versions of the input
  /// tensor.
  /// \param tensorIds Slice tensors to generate (must not exist in the IR).
  /// \param concatId Input tensor name (must exist in the IR).
  /// \param settings Op::Settings to apply to the AddOp
  ///                 (1 setting in total or 1 setting per slice tensor).
  /// <pre>
  ///
  /// Const(0)  concatId   Const(n)
  ///       |  /        \  |
  ///     AddOp   ...    AddOp
  ///       |              |
  /// tensorIds(0)   tensorIds(n)
  /// </pre>
  std::vector<Op *> offsets(std::vector<TensorId> tensorIds,
                            TensorId concatId,
                            std::vector<Op::Settings> settings) const;

  /// Update tensor with a dynamic DynamicUpdateOp (outlineable).
  /// \param axis Tensor axis along which to concatenate,
  /// \param num_shards Number of addressable indices along the axis.
  /// \param sliceId Slice tensor to update into the concatOutId tensor
  ///        (must exist in IR).
  /// \param concatInId Tensor to be updated (must exist in the IR).
  /// \param concatOutId Updated tensor (must not exist in the IR).
  /// \param indexId Index tensor to indicate where to apply the update
  ///        (must exist in the IR).
  /// \param settings Op::Settings to apply to the DynamicUpdateOp.
  ///
  /// <pre>
  /// sliceId concatInId indexId
  ///    \      |        /
  ///     DynamicUpdateOp
  ///           |
  ///         concatOutId
  /// </pre>
  std::vector<Op *> dynamicUpdate(int64_t axis,
                                  int64_t num_shards,
                                  TensorId sliceId,
                                  TensorId concatInId,
                                  TensorId concatOutId,
                                  TensorId indexId,
                                  Op::Settings settings) const;

  /// Concatenate tensors with a set of DynamicUpdateOps (outlineable).
  /// \param axis Tensor axis along which to concatenate.
  /// \param tensorIds Tensors to concatenate (must exist in the IR).
  /// \param concatId Output tensor name (can exist in the IR).
  /// \param settings Op::Settings to apply to the DynamicUpdateOps
  ///                 (1 setting in total or 1 setting per slice tensor, plus
  ///                  2 additional settings for pre/post processing Ops).
  ///
  /// <pre>
  ///                   InitOp          < settings.at(n)
  ///                     |
  /// tensorIds(0) - DynamicUpdateOp(0) < settings.at(0)
  ///                     |
  /// tensorIds(1) - DynamicUodateOp(1) < settings.at(1)
  ///    ...             ...
  /// tensorIds(n) - DynamicUpdateOp(n) < settings.at(n-1)
  ///                     |
  ///                  ReshapeOp        < settings.at(n+1)
  ///                     |
  ///                  concatId
  /// </pre>
  std::vector<Op *> dynamicConcat(int64_t axis,
                                  std::vector<TensorId> tensorIds,
                                  TensorId concatId,
                                  std::vector<Op::Settings> settings) const;

  /// Slice tensor with a dynamic DynamicSliceOp (outlineable).
  /// \param axis Tensor axis along which to concatenate.
  /// \param num_shards Number of addressable indices along the axis.
  /// \param sliceId Slice tensor to cut from the concatId tensor
  ///        (must not exist in IR).
  /// \param concatId Tensor to be sliced (must exist in the IR).
  /// \param indexId Index tensor to indicate where to slice the tensor
  ///        (must exist in the IR).
  /// \param settings Op::Settings to apply to the DynamicSliceOp.
  ///
  /// <pre>
  ///   concatId  indexId
  ///        |     |
  ///     DynamicSliceOp
  ///           |
  ///        sliceId
  /// </pre>
  std::vector<Op *> dynamicSlice(int64_t axis,
                                 int64_t num_shards,
                                 TensorId sliceId,
                                 TensorId concatId,
                                 TensorId indexId,
                                 Op::Settings settings) const;

  /// Shard tensors with a set of DynamicSliceOps (outlineable).
  /// \param axis Tensor axis along which to slice.
  /// \param tensorIds Slice tensors to generate (must not exist in the IR).
  /// \param concatId Input tensor name (must exist in the IR).
  /// \param settings Op::Settings to apply to the ConcatOp
  ///                 (1 setting in total or 1 setting per slice tensor, plus
  ///                  2 additional settings for pre/post processing Ops).
  ///
  /// <pre>
  ///               concatId
  ///                  |
  ///               ReshapeOp                    < settings.at(n)
  ///        /                   \
  ///  DynamicSliceOp(0) ... DynamicSliceOp(n)   < settings.at([0, n-1])
  ///       |                     |
  /// tensorIds(0)          tensorIds(n)
  /// </pre>
  std::vector<Op *> dynamicShard(int64_t axis,
                                 std::vector<TensorId> tensorIds,
                                 TensorId concatId,
                                 std::vector<Op::Settings> settings) const;

  /// Create an InitOp that produces a tensor.
  /// \param info Data type and shape of the output tensor.
  /// \param id Tensor name from which to derive the output tensor ID.
  /// \param type Initialization type for the created tensor
  ///             (uninitialized or zero).
  /// \returns Tensor with TensorId based on the graph-scope, id and type.
  Tensor *initTensor(TensorInfo info,
                     TensorId id,
                     InitType type,
                     Op::Settings settings) const;

  /// Create an IdLossOp with fromLoss/toLoss set to true (final loss).
  /// \param reductionType
  /// \param intermediateId
  /// \param lossOutId
  /// \param settings
  Op *idLoss(ReductionType reductionType,
             TensorId intermediateId,
             TensorId lossOutId,
             Op::Settings settings) const;

  /// Create a ScaleOp with factor.
  /// \param factor scaling factor.
  /// \param inId input to connect (must exist in the IR).
  /// \param outId output to connect (must exist in the IR).
  /// \param settings Op::Settings to apply to the ScaleOp.
  Op *scale(float factor,
            TensorId inId,
            TensorId outId,
            Op::Settings settings) const;

  /// Connect tensor as output to an Op.
  /// \param op Op to connect the output tensor to.
  /// \param id output to connect (can be a new or existing tensor ID).
  /// \param index OutIndex at which to connect the output to the Op.
  void connectOutTensor(Op *op, TensorId id, OutIndex index) const;

  /// Create a ReshapeOp.
  /// \param inId Tensor to reshape.
  /// \param newShape Shape to reshape the in tensor into.
  /// \param outId Tensor to create (can be a new or existing tensor ID).
  /// \param settings Op::Settings to apply to the ConcatOp.
  /// \returns The ReshapeOp created.
  std::vector<Op *> reshapeForSlice(TensorId inId,
                                    Shape newShape,
                                    TensorId outId,
                                    Op::Settings settings) const;

  /// Create a constant index tensor.
  /// \param index Unsigned integer index.
  /// \param settings Settings to derive virtual graph and tile set from.
  /// \returns The ID of the constant index tensor created.
  TensorId createOrGetIndexTensor(uint32_t index, Op::Settings settings) const;

  /// Create a constant value tensor.
  /// \param type DataType of the tensor to generate.
  /// \param value Value of the tensor to be generated, with a type T compatible
  ///              to the DataType type.
  /// \param settings Settings to derive virtual graph and tile set from.
  /// \returns The ID of the constant tensor created.
  template <class T>
  TensorId
  createOrGetConstTensor(DataType type, T value, Op::Settings settings) const;

private:
  /// The graph on which to operate.
  Graph *graph;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OPSHARDING_HPP_
