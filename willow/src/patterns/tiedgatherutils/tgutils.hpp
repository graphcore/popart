// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TEMP_TIEDGATHER_UTILS_HPP
#define GUARD_NEURALNET_TEMP_TIEDGATHER_UTILS_HPP

#include <queue>
#include <vector>

#include <popart/op.hpp>

namespace popart {
class Tensor;
} // namespace popart

namespace popart {
namespace tgutil {

/**
 * \brief Returns whether the Tensor #t is prodcued by a transpose oeration.
 *
 * \param t Tensor we are querying.
 * \return true if #t is produced by a transpose operation, false otherwise.
 */
bool isProducedByTranspose(const Tensor *t);

/**
 * \brief Finds the root variable that #t is a view of.
 * \details Starting at #t, performs a pre-order depth-first search backwards
 *          through the graph (up through the producers), returning the first
 *          Variable or Const Tensor. Only view-changing operations are
 *          traversed.
 *
 * \param t The Tensor from which to start the search.
 * \return Tensor* The root variable #t is a view of, or nullptr if not found.
 */
Tensor *getVariable(Tensor *t);

/**
 * \brief Finds the Op of type #T that consumes the (view of a) weight #w.
 * \details First, starting at #w, performs a pre-order depth-first search
 *          backwards through the graph (up through the producers), returning
 *          the first Variable or Const Tensor. Only view-changing operations
 *          are traversed.
 *          Second, if a Tensor was found, performs a pre-order breadth-first
 *          search forwards through the graph (through consumers), returning the
 *          first Op of type #T with #ExecutionContext::Normal found. Only
 *          traverses through view-changing operations and and certain other Ops
 *          See #searchConsumersFor for exact details.
 *
 * \tparam T The Op subclass to search for in the consumers of #w.
 * \param w The Tensor to start the search for the consumer from.
 * \return T* The Op of type #T that consumes the weight #w, or nullptr if not
 *         found.
 */
template <typename T> T *weightConsumedBy(Tensor *w);

/**
 * \brief Starting at the Tensor #t, searches for the Op of type #T with
 *        ExecutiionContext #Ctx that consumes it.
 * \details Performs a pre-order breadth-first search forwards through the
 *          graph (through consumers), returning the first Op of type #T with
 *          ExecutionContext #Ctx found. Only traverses through view-changing
 *          operations and certain other Ops. See implementation for exact
 *          details.
 *
 * \tparam T The Op subclass to search for in the consumers of #t.
 * \tparam Ctx The ExecutionContext that the consumer must have.
 * \param t The Tensor to start the search for the consumer from.
 * \return T* The Op of type #T with ExecutionContext #Ctx that consumes the
 *            Tensor #t, or nullptr if not found.
 */
template <class T, ExecutionContext Ctx = ExecutionContext::Normal>
T *searchConsumersFor(Tensor *t);

/**
 * \brief Starting at Tensor #t, searches through the producers for an Op of
 *        type T with ExecutionContext #Ctx.
 * \details Performs a pre-order depth-first search backwards through the
 *          producers of Tensor #t, returning the first Op of type #T with
 *          ExecutionContext #Ctx found. Only traverses through view-changing
 *          operations and certain other Ops. See implementation for more
 *          details.
 *
 * \tparam T The Op subclass to search for in the producers of #t.
 * \tparam Ctx The ExuectionContext that the producer must have.
 * \param t The Tensor to start the search from.
 * \return T* The Op of type #T with ExecutionContext #Ctx in the producers of
 *            Tensor #t, or nullptr if not found.
 */
template <class T, ExecutionContext Ctx = popart::ExecutionContext::Normal>
T *searchProducersFor(Tensor *t);

/**
 * \brief Starting at Tensor #t, finds all consumers of type #T with
 *        ExecutionContext #Ctx.
 * \details Performs a pre-order breadth-first search through forwards through
 *          the graph (through consumers), returning all Ops of type #T with
 *          ExecutionContext #Ctx found. Only traverses through view-changing
 *          operations and certain other Ops. See implementation for exact
 *          details. Note the Ops traversable may not be the exact same as in
 *          #searchConsumersFor.
 *
 * \tparam T The Op subclass to search for in the consumers of #t.
 * \tparam Ctx The ExecutionContext that the consumers have.
 * \param t The Tensor to start the search from.
 * \return std::vector<T *> All found consumers of #t, possibly empty.
 */
template <class T, ExecutionContext Ctx = ExecutionContext::Normal>
std::vector<T *> findAllConsumers(Tensor *t);

} // namespace tgutil
} // namespace popart

#include "tgutils_impl.hpp"

#endif
