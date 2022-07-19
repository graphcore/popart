// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_CODECOPY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_CODECOPY_HPP_

#include <memory>
#include <popart/op.hpp>
#include <popart/tensorlocation.hpp>

#include "popart/graphid.hpp"
#include "popart/op/exchange/exchange.hpp"

namespace popart {
struct OperatorIdentifier;

/**
 * An op to load code from streaming memory on to the device. A graphId is
 * specified to load the code for and a poplar::FunctionBuffer is created to
 * copy code from to the poplar::Function. It can copy from streaming memory to
 * a buffer or executable memory. \sa InternalCodeCopyOp for copying from/to
 * memory on the device.
 *
 * ExternalCopyOps encapsulate those "copies of code" that require an external
 * exchange (e.g. involving streaming memory).
 * NOTE: Currently only copying from streaming memory to executable on chip
 * memory is supported for RemoteCodeLoadOps.
 * NOTE: RemoteCodeLoadOps are derived from ExchangeBaseOps which are
 * suitable for external exchanges, and can be merged using the mergeexchange
 * transform.
 */
class RemoteCodeLoadOp : public ExchangeBaseOp {
public:
  /**
   * Construct a new External Code Copy Op.
   *
   * \param gid Use the GraphId, rather than Graph object to track the graph we
   * are loading code for, in case the graph doesn't exist yet.
   * \param destinationType The destination memory type to copy to.
   *  One of:
   *  Buffer - Stored in non-executable buffer memory.
   *  ExecutableMemory - Stored in executable memory.
   * Note: The destination TileSet (Compute / IO) is determined by the op's
   * TileSet attribute.
   */
  RemoteCodeLoadOp(const OperatorIdentifier &,
                   const GraphId &gid,
                   const CodeMemoryType destinationType,
                   const Op::Settings &);

  /**
   * Get the Exchange Descriptor object. \sa ExchangeDescriptor
   *
   * \param index
   * \returns ExchangeDescriptor
   */
  ExchangeDescriptor getExchangeDescriptor(int index) const final;

  std::unique_ptr<Op> clone() const final;

  void setup() final;
  bool hasSideEffect() const override { return true; }
  void appendOutlineAttributes(OpSerialiserBase &) const final;

private:
  const GraphId graphId;
  const CodeMemoryType destinationType;
};

class InternalCodeCopyOp : public Op {
public:
  /**
   * Construct a new Internal Code Copy Op. This copies code to or from Buffers
   * and Executable memory on the device. This is in contrast to the
   * RemoteCodeLoadOp that copies from streaming memory on to the device.
   * InternalCopyOps encapsulate copies that only require an internal exchange.
   *
   * NOTE: InternalCopyOps do not involve external exchanges and are thus not
   * derived from ExchangeBaseOp.
   *
   * \param gid Use the GraphId, rather than Graph object to track the graph we
   * are loading code for, in case the graph doesn't exist yet.
   * \param source The source tileset to copy from.
   * \param sourceType The source memory type to copy from.
   * \param destinationType The destination memory type to copy to.
   * Note: Destination tileset is taken from the op's settings.tileSet property.
   */
  InternalCodeCopyOp(const OperatorIdentifier &,
                     const GraphId &gid,
                     const TileSet source,
                     const CodeMemoryType sourceType,
                     const CodeMemoryType destinationType,
                     const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  void setup() final {
    throw error("InternalCodeCopyOp is not yet implemented.");
  }
  bool hasSideEffect() const override { return true; }

private:
  const GraphId graphId;
  const TileSet source;
  const CodeMemoryType sourceType;
  const CodeMemoryType destinationType;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_EXCHANGE_CODECOPY_HPP_
