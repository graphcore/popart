// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_EXCHANGEBUNDLE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_EXCHANGEBUNDLE_HPP_

#include <snap/RemoteBuffer.hpp>
#include <snap/Tensor.hpp>
#include <popart/ir.hpp>

namespace popart {
namespace popx {

/**
 * Helper class to bundle all host and remote exchange lowering
 * information together
 */
class ExchangeBundle {
public:
  /**
   * Construct empty exchange sharding bundle
   * Creates the replicatedTensorShardingTracer with the IR object
   * \param ir IR to create the \c ReplicatedTensorShardingTracer with
   */
  ExchangeBundle(const Ir &ir);

  /**
   * Check if the remote buffer exists
   * \param  id ID of the remote buffer
   * \return    True if the remote buffer and landing pads exist
   */
  bool hasRemoteBuffer(RemoteBufferId id) const;

  /**
   * Return the remote buffer and landing pad tensors
   * \param id ID of the remote buffer
   * \return   Pair of remote buffer and landing pad tensors
   */
  const std::pair<snap::RemoteBuffer, std::vector<snap::Tensor>> &
  getRemoteBuffer(RemoteBufferId id) const;

  /**
   * Given the ID of the remote buffer, return the name of the remote buffer
   * \param id ID of the remote buffer
   * \return   Name of the remote buffer
   */
  static const std::string getRemoteBufferName(RemoteBufferId id);

  bool getRemoteBufferSeparateLoadStorePadsRequired(RemoteBufferId id) const {
    return remoteBufferSeparateLoadStorePadsRequired.at(id);
  }

  /**
   * Create a remote buffer and associated landing pad tensors
   * \param graph   Graph to create the remote buffer on
   * \param id      ID of the remote buffer
   * \param tensors Landing pad tensors to register
   */
  void createRemoteBuffer(snap::Graph &graph,
                          RemoteBufferId id,
                          const std::vector<snap::Tensor> &tensors);

  /**
   * Check if a stream landing pad tensor exists for \p TensorId id
   * \param id \p TensorId to probe
   * \return   true if the tensor exists, false if not.
   */
  bool hasStreamTensor(TensorId tid) const;

  /**
   * Get the stream landing pad tensor for \p TensorId id
   * \param tid \p TensorId to fetch
   * \return    Returns the \p snap::Tensor
   */
  snap::Tensor getStreamTensor(TensorId tid) const;

  /**
   * Set a new stream landing pad tensor for \p TensorId id
   * \param tid \p TensorId to set
   * \param t   \p snap::Tensor to set
   */
  void setStreamTensor(TensorId tid, snap::Tensor t);

private:
  /**
   * Reference to the IR
   */
  const Ir &ir;

  /**
   * Map to store which remote buffers require separate load and store pads
   * in order to be able to merge exchanges optimally
   */
  std::map<RemoteBufferId, bool> remoteBufferSeparateLoadStorePadsRequired;

  /**
   * Remote buffers
   */
  std::map<RemoteBufferId,
           std::pair<snap::RemoteBuffer, std::vector<snap::Tensor>>>
      remoteBuffers;

  /**
   * Stream tensors are temporary landing pad tensors for host exchanges
   * that are implementation specific, and used to facilitate overlapped
   * IO by avoiding internal exchanges following a host exchange.
   * They are not exposed to the IR.
   */
  std::map<TensorId, snap::Tensor> streamTensors;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_EXCHANGEBUNDLE_HPP_
