// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ISTEPIO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ISTEPIO_HPP_

#include <cstdint>
#include <popart/names.hpp>
#include <popart/voiddata.hpp>

namespace popart {

namespace popx {
class Executablex;
}

/**
 * An abstract base class through which input and output data is passed to a
 * Session (see Session::run). Data is passed via buffers. In the case of
 * buffers returned by IStepIO::in, PopART *reads* from these buffers. In
 * the case of IStepIO::out, PopART *writes* to these buffers. The
 * IStepIO::inComplete() and IStepIO::outComplete() functions are called by
 * PopART to signal it is done with an input or output buffer.
 *
 * An IStepIO implementation should conceptually implement a rolling queue of
 * active buffers for each input and output tensor. Every successful call to
 * IStepIO::in should yield a new data buffer for PopART to read from and add
 * it to the head of the conceptual queue. Conversely, every call to
 * IStepIO::inComplete() should be taken to mean that the buffer at the tail-end
 * of the queue is no longer being used by PopART. This buffer is removed from
 * the conceptual queue.
 *
 * Note that a IStepIO::in call with the \c prefetch flag set is only
 * considered successful when it returns data.
 *
 * Output works analogously to input.
 *
 * The expected total number of input (or output) buffers that are
 * 'completed' for a tensor in one Session::run call is `bps` \f$\times\f$
 * SessionOptions::accumulationFactor \f$\times\f$
 * SessionOptions::replicatedGraphCount, where `bps` is the number of
 * batches per call to Session::run (this is a value captured by the DataFlow
 * instance passed to the Session instance).
 *
 * Note, however, that there may be additional 'incomplete' calls to
 * IStepIO::in and IStepIO::out.
 *
 * Furthermore, the number of input (or output) buffers that may
 * be 'incomplete' *at a given time for a given tensor* should not normally be
 * more than SessionOptions::bufferingDepth \f$\times\f$
 * SessionOptions::replicatedGraphCount, but this bound is not guaranteed.
 *
 * **EXAMPLE**: Suppose a session is configured such that the total expected
 * number of input buffers is 6 and these are input buffers for a tensor with
 * ID `t` with 100 elements. The associated input calls in IStepIO may look
 * like this if SessionOptions::bufferingDepth is 3:
 *
 * ```
 * in("t", 100, false) -> Give buffer[0] to PopART.
 * in("t", 100, true) -> Give buffer[1] to PopART.
 * in("t", 100, true) -> Give buffer[2] to PopART.
 * inComplete("t", 100) -> buffer[0] is no longer required and can be reused.
 * in("t", 100, true) -> Give buffer[3] to PopART.
 * inComplete("t", 100) -> buffer[1] is no longer required and can be reused.
 * in("t", 100, true) -> Give buffer[4] to PopART.
 * inComplete("t", 100) -> buffer[2] is no longer required and can be reused.
 * in("t", 100, true) -> Give buffer[5] to PopART.
 * inComplete("t", 100) -> buffer[3] is no longer required and can be reused.
 * in("t", 100, true) -> No data available, return nullptr.
 * inComplete("t", 100) -> buffer[4] is no longer required and can be reused.
 * inComplete("t", 100) -> buffer[5] is no longer required and can be reused.
 * ```
 */
class IStepIO {
public:
  /// Destructor for IStepIO.
  virtual ~IStepIO() = default;

  /**
   * Request a new input data buffer.
   *
   * The memory in this buffer is
   * available for use in PopART until the corresponding inComplete() call.
   *
   * \note: Failing to provide a valid data buffer will result in a runtime
   * failure if \c prefetch is set to `false`.
   *
   * \param id The ID of the tensor to return data for.
   * \param numElements The number of elements in the tensor.
   * \param prefetch If set to `true` the inability to provide data is not
   *     considered an error. If `false`, it is considered an error if no data
   *     can be provided.
   * \return The input buffer for this tensor (or nullptr on failure) returned
   *     as a ConstVoidData object.
   */
  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch) = 0;

  /**
   * Notify the user (running a PopART program) that a previously retrieved
   * input data buffer is no longer used by PopART.
   *
   * \param id The ID of the tensor to return data for.
   * \param numElements The number of elements in the tensor.
   */
  virtual void inComplete(TensorId id, int64_t numElements) = 0;

  /**
   * Request a new output data buffer.
   *
   * The memory in this buffer is available for use in PopART until the
   * corresponding inComplete() call and will be modified in-place.
   *
   * \note Failing to provide a valid data buffer will result in a runtime
   * failure.
   *
   * \param id The ID of the tensor to return data for.
   * \param numElements The number of elements in the tensor.
   * \return The output buffer for this tensor returned as a MutableVoidData
   *     object.
   */
  virtual MutableVoidData out(TensorId id, int64_t numElements) = 0;

  /**
   * Notify the user (running a PopART program) that a previously retrieved
   * input data buffer is no longer used by PopART.
   *
   * \param id The ID of the tensor to return data for.
   * \param numElements The number of elements in the tensor.
   */
  virtual void outComplete(TensorId) {}

  /**
   * Enable or disable runtime asserts.
   *
   * If runtime asserts are enabled, then a check that the input and output
   * buffers have the correct number of elements is performed. As Session.run()
   * is called multiple times during a user's session, the check is only
   * performed in the first call to Session.run(), under the assumption that the
   * user is unlikely to change the size of buffers between runs.
   *
   * \param b The setting to enable runtime asserts (`true`) or
   *     disable runtime asserts (`false`).
   */
  void enableRuntimeAsserts(bool b) { runtimeAssertsOn = b; }

  /**
   * Check if runtime asserts are enabled.
   *
   * \returns `true` if runtime asserts are enabled, otherwise `false`.
   */
  bool runtimeAssertsEnabled() const { return runtimeAssertsOn; }

  /**
   * Check number of elements.
   *
   * This check is performed when runtimeAssertsEnabled() is `true`.
   *
   * \param Executablex The input executable to be checked that the input and
   *    output buffers have the correct number of elements.
   */
  virtual void assertNumElements(const popx::Executablex &) const = 0;

private:
  bool runtimeAssertsOn{true};
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ISTEPIO_HPP_
