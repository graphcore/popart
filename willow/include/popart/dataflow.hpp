// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DATAFLOW_HPP
#define GUARD_NEURALNET_DATAFLOW_HPP

#include <set>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// Forward declaration.
struct SessionOptions;

/**
 * An anchor tensor is a tensor that the user wants returned after a
 * call to Session::run. Each call to Session::run results in
 * `batchesPerStep x accumulationFactor x replicationFactor` of such
 * tensors being computed. We refer to the samples associated
 * with each such computation as a micro batch. The dimensions are
 * user-specified by the following parameters:
 *
 *  * `batchesPerStep` is the value in DataFlow.
 *  * `accumulationFactor` is the value defined by
 *    SessionOptions::accumulationFactor.
 *  * `replicationFactor` is the value defined by
 *    SessionOptions::globalReplicationFactor.
 *
 * This enum type describes the strategy with which the micro batch values
 * for anchor tensors (or summaries thereof) are written or to the IStepIO
 * instance passed to Session::run.
 *
 * See also:  AnchorReturnType.
 *
 * **NOTE**: Anchors are essentially what tensorflow calls "fetches".
 */
enum class AnchorReturnTypeId {
  /// Only return the tensor value for the last micro batch of the Session::run
  /// call for each replica.
  ///
  /// The buffer shape required for this anchor in IStepIO is
  /// `[replicationFactor, <anchorTensorShape>]`
  /// (with dimensions of size 1 removed).
  Final = 0,
  /// Return the tensor value for *all* micro batches for each replica
  ///
  /// The buffer shape required for this anchor in IStepIO is
  /// `[batchesPerStep, accumulationFactor, replicationFactor,
  /// <anchorTensorShape>]`
  /// (with dimensions of size 1 removed).

  /// Return the tensor value for every `N`th global batch for each replica and
  /// for all accumulation steps in that global batch. Note that the value of
  /// `N` is captured by AnchorReturnType.
  ///
  /// The buffer shape required for this anchor in IStepIO is
  /// `[batchesPerStep // N, accumulationFactor, replicationFactor,
  /// <anchorTensorShape>]`
  /// (with dimensions of size 1 removed).
  EveryN,
  /// Return the tensor value for *all* micro batches for each replica.
  ///
  /// The buffer shape required for this anchor in IStepIO is
  /// `[batchesPerStep, accumulationFactor, replicationFactor,
  /// <anchorTensorShape>]`
  /// (with dimensions of size 1 removed).
  All,
  /// Return one tensor value for each replica, doing a sum reduction over the
  /// `batchesPerStep` and `accumulationFactor` dimensions.
  ///
  /// The buffer shape required for this anchor in IStepIO is
  /// `[replicationFactor, <anchorTensorShape>]`
  /// (with dimensions of size 1 removed).
  Sum,
};

std::ostream &operator<<(std::ostream &, AnchorReturnTypeId);

/**
 * A class that captures an AnchorReturnTypeId value and, when this value is
 * AnchorReturnTypeId::EVERYN, the associated `N` number. The constructor
 * takes `std::string` values and converts them as appropriate.
 *
 * See also: #AnchorReturnTypeId.
 */
class AnchorReturnType {
public:
  /// Constructor.
  /// \param artString - the string to convert to an #AnchorReturnTypeId value.
  /// The following values are acceptable (case insensitive):
  ///  * `"final"` - AnchorReturnTypeId::FINAL
  ///  * `"all"` - AnchorReturnTypeId::ALL
  ///  * `"sum"` - AnchorReturnTypeId::SUM
  ///
  /// **NOTE**: Constructing an AnchorReturnType with of type
  /// AnchorReturnTypeId::EVERYN
  /// using this constructor will result in an error. Use the constructor that
  /// also specifies a return period.
  AnchorReturnType(std::string artString);
  /// Constructor.
  /// \param artString the string to convert to an #AnchorReturnTypeId value.
  /// The following values are acceptable (case insensitive):
  ///  * `"final"` - AnchorReturnTypeId::FINAL
  ///  * `"everyn"` - AnchorReturnTypeId::EVERYN
  ///  * `"all"` - AnchorReturnTypeId::ALL
  ///  * `"sum"` - AnchorReturnTypeId::SUM
  /// \param returnPeriod the value of `N` in case of
  /// AnchorReturnTypeId::EVERYN.
  ///
  /// **NOTE**: Constructing a #AnchorReturnType with of type
  /// AnchorReturnTypeId::EVERYN will result in an error. Use the constructor
  /// that also specifies the return period.
  AnchorReturnType(std::string artString, int returnPeriod);

  // Return the associated #AnchorReturnTypeId, not currently part of public
  // API.
  AnchorReturnTypeId id() const { return artId_; }
  // Return the associated return period (`N`) if the #AnchorReturnTypeId is
  // AnchorReturnTypeId::EVERYN,
  // not currently part of public API.
  int rp() const;

  // A hash value, not currently part of public API.
  std::size_t hash() const;

private:
  AnchorReturnTypeId getIdFromStr(std::string artString);

  std::string artStr_;
  AnchorReturnTypeId artId_;

  int returnPeriod_;
};

/**
 * This class specifies parameters for host-device data streams. The parameters
 * are used to control the amount input data processed each step (that is: each
 * Session::run call) determines how data is returned to the user.
 *
 * See also: AnchorReturnType, #AnchorReturnTypeId.
 **/
class DataFlow {
public:
  /// Default constructor, sets `batchesPerStep` to 0 and does not have any
  /// anchors.
  DataFlow();
  /// Construct DataFlow instance without anchor tensors.
  /// \param batchesPerStep - the number of global batches to run the inference
  ///     or training session for per call to Session::run before returning
  ///     control to the caller.
  DataFlow(int batchesPerStep);
  /// Constructor DataFlow instance with anchor tensors.
  /// \param batchesPerStep the number of global batches to run the inference or
  ///     training session for per call to Session::run before returning control
  ///     to the caller.
  /// \param anchorMap a mapping from output tensor TensorId to AnchorReturnType
  ///     indicating the strategy with which to write the anchor tensor values
  ///     to the IStepIO object provided to Session::run.
  DataFlow(int batchesPerStep,
           const std::map<TensorId, AnchorReturnType> &anchorMap);
  /// Constructor DataFlow instance with anchor tensors.
  /// \param batchesPerStep the number of global batches to run the inference or
  ///     training session for per call to Session::run before returning control
  ///     to the caller.
  /// \param anchorTensorIds the tensor ID of anchor tensors.
  /// \param anchorReturnType the strategy with which to write anchor tensor
  ///     values to the IStepIO object provided to Session::run.
  DataFlow(int batchesPerStep,
           const std::vector<TensorId> anchorTensorIds,
           const AnchorReturnType &anchorReturnType = AnchorReturnType("All"));

  // Default copy constructor, not currently part of public API.
  DataFlow(const DataFlow &rhs) = default;
  // Default assignment operator, not currently part of public API.
  DataFlow &operator=(const DataFlow &rhs) = default;

  // Determine if a tensor is an anchor, not currently part of public API.
  bool isAnchored(TensorId) const;
  // Determine if batch counting is required, not currently part of public API.
  bool isBatchCountingRequired() const;
  // Return a vector of all anchor tensors, not currently part of public API.
  const std::vector<TensorId> &anchors() const { return v_anchors; }
  // Return a vector of all return periods, not currently part of public API.
  const std::vector<int> &rps() const { return v_rps; }
  // Number of anchor tensors, not currently part of public API.
  int nAnchors() const { return static_cast<int>(v_anchors.size()); }
  // Number of global batches per Session::run call, not currently part of
  // public API.
  int batchesPerStep() const { return batchesPerStep_; }
  // Get AnchorReturnType for a anchor tensor, not currently part of public API.
  AnchorReturnType art(TensorId anchorId) const;
  // Get number of fetches per replica for a tensor, not currently part of
  // public API.
  unsigned numOutFetchesPerRepl(const struct SessionOptions &opts,
                                const TensorId &anchorId) const;
  // Get a hash for this object, not currently part of public API.
  std::size_t hash() const;

private:
  // The number of batches processed by the backend in one call to train
  // or infer.
  int batchesPerStep_;

  // The set of tensors to return to the user after execution, and how
  // frequently they are returned during multi-batch training or inference
  std::map<TensorId, AnchorReturnType> m_anchors;

  // The set of anchor tensors (as std::vector).
  std::vector<TensorId> v_anchors;
  // The set of anchor tensors (as std::vector).
  std::set<TensorId> s_anchors;

  // The unique set of return periods for all anchors.
  // Depending on the anchor return type, extra tensors are added
  // to the graph during its construction to keep track of batch
  // count. This member ensures the minimum number of tensors are
  // added.
  std::vector<int> v_rps;

  void isValidAnchorReturnPeriod(TensorId anchorId, int batchesPerStep);
};

} // namespace popart

namespace std {
template <> struct hash<popart::DataFlow> {
  std::size_t operator()(const popart::DataFlow &df) const { return df.hash(); }
};

template <> struct hash<popart::AnchorReturnType> {
  std::size_t operator()(const popart::AnchorReturnType &art) const {
    return art.hash();
  }
};
} // namespace std

#endif
