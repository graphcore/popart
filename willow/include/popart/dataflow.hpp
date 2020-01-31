#ifndef GUARD_NEURALNET_DATAFLOW_HPP
#define GUARD_NEURALNET_DATAFLOW_HPP

#include <set>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// An anchor tensor is a tensor which the user wants returned
// after a step is run. Anchors are essentially what tensorflow calls
// "fetches". AnchorReturnType specifies what exactly should be
// returned for a tensor, currently the 3 options are:

enum class AnchorReturnTypeId {
  FINAL = 0, // return just the final batch of the step
  EVERYN,    // return every Nth batch in the step
  ALL        // return all batches in the step.
};

std::ostream &operator<<(std::ostream &, AnchorReturnTypeId);

// As an example, suppose we have an anchor scalar (0-d) tensor,
// Suppose batchesPerStep = 4 and we process them in a batch of batchSize = 2
// Suppose that the 2*4 = 8 samples are supplied in a 2d tensor with values:
// [[1, 2], [1, 0], [1, 3], [2, 0]]
// Then, under each of the AnchorReturnTypes the returned tensors are,
// FINAL       : [2, 0]                           (1-d tensor)
// EVERYN, N=2 : [[1, 0], [2, 0]]                 (2-d tensor)
// ALL         : [[1, 2], [1, 0], [1, 3], [2, 0]] (2-d tensor)

class AnchorReturnType {
public:
  // If AnchorReturnTypeId is EVERYN, a valid return period must
  // also be supplied. Othwise just supply the Id.
  AnchorReturnType(std::string artString);
  AnchorReturnType(std::string artString, int returnPeriod);

  AnchorReturnTypeId id() const { return artId_; }
  // Return period
  int rp() const;
  std::size_t hash() const;

private:
  AnchorReturnTypeId getIdFromStr(std::string artString);

  std::string artStr_;
  AnchorReturnTypeId artId_;

  int returnPeriod_;
};

// Specifies parameters for the host-device data streams.
// Used to control the amount input data processed each step,
// and how the user wants data returned
class DataFlow {
public:
  DataFlow();
  DataFlow(int batchesPerStep, const std::map<TensorId, AnchorReturnType> &);

  DataFlow(const DataFlow &rhs) = default;
  DataFlow &operator=(const DataFlow &rhs) = default;

  bool isAnchored(TensorId) const;
  bool isBatchCountingRequired() const;
  const std::vector<TensorId> &anchors() const { return v_anchors; }
  const std::vector<int> &rps() const { return v_rps; }
  int nAnchors() const { return static_cast<int>(v_anchors.size()); }
  int batchesPerStep() const { return batchesPerStep_; }
  AnchorReturnType art(TensorId anchorId) const;
  std::size_t hash() const;

private:
  /// The number of batches processed by the backend in one call to train,
  /// evaluate or infer.
  int batchesPerStep_;

  /// The set of tensors to return to the user after execution, and how
  /// frequently they are returned during multi-batch training, inference,
  /// or evaluation
  std::map<TensorId, AnchorReturnType> m_anchors;

  /// The set of anchor tensors, extracted from the map
  std::vector<TensorId> v_anchors;
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
}; // namespace std

#endif
