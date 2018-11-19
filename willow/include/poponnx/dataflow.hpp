#ifndef GUARD_NEURALNET_DATAFLOW_HPP
#define GUARD_NEURALNET_DATAFLOW_HPP

#include <poponnx/dataflow.hpp>
#include <poponnx/tensorinfo.hpp>

namespace willow {

// An anchor tensor is a tensor which the user wants returned
// after a step is run. Anchors are essentially what tensorflow calls
// "fetches". AnchorReturnType specifies what exactly should be
// returned for a tensor, currently the 3 options are:

enum class AnchorReturnType {
  FINAL = 0, // return just the final batch of the step
  SUM,       // return the sum of all samples at the end
             // of the step
  ALL        // return all batches in the step.
};
// As an example, suppose we have an anchor scalar (0-d) tensor,
// Suppose batchesPerStep = 3 and batchSize = 2.
// Suppose that the 2*3 = 6 samples processed in a step have values
// 1, 2, 1, 0, 1, 3
// Then, under each of the AnchorReturnTypes the returned tensors are,
// FINAL : [1, 3]             (1-d tensor)
// SUM   : 8                  (0-d tensor)
// ALL   : [1, 2, 1, 0, 1, 3] (1-d tensor)

// Describe what and when the user wants returned.
class DataFlow {
public:
  DataFlow();
  DataFlow(int batchesPerStep,
           int batchSize,
           const std::vector<TensorId> &,
           AnchorReturnType);

  DataFlow(const DataFlow &rhs) = default;
  DataFlow &operator=(const DataFlow &rhs) = default;

  bool isAnchored(TensorId) const;
  const std::vector<TensorId> &anchors() const { return v_anchors; }
  int nAnchors() const { return static_cast<int>(v_anchors.size()); }
  int batchSize() const { return batchSize_; }
  int batchesPerStep() const { return batchesPerStep_; }
  AnchorReturnType art() const { return art_; }

private:
  /// The number of batches processed by the backend in one call to train,
  /// evaluate or infer.
  int batchesPerStep_;

  /// The size of the minibatch
  int batchSize_;

  /// The set of tensors to return to the user after execution
  std::vector<TensorId> v_anchors;
  std::set<TensorId> s_anchors;

  /// Anchor return type for multi-batch training, inference, or evaluation
  AnchorReturnType art_;
};

} // namespace willow

#endif
