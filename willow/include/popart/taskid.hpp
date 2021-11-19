
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_TASK_ID_HPP
#define NEURALNET_TASK_ID_HPP

#include <ostream>

#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

// Forward declaration.
class TaskId;

} // namespace popart

namespace std {
template <> struct hash<popart::TaskId> {
  std::size_t operator()(const popart::TaskId &taskId) const;
};

} // namespace std

namespace popart {

// Output to stream.
std::ostream &operator<<(std::ostream &out, const TaskId &taskId);

/**
 * A class describing an IR-to-poplar lowering task. This is a class that is
 * cheap to construct. We construct and compare TaskIds a lot in
 * `irlowering.cpp` so it pays to make these cheap operations. Note that
 * previously TaskId was a `std::string` and creating a TaskId typically
 * involved some string manipulation, meaning heap memory may be involved.
 * Comparing strings for equality or ordering strings is also typically
 * not constant-time.
 */
class TaskId {
public:
  /**
   * TaskId type.
   **/
  enum class Type {
    AnchorStreamToHostTask = 0,
    AnchorSumTask,
    AnchorToHostTask,
    FromHostTask,
    FromOpTask,
    InitBatchCounterTensorsTask,
    InitRngSeedsTask,
    InitRandomSeedTask,
    InitRngStateTensorTask,
    InitTensorTask,
    PipelinedCopyTask,
    RngStateFromHostTask,
    RngStateToHostTask,
    SetInitTensorValTask,
    StreamFromHostTask,
    UpdateBatchCountTask,
    WeightStreamToHostTask,
    WeightToHostTask,
    Undefined
  };

  // Constructors.
  TaskId();
  explicit TaskId(Type type);
  TaskId(Type, const TensorId &tensorId);
  TaskId(Type type, const OpId &opId, const OperatorIdentifier &opIdentifier);
  TaskId(Type type,
         const OpId &opId,
         const OperatorIdentifier &opIdentifier,
         const OpxGrowPartId &opxGrowPartId);
  TaskId(Type type,
         nonstd::optional<TensorId> tensorId,
         nonstd::optional<OpId> opId,
         nonstd::optional<OperatorIdentifier> opIdentifier,
         nonstd::optional<OpxGrowPartId> opxGrowPartId);

  // True if type == Undefined (default constructed).
  bool empty() const;

  // Less than comparison.
  bool operator<(const TaskId &rhs) const;
  // Equality.
  bool operator==(const TaskId &rhs) const;

  const nonstd::optional<TensorId> &getTensorId() const { return tensorId; }

  const Type &getType() const { return type; }

private:
  // The type of task.
  Type type;
  // Optional tensor param.
  nonstd::optional<TensorId> tensorId;
  // Optional OpId.
  nonstd::optional<OpId> opId;
  // Optional OperatorIdentifier.
  nonstd::optional<OperatorIdentifier> opIdentifier;
  // Optional OpxGrowPartId
  nonstd::optional<OpxGrowPartId> opxGrowPartId;

  // Ensure the hash function / stream operator can access members.
  friend std::size_t std::hash<TaskId>::operator()(const TaskId &) const;
  friend std::ostream &operator<<(std::ostream &out, const TaskId &taskId);
};

// Output to stream.
std::ostream &operator<<(std::ostream &out, const TaskId::Type &type);

} // namespace popart

#endif
