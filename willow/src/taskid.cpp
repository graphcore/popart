// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/taskid.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

TaskId::TaskId()
    : TaskId{TaskId::Type::Undefined,
             nonstd::nullopt,
             nonstd::nullopt,
             nonstd::nullopt,
             nonstd::nullopt} {}

TaskId::TaskId(Type type)
    : TaskId{type,
             nonstd::nullopt,
             nonstd::nullopt,
             nonstd::nullopt,
             nonstd::nullopt} {}

TaskId::TaskId(Type type, const TensorId &tensorId)
    : TaskId{type,
             tensorId,
             nonstd::nullopt,
             nonstd::nullopt,
             nonstd::nullopt} {}

TaskId::TaskId(Type type,
               const OpId &opId,
               const OperatorIdentifier &opIdentifier)
    : TaskId{type, nonstd::nullopt, opId, opIdentifier, nonstd::nullopt} {}

TaskId::TaskId(Type type,
               const OpId &opId,
               const OperatorIdentifier &opIdentifier,
               const OpxGrowPartId &opxGrowPartId)
    : TaskId{type, nonstd::nullopt, opId, opIdentifier, opxGrowPartId} {}

TaskId::TaskId(Type type_,
               nonstd::optional<TensorId> tensorId_,
               nonstd::optional<OpId> opId_,
               nonstd::optional<OperatorIdentifier> opIdentifier_,
               nonstd::optional<OpxGrowPartId> opxGrowPartId_)
    : type{type_}, tensorId{tensorId_}, opId{opId_},
      opIdentifier{opIdentifier_}, opxGrowPartId(opxGrowPartId_) {}

bool TaskId::empty() const { return type == TaskId::Type::Undefined; }

bool TaskId::operator<(const TaskId &rhs) const {

  if (type != rhs.type) {
    return (type < rhs.type);
  }

  if (tensorId != rhs.tensorId) {
    return (tensorId < rhs.tensorId);
  }

  if (opId != rhs.opId) {
    return (opId < rhs.opId);
  }

  if (opIdentifier != rhs.opIdentifier) {
    return (opIdentifier < rhs.opIdentifier);
  }

  if (opxGrowPartId != rhs.opxGrowPartId) {
    return (opxGrowPartId < rhs.opxGrowPartId);
  }

  return false;
}

bool TaskId::operator==(const TaskId &rhs) const {
  return (type == rhs.type) && (tensorId == rhs.tensorId) &&
         (opId == rhs.opId) && (opIdentifier == rhs.opIdentifier) &&
         (opxGrowPartId == rhs.opxGrowPartId);
}

std::ostream &operator<<(std::ostream &out, const TaskId &taskId) {
  out << taskId.type;
  if (taskId.tensorId) {
    out << "_" << *taskId.tensorId;
  }
  if (taskId.opId) {
    out << "_" << *taskId.opId;
  }
  if (taskId.opIdentifier) {
    out << "_" << *taskId.opIdentifier;
  }
  if (taskId.opxGrowPartId) {
    out << "_" << *taskId.opxGrowPartId;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const TaskId::Type &type) {
  switch (type) {
  case (TaskId::Type::AnchorStreamToHostTask): {
    out << "anchorStreamToHostTask";
    break;
  }
  case (TaskId::Type::AnchorSumTask): {
    out << "anchorSumTask";
    break;
  }
  case (TaskId::Type::AnchorToHostTask): {
    out << "anchorToHostTask";
    break;
  }
  case (TaskId::Type::FromHostTask): {
    out << "fromHostTask";
    break;
  }
  case (TaskId::Type::FromOpTask): {
    out << "fromOpTask";
    break;
  }
  case (TaskId::Type::InitBatchCounterTensorsTask): {
    out << "initBatchCounterTensorsTask";
    break;
  }
  case (TaskId::Type::InitRandomSeedTask): {
    out << "initRandomSeedTask";
    break;
  }
  case (TaskId::Type::InitRngStateTensorTask): {
    out << "initRngStateTensorTask";
    break;
  }
  case (TaskId::Type::InitTensorTask): {
    out << "initTensorTask";
    break;
  }
  case (TaskId::Type::PipelinedCopyTask): {
    out << "pipelinedCopyTask";
    break;
  }
  case (TaskId::Type::RngStateFromHostTask): {
    out << "rngStateFromHostTask";
    break;
  }
  case (TaskId::Type::RngStateToHostTask): {
    out << "rngStateToHostTask";
    break;
  }
  case (TaskId::Type::SetInitTensorValTask): {
    out << "setInitTensorValTask";
    break;
  }
  case (TaskId::Type::StreamFromHostTask): {
    out << "streamFromHostTask";
    break;
  }
  case (TaskId::Type::UpdateBatchCountTask): {
    out << "updateBatchCountTask";
    break;
  }
  case (TaskId::Type::WeightStreamToHostTask): {
    out << "weightStreamToHostTask";
    break;
  }
  case (TaskId::Type::WeightToHostTask): {
    out << "weightToHostTask";
    break;
  }
  case (TaskId::Type::Undefined): {
    out << "undefined";
    break;
  }
  default: {
    throw internal_error("[TaskId] Unsupported TaskId::Type value ({})",
                         static_cast<int>(type));
  }
  }

  return out;
}

} // namespace popart

namespace std {

using namespace popart;

std::size_t
hash<popart::TaskId>::operator()(const popart::TaskId &taskId) const {
  std::size_t seed = 0;

  boost::hash_combine(seed, taskId.type);
  if (taskId.tensorId) {
    boost::hash_combine<TensorId>(seed, *taskId.tensorId);
  }
  if (taskId.opId) {
    boost::hash_combine<OpId>(seed, *taskId.opId);
  }
  if (taskId.opIdentifier) {
    boost::hash_combine<OpDomain>(seed, taskId.opIdentifier->domain);
    boost::hash_combine<OpType>(seed, taskId.opIdentifier->type);
    boost::hash_combine<OpVersion>(seed, taskId.opIdentifier->version);
    boost::hash_combine<int>(seed, taskId.opIdentifier->numInputs.min);
    boost::hash_combine<int>(seed, taskId.opIdentifier->numInputs.max);
    boost::hash_combine<int>(seed, taskId.opIdentifier->numOutputs);
  }
  if (taskId.opxGrowPartId) {
    boost::hash_combine<OpxGrowPartId>(seed, *taskId.opxGrowPartId);
  }
  return seed;
}

} // namespace std
