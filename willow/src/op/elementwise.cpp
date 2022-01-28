// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/alias/aliasmodel.hpp>
#include <popart/broadcastutil.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/identity.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

view::RegMap binaryFwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex) {
  auto out_shape = op.outShape(op.getOutIndex());
  auto in_shape  = op.inShape(argIndex);

  auto fwdMap = [out_shape, in_shape](const view::Region &r) {
    auto out_size = out_shape.size();

    if (r.isEmpty()) {
      // Make sure to handle rank-0 empty regions correctly
      return view::Regions(1, view::Region::getEmpty(out_size));
    }

    auto arg_shape = padShape(in_shape, out_size, int64_t{1});
    auto lower     = padShape(r.getLower(), out_size, int64_t{0});
    auto upper     = padShape(r.getUpper(), out_size, int64_t{1});

    // broadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        upper[i] = out_shape[i];
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
  return fwdMap;
}

view::RegMap binaryBwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex) {
  auto arg_shape = op.inShape(argIndex);
  auto arg_size  = arg_shape.size();
  auto out_shape = unpadShape(op.outShape(op.getOutIndex()), arg_size);

  auto bwdMap = [arg_size, out_shape, arg_shape](const view::Region &r) {
    if (r.isEmpty()) {
      // Make sure to handle rank-0 empty regions correctly
      return view::Regions(1, view::Region::getEmpty(arg_shape.size()));
    }

    auto lower = unpadShape(r.getLower(), arg_size);
    auto upper = unpadShape(r.getUpper(), arg_size);

    // unbroadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        lower[i] = 0;
        upper[i] = 1;
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
  return bwdMap;
}

ElementWiseUnaryOp::ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> ElementWiseUnaryOp::clone() const {
  return std::make_unique<ElementWiseUnaryOp>(*this);
}

void ElementWiseUnaryOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

ReplicatedTensorShardingIndices
ElementWiseUnaryOp::getReplicatedTensorShardingIndices() const {
  return {{{ElementWiseUnaryOp::getInIndex()},
           {ElementWiseUnaryOp::getOutIndex()}}};
}

ElementWiseUnaryBooleanOp::ElementWiseUnaryBooleanOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> ElementWiseUnaryBooleanOp::clone() const {
  return std::make_unique<ElementWiseUnaryBooleanOp>(*this);
}

std::unique_ptr<Op> ElementWiseInplaceUnaryOp::clone() const {
  return std::make_unique<ElementWiseInplaceUnaryOp>(*this);
}

void ElementWiseUnaryBooleanOp::setup() {
  outInfo(getOutIndex()) = {DataType::BOOL, inInfo(getInIndex()).shape()};
}

ElementWiseNonLinearUnaryGradOp::ElementWiseNonLinearUnaryGradOp(
    const OperatorIdentifier &_opid,
    const ElementWiseUnaryOp &op)
    : Op(_opid, op.getSettings()) {}

std::unique_ptr<Op> ElementWiseNonLinearUnaryGradOp::clone() const {
  return std::make_unique<ElementWiseNonLinearUnaryGradOp>(*this);
}

const std::vector<GradInOutMapper> &
ElementWiseNonLinearUnaryGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(),
       ElementWiseUnaryOp::getOutIndex(),
       GradOpInType::GradOut},
      {getFwdArgInIndex(), ElementWiseUnaryOp::getInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &
ElementWiseNonLinearUnaryGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ElementWiseUnaryOp::getInIndex()}};

  return outInfo;
}

void ElementWiseNonLinearUnaryGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdArgInIndex());
}

ElementWiseBinaryBaseOp::ElementWiseBinaryBaseOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> ElementWiseBinaryBaseOp::clone() const {
  return std::make_unique<ElementWiseBinaryBaseOp>(*this);
}

void ElementWiseBinaryBaseOp::setup() {
  outInfo(getOutIndex()) =
      prettyNpOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

ReplicatedTensorShardingIndices
ElementWiseBinaryBaseOp::getReplicatedTensorShardingIndices() const {
  std::set<InIndex> rtsInIndices;
  std::set<InIndex> rtsOutIndices;

  auto arg0Inf = inInfo(ElementWiseBinaryBaseOp::getArg0InIndex());
  auto arg1Inf = inInfo(ElementWiseBinaryBaseOp::getArg1InIndex());
  auto outInf  = outInfo(ElementWiseBinaryBaseOp::getOutIndex());

  // Non-broadcasted inputs that match the output need to be RTS
  // Catch all cases where the input tensor may already be sharded, but the
  // output is not yet, or vice versa
  // Cases:
  // a.) Input and output are not sharded yet (no meta shape)
  // b.) Input and output are sharded (both have meta shape)
  // c.) Input is sharded (has meta shape) but output is not yet
  // d.) Input is not sharded, but the output is (has meta shape)
  //
  // the meta-shape stores the shape that the tensor had before sharding
  // (the combined shape over all replicas that shard the tensor)

  bool arg0ms = arg0Inf.metaShape().empty();
  bool arg1ms = arg1Inf.metaShape().empty();
  bool outms  = outInf.metaShape().empty();

  if ((arg0ms && arg0Inf.shape() == outInf.shape() && outms) ||
      (!arg0ms && arg0Inf.metaShape() == outInf.metaShape() && !outms) ||
      (!arg0ms && arg0Inf.metaShape() == outInf.shape() && outms) ||
      (arg0ms && arg0Inf.shape() == outInf.metaShape() && !outms)) {
    rtsInIndices.insert(ElementWiseBinaryBaseOp::getArg0InIndex());
  }

  if ((arg1ms && arg1Inf.shape() == outInf.shape() && outms) ||
      (!arg1ms && arg1Inf.metaShape() == outInf.metaShape() && !outms) ||
      (!arg1ms && arg1Inf.metaShape() == outInf.shape() && outms) ||
      (arg1ms && arg1Inf.shape() == outInf.metaShape() && !outms)) {
    rtsInIndices.insert(ElementWiseBinaryBaseOp::getArg1InIndex());
  }

  if (!rtsInIndices.empty()) {
    rtsOutIndices.insert(ElementWiseBinaryBaseOp::getOutIndex());
  }

  return {{rtsInIndices, rtsOutIndices}};
}

ElementWiseBinaryOp::ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &_settings)
    : ElementWiseBinaryBaseOp(_opid, _settings) {}

std::unique_ptr<Op> ElementWiseBinaryOp::clone() const {
  return std::make_unique<ElementWiseBinaryOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
ElementWiseBinaryOp::inplacePriorityDefault() const {
  auto outSize  = outInfo(getOutIndex()).nelms();
  auto arg0Size = inInfo(getArg0InIndex()).nelms();
  auto arg1Size = inInfo(getArg1InIndex()).nelms();

  std::vector<std::tuple<OperatorIdentifier, float>> result;

  if (hasLhsInplaceVariant() && outSize == arg0Size) {
    auto lhsPriority = getInplacePriority(getLhsOperatorIdentifier());
    result.emplace_back(getLhsOperatorIdentifier(), lhsPriority);
  }
  if (hasRhsInplaceVariant() && outSize == arg1Size) {
    auto rhsPriority = getInplacePriority(getRhsOperatorIdentifier());
    result.emplace_back(getRhsOperatorIdentifier(), rhsPriority);
  }

  return result;
}

std::unique_ptr<Op> ElementWiseBinaryOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (hasLhsInplaceVariant() && operator_id == getLhsOperatorIdentifier()) {
    return getLhsInplaceVariant();
  } else if (hasRhsInplaceVariant() &&
             operator_id == getRhsOperatorIdentifier()) {
    return getRhsInplaceVariant();
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void ElementWiseBinaryOp::setInplacePriority(
    const OperatorIdentifier &opidParam,
    float priority) {
  inplacePriorities[opidParam] = priority;
}

float ElementWiseBinaryOp::getInplacePriority(
    const OperatorIdentifier &opidParam) const {
  constexpr float defaultPriority = 10.0f;

  if (inplacePriorities.count(opidParam) == 0) {
    return defaultPriority;
  } else {
    return inplacePriorities.at(opidParam);
  }
}

bool ElementWiseBinaryOp::hasLhsInplaceVariant() const { return false; }

bool ElementWiseBinaryOp::hasRhsInplaceVariant() const { return false; }

std::unique_ptr<Op> ElementWiseBinaryOp::getLhsInplaceVariant() const {
  throw error("Operator {} cannot return LHS inplace variant", opid);
}

std::unique_ptr<Op> ElementWiseBinaryOp::getRhsInplaceVariant() const {
  throw error("Operator {} cannot return RHS inplace variant", opid);
}

OperatorIdentifier ElementWiseBinaryOp::getLhsOperatorIdentifier() const {
  throw error("Operator {} does not have LHS OperatorIdentifier", opid);
}

OperatorIdentifier ElementWiseBinaryOp::getRhsOperatorIdentifier() const {
  throw error("Operator {} does not have RHS OperatorIdentifier", opid);
}

std::unique_ptr<Op> ElementWiseBinaryInplaceLhsOp::clone() const {
  return std::make_unique<ElementWiseBinaryInplaceLhsOp>(*this);
}

std::unique_ptr<Op> ElementWiseBinaryInplaceRhsOp::clone() const {
  return std::make_unique<ElementWiseBinaryInplaceRhsOp>(*this);
}

ElementWiseBinaryGradOp::ElementWiseBinaryGradOp(
    const OperatorIdentifier &_opid,
    const std::vector<int64_t> &reduction_axes_,
    const TensorInfo &forward_op_arg_info_,
    const Op::Settings &settings_)
    : Op(_opid, settings_), forward_op_arg_info(forward_op_arg_info_),
      reduction_axes(reduction_axes_) {}

void ElementWiseBinaryGradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
}

std::unique_ptr<Op> ElementWiseBinaryArg0GradOp::clone() const {
  return std::make_unique<ElementWiseBinaryArg0GradOp>(*this);
}

std::unique_ptr<Op> ElementWiseBinaryArg1GradOp::clone() const {
  return std::make_unique<ElementWiseBinaryArg1GradOp>(*this);
}

BinaryComparisonOp::BinaryComparisonOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> BinaryComparisonOp::clone() const {
  return std::make_unique<BinaryComparisonOp>(*this);
}

void BinaryComparisonOp::setup() {
  outInfo(getOutIndex()) = {DataType::BOOL,
                            prettyNpOut(inInfo(getArg0InIndex()).shape(),
                                        inInfo(getArg1InIndex()).shape())};
}

void ElementWiseBinaryBaseOp::growAliasModel(AliasModel &m) const {
  m.insertBinaryModifier(*this);
}

poprithms::memory::inplace::Proposal
ElementWiseBinaryOp::mapInplaceProposal(const AliasModel &aliasModel,
                                        OperatorIdentifier opId) const {

  const std::string inplaceName = opId.type;
  auto index = (inplaceName.find("Rhs") != std::string::npos) ? 1 : 0;
  return {aliasModel.getGate(id), index};
}

poprithms::memory::inplace::Proposal
ElementWiseUnaryOp::mapInplaceProposal(const AliasModel &aliasModel,
                                       OperatorIdentifier id) const {
  return mapInplaceProposalGate0(aliasModel, id);
}

void ElementWiseUnaryOp::growAliasModel(AliasModel &m) const {
  if (!isIdentity()) {
    m.insertUnaryModifier0(*this);
  } else {

    auto id0 = m.getPoprithmsTensorId(inId(0));

    const auto gate =
        isOutplace() ? m.g.aliasGate({id0}) : m.g.aliasGate({id0}, 0);
    m.insertTensor(gate, *outTensor(0));
  }
}

} // namespace popart
