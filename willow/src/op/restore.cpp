#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

RestoreOp::RestoreOp(const OperatorIdentifier &_opid,
                     int64_t stashSize_,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), stashSize(stashSize_) {}

std::unique_ptr<Op> RestoreOp::clone() const {
  return std::make_unique<RestoreOp>(*this);
}

void RestoreOp::setup() {
  auto stash   = input->tensor(getStashInIndex());
  auto stashOp = stash->getProducer();
  auto act     = stashOp->input->tensor(StashOp::getInIndex());
  outInfo(getRestoredActOutIndex()) = act->info;
}

TensorId RestoreOp::getRestoredTensorId() const {
  auto stash   = input->tensor(getStashInIndex());
  auto stashOp = stash->getProducer();
  auto act     = stashOp->input->tensor(StashOp::getInIndex());
  return reservedRestoredPrefix() + act->id;
}

void RestoreOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("stashSize", stashSize);
}

RestoreInplaceOp::RestoreInplaceOp(const OperatorIdentifier &_opid,
                                   int64_t stashSize_,
                                   const Op::Settings &settings_)
    : RestoreOp(_opid, stashSize_, settings_) {}

std::unique_ptr<Op> RestoreInplaceOp::clone() const {
  return std::make_unique<RestoreInplaceOp>(*this);
}

view::Region RestoreInplaceOp::aliases(InIndex index) const {
  if (index == getActToRestoreInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

// Modifies is the same as aliases
view::Region RestoreInplaceOp::modifies(InIndex index) const {
  return aliases(index);
}

} // namespace popart
