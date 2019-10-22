#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {
GetRandomSeedOp::GetRandomSeedOp(const OperatorIdentifier &_opid,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void GetRandomSeedOp::setup() {
  outInfo(getUpdatedSeedOutIndex()) = inInfo(getSeedInIndex());
}

std::unique_ptr<Op> GetRandomSeedOp::clone() const {
  return std::make_unique<GetRandomSeedOp>(*this);
}

view::Region GetRandomSeedOp::aliases(InIndex index) const {
  return view::Region::getFull(inShape(index));
}

// Modifies is the same as aliases
view::Region GetRandomSeedOp::modifies(InIndex index) const {
  return aliases(index);
}

namespace {} // namespace

} // namespace popart
