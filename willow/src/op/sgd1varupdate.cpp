#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op> SGD1VarUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1VarUpdateOp>(x, initSlr1, settings);
}

std::unique_ptr<Op> SGD1VarUpdateOp::clone() const {
  return std::make_unique<SGD1VarUpdateOp>(*this);
}

// T12001
std::map<InIndex, TensorId> SGD1VarUpdateOp::optimizerInputs() const {
  throw error("SGD1 optimizer inputs not implemented yet");
}

void SGD1VarUpdateOp::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initSlr1.isConst()) {
    os.appendAttribute("const scaled learning rate", initSlr1.val());
  }
}

SGD1VarUpdateOp::SGD1VarUpdateOp(const TensorId &varToUpdate,
                                 OptimizerValue slr1,
                                 const Op::Settings &opSettings)
    : VarUpdateOp(Onnx::CustomOperators::SGD1VarUpdate,
                  varToUpdate,
                  opSettings),
      initSlr1(slr1) {}

namespace {} // namespace

} // namespace popart
