#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1accumulate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op>
SGD1AccumulateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1AccumulateOp>(x, initDpsf1, settings);
}

std::unique_ptr<Op> SGD1AccumulateOp::clone() const {
  return std::make_unique<SGD1AccumulateOp>(*this);
}

// T12001
std::map<InIndex, TensorId> SGD1AccumulateOp::optimizerInputs() const {
  throw error("SGD1 optimizer inputs not implemented");
}

void SGD1AccumulateOp::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initDpsf1.isConst()) {
    os.appendAttribute("const dampening scale factor", initDpsf1.val());
  }
}

SGD1AccumulateOp::SGD1AccumulateOp(const TensorId &varToUpdate,
                                   OptimizerValue dpsf1,
                                   const Op::Settings &settings)
    : VarUpdateOp(Onnx::CustomOperators::SGD1Accumulate, varToUpdate, settings),
      initDpsf1(dpsf1) {}

namespace {} // namespace

} // namespace popart
