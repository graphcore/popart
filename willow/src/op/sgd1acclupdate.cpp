#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

std::unique_ptr<Op>
SGD1AcclUpdateOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1AcclUpdateOp>(x, initSmm1, initSwd1, settings);
}

std::unique_ptr<Op> SGD1AcclUpdateOp::clone() const {
  return std::make_unique<SGD1AcclUpdateOp>(*this);
}

// T12001 : implement this
std::map<InIndex, TensorId> SGD1AcclUpdateOp::optimizerInputs() const {
  throw error("SGD1 optimizer inputs not implemented yet");
}

void SGD1AcclUpdateOp::appendAttributes(OpSerialiserBase &os) const {

  Op::appendAttributes(os);

  if (initSmm1.isConst()) {
    os.appendAttribute("const momentum", initSmm1.val());
  }

  if (initSwd1.isConst()) {
    os.appendAttribute("const weight decay scale factor", initSwd1.val());
  }
}

SGD1AcclUpdateOp::SGD1AcclUpdateOp(const TensorId &varToUpdate,
                                   OptimizerValue smm1,
                                   OptimizerValue swd1,
                                   const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::SGD1AcclUpdate,
                             varToUpdate,
                             opSettings),
      initSmm1(smm1), initSwd1(swd1) {}

namespace {} // namespace

} // namespace popart
