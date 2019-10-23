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
  return std::make_unique<SGD1AcclUpdateOp>(x, initMm1, initWdsf1, settings);
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

  if (initMm1.isConst()) {
    os.appendAttribute("const momentum", initMm1.val());
  }

  if (initWdsf1.isConst()) {
    os.appendAttribute("const weight decay scale factor", initWdsf1.val());
  }
}

SGD1AcclUpdateOp::SGD1AcclUpdateOp(const TensorId &varToUpdate,
                                   OptimizerValue mm1,
                                   OptimizerValue wdsf1,
                                   const Op::Settings &opSettings)
    : VarUpdateOp(Onnx::CustomOperators::SGD1AcclUpdate,
                  varToUpdate,
                  opSettings),
      initMm1(mm1), initWdsf1(wdsf1) {}

namespace {} // namespace

} // namespace popart
