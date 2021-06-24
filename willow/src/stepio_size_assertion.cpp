// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <popart/error.hpp>
#include <popart/stepio_size_assertion.hpp>

namespace popart {
namespace iosizecheck {

void CorrectnessAsserter::throwBadInputSize(const TensorId &id,
                                            int64_t expected,
                                            int64_t nElms) const {
  throw error(getBaseError("input", id, expected, nElms));
}

void CorrectnessAsserter::warnOfUnunsedInput(const TensorId &id,
                                             bool isFromOnnx) const {
  std::ostringstream oss;
  oss << "The input Tensor " << id << " to the" << (isFromOnnx ? " ONNX " : " ")
      << "Model does not appear in the optimized "
      << "PopART Ir, possibly due to constant folding. "
      << "Therefore, the input buffer provided will not be used. ";
  logging::devicex::warn(oss.str());
}

void CorrectnessAsserter::throwMissingInput(const TensorId &id,
                                            bool isFromOnnx) const {
  std::ostringstream oss;
  oss << "Testing that the buffer provided by user for input Tensor " << id
      << " has the correct number of elements,"
      << " But there is no Tensor named " << id << " in the Ir's main Graph"
      << (isFromOnnx ? ", and it does not exist in the original ONNX Model."
                     : ".");
  throw error(oss.str());
}

void CorrectnessAsserter::throwIncorrectInput(const TensorId &id) const {
  std::ostringstream oss;
  oss << "The tensor '" << id
      << "' has been provided as one of the inputs to the stepIO, but it is "
         "not registered as one of the inputs to the model. Please check this "
         "tensor and your stepIO.";
  throw error(oss.str());
}

void CorrectnessAsserter::throwBadOutputSize(const TensorId &id,
                                             int64_t expected,
                                             int64_t nElms,
                                             AnchorReturnType art) const {
  std::ostringstream oss;
  oss << getBaseError("output", id, expected, nElms)
      << "\nThe anchor return type is  " << art.id();
  if (art.id() == AnchorReturnTypeId::EveryN) {
    oss << "\nThe return period is " << art.rp();
  }
  throw error(oss.str());
}
void CorrectnessAsserter::throwMissingOutput(const TensorId &id) const {
  std::ostringstream oss;
  oss << "Testing that the buffer provided by user for output Tensor " << id
      << " has the correct number of elements, "
      << " But there is no Tensor named " << id << " in the Ir's tensors. ";
  throw error(oss.str());
}

std::string CorrectnessAsserter::getBaseError(const std::string &io,
                                              const TensorId &id,
                                              int64_t expected,
                                              int64_t nElms) const {

  const auto tensorElms = getNElms(id);

  std::ostringstream oss;
  oss << "Unexpected number of " << io << " elements for Tensor " << id
      << ". Expected " << expected << ", but received " << nElms << ".\n";
  oss << "To disable this check at your own risk, "
      << "use IStepIO::enableRuntimeAsserts.\n";
  oss << "This with, ";
  oss << "\n   replication  factor = " << rFact;
  oss << "\n   accumulation factor = " << aFact;
  oss << "\n   batches per step    = " << bps;
  oss << "\n   Tensor nelms        = " << tensorElms;
  return oss.str();
}

int64_t CorrectnessAsserter::getInExpected(const TensorId &id) const {
  const auto onnxTensorElms = getNElms(id);
  const auto expected       = onnxTensorElms * rFact * aFact * bps;
  return expected;
}

int64_t CorrectnessAsserter::getArtDivisor(AnchorReturnType art) const {
  switch (art.id()) {
  case (AnchorReturnTypeId::All):
    return 1;
  case (AnchorReturnTypeId::Final):
  case (AnchorReturnTypeId::Sum):
    // Only the final micro-batch of the final batch is returned with final.
    return bps * aFact;
  case (AnchorReturnTypeId::EveryN):
    return art.rp();
  default:
    throw error("Unknown anchor return type");
  }
}

int64_t CorrectnessAsserter::getOutExpected(const TensorId &id) const {
  if (!ir.getDataFlow().isAnchored(id)) {
    throw internal_error(
        "Non-anchored Tensor in CorrectnessAsserter::checkOut");
  }
  return getInExpected(id) / getArtDivisor(ir.getDataFlow().art(id));
}

} // namespace iosizecheck
} // namespace popart
