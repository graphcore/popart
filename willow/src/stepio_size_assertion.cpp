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

void CorrectnessAsserter::warnOfUnunsedInput(const TensorId &id) const {
  std::ostringstream oss;
  oss << "The input Tensor " << id
      << " to the ONNX Model does not appear in the optimized "
      << "PopART Ir, possibly due to constant folding. "
      << "Therefore, the input buffer provided will not be used. ";
  logging::devicex::warn(oss.str());
}

void CorrectnessAsserter::throwMissingInput(const TensorId &id) const {
  std::ostringstream oss;
  oss << "Testing that the buffer provided by user for input Tensor " << id
      << " has the correct number of elements, "
      << " But there is no Tensor named " << id << " in the Ir's main Graph, "
      << " and it does not exist in the original ONNX Model.";
  throw error(oss.str());
}

void CorrectnessAsserter::throwBadOutputSize(const TensorId &id,
                                             int64_t expected,
                                             int64_t nElms,
                                             AnchorReturnType art) const {
  std::ostringstream oss;
  oss << getBaseError("output", id, expected, nElms)
      << "\nThe anchor return type is  " << art.id();
  if (art.id() == AnchorReturnTypeId::EVERYN) {
    oss << "\nThe return period is " << art.rp();
  }
  throw error(oss.str());
}
void CorrectnessAsserter::throwMissingOutput(const TensorId &id) const {
  std::ostringstream oss;
  oss << "Testing that the buffer provided by user for output Tensor " << id
      << " has the correct number of elements, "
      << " But there is no Tensor named " << id << " in the Ir's main Graph. ";
  throw error(oss.str());
}

std::string CorrectnessAsserter::getBaseError(const std::string &io,
                                              const TensorId &id,
                                              int64_t expected,
                                              int64_t nElms) const {

  const auto onnxTensorElms = getNElms(id);

  std::ostringstream oss;
  oss << "Unexpected number of " << io << " elements for Tensor " << id
      << ". Expected " << expected << ", but received " << nElms << ".\n";
  oss << "To disable this check at your own risk, "
      << "use IStepIO::enableRuntimeAsserts.\n";
  oss << "This with, ";
  oss << "\n   replication  factor = " << rFact;
  oss << "\n   accumulation factor = " << aFact;
  oss << "\n   batches per step    = " << bps;
  oss << "\n   ONNX Tensor nelms   = " << onnxTensorElms;
  return oss.str();
}

int64_t CorrectnessAsserter::getInExpected(const TensorId &id) const {
  const auto onnxTensorElms = getNElms(id);
  const auto expected       = onnxTensorElms * rFact * aFact * bps;
  return expected;
}

int64_t CorrectnessAsserter::getArtDivisor(AnchorReturnType art) const {
  switch (art.id()) {
  case (AnchorReturnTypeId::ALL):
    return 1;
  case (AnchorReturnTypeId::FINAL):
  case (AnchorReturnTypeId::SUM):
    // Only the final micro-batch of the final batch is returned with final.
    return bps * aFact;
  case (AnchorReturnTypeId::EVERYN):
    return art.rp();
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
