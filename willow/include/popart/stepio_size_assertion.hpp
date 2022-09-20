// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_STEPIO_SIZE_ASSERTION_HPP_
#define POPART_WILLOW_INCLUDE_POPART_STEPIO_SIZE_ASSERTION_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <popart/ir.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/dataflow.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
namespace iosizecheck {

class CorrectnessAsserter {
public:
  CorrectnessAsserter(const popx::Executablex &_exe_)
      : exe(_exe_), ir(exe.ir()),
        rFact(ir.getSessionOptions().replicatedGraphCount),
        aFact(ir.getSessionOptions().accumulationFactor),
        bps(ir.getDataFlow().batchesPerStep()), onnxIns(ir.getModelInputIds()) {
  }

  // for every element in "m", assert that the number of elements agrees with
  // the corresponding Tensor in "ir"
  //
  // template types:
  // M is a map type from TensorId to T
  // G is a class with operator(const T &) and returns number of elements
  template <class M, class G> void checkIn(const M &m, const G &g) const {
    const bool hasOnnxModel = ir.hasOnnxModel();
    for (auto x = m.cbegin(); x != m.cend(); ++x) {
      const auto id = x->first;
      if (exe.containsTensor(id)) {
        const auto expected = getInExpected(id);
        const auto nElms    = g(x->second);
        if (nElms != expected) {
          throwBadInputSize(id, expected, nElms);
        } else if (hasOnnxModel &&
                   (std::find(onnxIns.cbegin(), onnxIns.cend(), id) ==
                    onnxIns.cend())) {
          throwIncorrectInput(id);
        }
      } else if (std::find(onnxIns.cbegin(), onnxIns.cend(), id) !=
                 onnxIns.cend()) {
        warnOfUnunsedInput(id, hasOnnxModel);
      } else {
        throwMissingInput(id, hasOnnxModel);
      }
    }
  }

  template <class M, class G> void checkOut(const M &m, const G &g) const {
    for (auto x = m.cbegin(); x != m.cend(); ++x) {
      auto id = x->first;
      if (exe.containsTensor(id)) {
        auto expected    = getOutExpected(id);
        const auto nElms = g(x->second);
        auto art         = ir.getDataFlow().art(id);
        if (nElms != expected) {
          throwBadOutputSize(id, expected, nElms, art);
        }
      } else {
        throwMissingOutput(id);
      }
    }
  }

private:
  uint64_t getNElms(const TensorId &id) const {
    return exe.getTensor(id)->info.nelms();
  }
  std::string getBaseError(const std::string &io,
                           const TensorId &id,
                           int64_t expected,
                           int64_t nElms) const;

  int64_t getBaseExpected(const TensorId &id) const;

  int64_t getInExpected(const TensorId &id) const;

  int64_t getArtDivisor(AnchorReturnType art) const;

  int64_t getOutExpected(const TensorId &id) const;

  [[noreturn]] void
  throwBadInputSize(const TensorId &, int64_t expected, int64_t nElms) const;

  [[noreturn]] void throwBadOutputSize(const TensorId &,
                                       int64_t expected,
                                       int64_t nElms,
                                       AnchorReturnType art) const;

  [[noreturn]] void throwMissingInput(const TensorId &,
                                      bool isFromOnnx = true) const;
  [[noreturn]] void throwIncorrectInput(const TensorId &) const;
  [[noreturn]] void throwMissingOutput(const TensorId &) const;

  void warnOfUnunsedInput(const TensorId &, bool isFromOnnx = true) const;

  const popx::Executablex &exe;
  const Ir &ir;
  int64_t rFact;
  int64_t aFact;
  int64_t bps;
  const std::vector<TensorId> onnxIns;
};

template <class M, class G>
void assertInCorrect(const popx::Executablex &exe, const M &_m_, const G &g) {
  CorrectnessAsserter(exe).checkIn<M, G>(_m_, g);
}

template <class M, class G>
void assertOutCorrect(const popx::Executablex &exe, const M &_m_, const G &g) {
  CorrectnessAsserter(exe).checkOut<M, G>(_m_, g);
}

} // namespace iosizecheck
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_STEPIO_SIZE_ASSERTION_HPP_
