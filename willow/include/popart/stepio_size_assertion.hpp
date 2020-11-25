// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STEPIO_SIZE_ASSERT_HPP
#define GUARD_NEURALNET_STEPIO_SIZE_ASSERT_HPP

#include <popart/ir.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

// TODO(T15449) factorize out templates advance and get from StepIO and PyStepIO
//

namespace popart {
namespace iosizecheck {

class CorrectnessAsserter {
public:
  CorrectnessAsserter(const Ir &_ir_)
      : ir(_ir_), tensors(ir.getMainGraphTensors()),
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
    for (auto x = m.cbegin(); x != m.cend(); ++x) {
      const auto id = x->first;
      if (tensors.contains(id)) {
        const auto expected = getInExpected(id);
        const auto nElms    = g(x->second);
        std::ostringstream oss;
        if (nElms != expected) {
          throwBadInputSize(id, expected, nElms);
        } else if (std::find(onnxIns.cbegin(), onnxIns.cend(), id) ==
                   onnxIns.cend()) {
          throwIncorrectInput(id);
        }
      } else if (std::find(onnxIns.cbegin(), onnxIns.cend(), id) !=
                 onnxIns.cend()) {
        warnOfUnunsedInput(id);
      } else {
        throwMissingInput(id);
      }
    }
  }

  template <class M, class G> void checkOut(const M &m, const G &g) const {
    for (auto x = m.cbegin(); x != m.cend(); ++x) {
      auto id = x->first;
      if (tensors.contains(id)) {
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
    return tensors.get(id)->info.nelms();
  }
  std::string getBaseError(const std::string &io,
                           const TensorId &id,
                           int64_t expected,
                           int64_t nElms) const;

  int64_t getInExpected(const TensorId &id) const;

  int64_t getArtDivisor(AnchorReturnType art) const;

  int64_t getOutExpected(const TensorId &id) const;

  [[noreturn]] void
  throwBadInputSize(const TensorId &, int64_t expected, int64_t nElms) const;

  [[noreturn]] void throwBadOutputSize(const TensorId &,
                                       int64_t expected,
                                       int64_t nElms,
                                       AnchorReturnType art) const;

  [[noreturn]] void throwMissingInput(const TensorId &) const;
  [[noreturn]] void throwIncorrectInput(const TensorId &) const;
  [[noreturn]] void throwMissingOutput(const TensorId &) const;

  void warnOfUnunsedInput(const TensorId &) const;

  const Ir &ir;
  const Tensors &tensors;
  int64_t rFact;
  int64_t aFact;
  int64_t bps;
  const std::vector<TensorId> onnxIns;
};

template <class M, class G>
void assertInCorrect(const Ir &_ir_, const M &_m_, const G &g) {
  CorrectnessAsserter(_ir_).checkIn<M, G>(_m_, g);
}

template <class M, class G>
void assertOutCorrect(const Ir &_ir_, const M &_m_, const G &g) {
  CorrectnessAsserter(_ir_).checkOut<M, G>(_m_, g);
}

} // namespace iosizecheck
} // namespace popart

#endif
