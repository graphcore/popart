// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NUMERICS_HPP
#define GUARD_NEURALNET_NUMERICS_HPP

#include <cmath>

#include <sstream>
#include <string>
#include <popart/names.hpp>

namespace popart {
namespace numerics {

class NumericsReport {
public:
  // compare update steps for model A: AStarts -> AEnds
  //                      and model B: BStarts -> BEnds
  NumericsReport(std::string AStarts, // A starts
                 std::string AEnds,   // A ends
                 std::string BStarts, // B starts
                 std::string BEnds    // B ends
  );
  std::string report(TensorId) const;
  std::string fullReport() const;
  std::map<TensorId, float> getRelativeErrors();

private:
  std::map<TensorId, std::string> reports;
  std::map<TensorId, float> relerrs;
};

template <class T> class NumericsTracker {
private:
  // sums of squares of weight differences
  T ss_dA{0};
  T ss_dB{0};
  T ss_dAB{0};

public:
  void insert(T v_AStarts, T v_AEnds, T v_BStarts, T v_BEnds) {
    T dA = v_AEnds - v_AStarts;
    T dB = v_BEnds - v_BStarts;
    ss_dA += dA * dA;
    ss_dB += dB * dB;
    ss_dAB += (dA - dB) * (dA - dB);
  }

  std::string str() {
    std::stringstream ss;
    ss.precision(16);
    ss << "|dA - dB|^2 / (|dA||dB| + 1e-8)  = " << getRelativeError()
       << "  and  |dA||dB| = " << std::sqrt(ss_dA * ss_dB)
       << "  and  |dA| = " << std::sqrt(ss_dA)
       << "  and  |dB| = " << std::sqrt(ss_dB);
    return ss.str();
  }

  T getRelativeError() { return (ss_dAB) / (std::sqrt(ss_dA * ss_dB) + 1e-8f); }
};

} // namespace numerics
} // namespace popart

#endif
