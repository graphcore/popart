#ifndef GUARD_NEURALNET_NUMERICS_HPP
#define GUARD_NEURALNET_NUMERICS_HPP

#include <poponnx/names.hpp>
namespace willow {

class Session;
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

private:
  std::map<TensorId, std::string> reports;
};

} // namespace numerics
} // namespace willow

#endif
