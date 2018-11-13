#ifndef GUARD_NEURALNET_NUMERICS_HPP
#define GUARD_NEURALNET_NUMERICS_HPP

#include <poponnx/names.hpp>
namespace willow {

class Session;
namespace numerics {

class NumericsReport {
public:
  // compare update steps for model A: A0 -> A1
  //                      and model B: B0 -> B1
  NumericsReport(std::string A0, // A starts
                 std::string A1, // A ends
                 std::string B0, // B starts
                 std::string B1  // B ends
  );
  std::string report(TensorId) const;
  std::string fullReport() const;

private:
  std::map<TensorId, std::string> reports;
};

} // namespace numerics
} // namespace willow

#endif
