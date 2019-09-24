#ifndef GUARD_NEURALNET_SERIALIZE_MATMULS_HPP
#define GUARD_NEURALNET_SERIALIZE_MATMULS_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class SerializeMatMuls : public Transform {
public:
  static std::size_t id();

  SerializeMatMuls() : Transform() {}
  virtual ~SerializeMatMuls() override {}
  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "SerializeMatMuls";
  }

private:
};

} // namespace popart

#endif // GUARD_NEURALNET_SERIALIZE_MATMULS_HPP
