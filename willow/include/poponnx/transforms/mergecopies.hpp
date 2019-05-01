#ifndef GUARD_NEURALNET_MERGECOPIES_HPP
#define GUARD_NEURALNET_MERGECOPIES_HPP

#include <poponnx/transforms/transform.hpp>

namespace poponnx {

class MergeCopies : public Transform {
public:
  static std::size_t id();

  MergeCopies() : Transform() {}
  virtual ~MergeCopies() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "MergeCopies"; }
};

} // namespace poponnx

#endif
