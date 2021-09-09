// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/autodiff.hpp"
#include "bindings/transforms/transform.hpp"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/transforms/autodiff.hpp>
#include <popart/vendored/optional.hpp>

#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace transforms {
void bindAutodiff(py::module &m) {

  py::enum_<AutodiffStitchStrategy>(
      m, "AutodiffStitchStrategy", py::module_local())
      .value("RecomputeMinimal", AutodiffStitchStrategy::RecomputeMinimal)
      .value("RecomputeAllNonInputs",
             AutodiffStitchStrategy::RecomputeAllNonInputs)
      .value("AddFwdOutputs", AutodiffStitchStrategy::AddFwdOutputs)
      .value("SafeAddFwdOutputs", AutodiffStitchStrategy::SafeAddFwdOutputs)
      .value("N", AutodiffStitchStrategy::N);

  using TensorIds              = std::vector<TensorId>;
  using FwdGraphToBwdGraphInfo = std::map<GraphId, BwdGraphInfo>;

  py::class_<Autodiff, Transform, PyTransform<Autodiff>>(m, "Autodiff")
      .def(py::init<>())
      .def("id", &Autodiff::id)
      .def("apply", py::overload_cast<Graph &>(&Autodiff::apply, py::const_))
      .def(
          "apply",
          [](Autodiff &self,
             Ir &ir,
             const GraphId &fwdGraphId,
             const TensorIds &gradsProvidedForFwdId,
             const TensorIds &gradsRequiredForFwdId,
             const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
             AutodiffStitchStrategy stitchStrategy) {
            return self.apply(ir,
                              fwdGraphId,
                              gradsProvidedForFwdId,
                              gradsRequiredForFwdId,
                              calledGraphsGradInfo,
                              stitchStrategy);
          },
          py::arg("ir"),
          py::arg("fwdGraphId"),
          py::arg("gradsProvidedForFwdId") = std::vector<TensorId>(),
          py::arg("gradsRequiredForFwdId") = std::vector<TensorId>(),
          py::arg("calledGraphsGradInfo")  = FwdGraphToBwdGraphInfo(),
          py::arg("stitchStrategy") = AutodiffStitchStrategy::SafeAddFwdOutputs)
      .def("applyToIr", &Autodiff::applyToIr)
      .def("getId", &Autodiff::getId)
      .def("getName", &Autodiff::getName);
}

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart
