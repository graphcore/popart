// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/autodiff.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/transforms/autodiff.hpp>
#include <popart/vendored/optional.hpp>

#include "bindings/transforms/transform.hpp"
#include "popart/graph.hpp" // IWYU pragma: keep
#include "popart/graphid.hpp"
#include "popart/ir.hpp" // IWYU pragma: keep
#include "popart/tensordebuginfo.hpp"

#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/stl.h>        // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
class Transform;

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
             const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
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
          py::arg("gradsProvidedForFwdId") = TensorIds(),
          py::arg("gradsRequiredForFwdId") = nonstd::optional<TensorIds>(),
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
