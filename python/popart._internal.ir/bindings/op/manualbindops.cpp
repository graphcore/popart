// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/manualbindops.hpp"
#include "bindings/basicoptionals.hpp"
#include "bindings/op/matmul.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/op/varupdate.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <popart/basicoptionals.hpp>

#include <popart/op/accumulate.hpp>
#include <popart/op/accumulatorscale.hpp>
#include <popart/op/accumulatorzero.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/varupdate.hpp>

namespace popart {
namespace _internal {
namespace ir {

void bindManualCreateOpFunctionToGraphClass(py::class_<Graph> g) {
  // CallOp
  g.def("createOp_CallOp",
        py::overload_cast<const popart::OperatorIdentifier &,
                          popart::Graph &,
                          const Op::Settings &>(
            &Graph::createOp<CallOp,
                             const popart::OperatorIdentifier &,
                             popart::Graph &,
                             const Op::Settings &>),
        py::arg("opid"),
        py::arg("callee"),
        py::arg("settings"),
        py::return_value_policy::reference)
      .def("createOp_CallOp",
           py::overload_cast<const popart::OperatorIdentifier &,
                             popart::Graph &,
                             const std::vector<int> &,
                             const Op::Settings &>(
               &Graph::createOp<CallOp,
                                const popart::OperatorIdentifier &,
                                popart::Graph &,
                                const std::vector<int> &,
                                const Op::Settings &>),
           py::arg("opid"),
           py::arg("callee"),
           py::arg("modifiedInputsViaAttrs"),
           py::arg("settings"),
           py::return_value_policy::reference);

  // MatMulOp
  g.def("createOp_MatMulOp",
        &Graph::createOp<MatMulOp,
                         const OperatorIdentifier &,
                         const Op::Settings &,
                         const nonstd::optional<float> &,
                         const MatMulOp::SerialiseSettings &,
                         const OptionalDataType &,
                         const MatMulPartialsType &>,
        py::arg("opid"),
        py::arg("settings"),
        py::arg("availableMemoryProportion"),
        py::arg("serialization"),
        py::arg("outputType"),
        py::arg("partialsType"),
        py::return_value_policy::reference);

  // IpuCopyOp
  g.def("createOp_IpuCopyOp",
        &Graph::createOp<IpuCopyOp,
                         const OperatorIdentifier &,
                         uint64_t, /* destination */
                         const Op::Settings &>,
        py::arg("opid"),
        py::arg("destIpu"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // AccumulateBaseOp
  g.def(
      "createOp_AccumulateBaseOp",
      [](Graph &self,
         OperatorIdentifier opid,
         AccumulationType type,
         OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<AccumulateBaseOp>>(
            opid, type, factor, settings);
      },
      py::arg("opid"),
      py::arg("accumulationType"),
      py::arg("optimizer_value"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulateOp
  g.def(
      "createOp_AccumulateOp",
      [](Graph &self,
         AccumulationType type,
         OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createOp<AccumulateOp>(type, factor, settings);
      },
      py::arg("accumulationType"),
      py::arg("factor"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // VarUpdateOp
  g.def(
      "createOp_VarUpdateOp",
      [](Graph &self, OperatorIdentifier opid, const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<>>(opid, settings);
      },
      py::arg("opid"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // VarUpdateWithUpdaterOp
  g.def(
      "createOp_VarUpdateWithUpdaterOp",
      [](Graph &self, OperatorIdentifier opid, const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<VarUpdateWithUpdaterOp>>(
            opid, settings);
      },
      py::arg("opid"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // SparseAccumulateOp
  g.def(
      "createOp_SparseAccumulateOp",
      [](Graph &self,
         AccumulationType accumulationType,
         const OptimizerValue &optimizer_value,
         unsigned axis,
         const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<SparseAccumulateOp>>(
            accumulationType, optimizer_value, axis, settings);
      },
      py::arg("accumulationType"),
      py::arg("optimizer_value"),
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulatorScaleOp
  g.def(
      "createOp_AccumulatorScaleOp",
      [](Graph &self,
         const OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<AccumulatorScaleOp>>(factor,
                                                                    settings);
      },
      py::arg("factor"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulatorZeroOp
  g.def(
      "createOp_AccumulatorZeroOp",
      [](Graph &self, const Op::Settings &settings) {
        return self.createOp<op::PyVarUpdateOp<AccumulatorZeroOp>>(settings);
      },
      py::arg("settings"),
      py::return_value_policy::reference);
}

void bindManualCreateConnectedOpFunctionToGraphClass(py::class_<Graph> g) {

  // CallOp
  g.def("createConnectedOp_CallOp",
        py::overload_cast<const std::map<InIndex, TensorId> &,
                          const std::map<OutIndex, TensorId> &,
                          const popart::OperatorIdentifier &,
                          popart::Graph &,
                          const Op::Settings &>(
            &Graph::createConnectedOp<CallOp,
                                      const popart::OperatorIdentifier &,
                                      popart::Graph &,
                                      const Op::Settings &>),
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("callee"),
        py::arg("settings"),
        py::return_value_policy::reference)
      .def("createConnectedOp_CallOp",
           py::overload_cast<const std::map<InIndex, TensorId> &,
                             const std::map<OutIndex, TensorId> &,
                             const popart::OperatorIdentifier &,
                             popart::Graph &,
                             const std::vector<int> &,
                             const Op::Settings &>(
               &Graph::createConnectedOp<CallOp,
                                         const popart::OperatorIdentifier &,
                                         popart::Graph &,
                                         const std::vector<int> &,
                                         const Op::Settings &>),
           py::arg("in"),
           py::arg("out"),
           py::arg("opid"),
           py::arg("callee"),
           py::arg("modifiedInputsViaAttrs"),
           py::arg("settings"),
           py::return_value_policy::reference);

  // MatMulOp
  g.def("createConnectedOp_MatMulOp",
        &Graph::createConnectedOp<MatMulOp,
                                  const OperatorIdentifier &,
                                  const Op::Settings &,
                                  const nonstd::optional<float> &,
                                  const MatMulOp::SerialiseSettings &,
                                  const OptionalDataType &,
                                  const MatMulPartialsType &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("availableMemoryProportion"),
        py::arg("serialization"),
        py::arg("outputType"),
        py::arg("partialsType"),
        py::return_value_policy::reference);

  // IpuCopyOp
  g.def(
      "createConnectedOp_IpuCopyOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         const OperatorIdentifier &_opid,
         uint64_t sourceIpu,
         uint64_t destIpu,
         const Op::Settings &settings_) {
        IpuCopyOp *op = self.createOp<IpuCopyOp>(_opid, destIpu, settings_);

        for (auto &input : in) {
          // connectInTensor(InIndex, TensorId) is marked private for IpuCopyOp,
          // so we must use the public version with sourceIpu. This is why we
          // can't use the template for createConnectedOp for IpuCopyOp.
          // At the time of constructing these ops, we don't know the ipu number
          // for the input, so we make the user specify this.
          op->connectInTensor(input.first, input.second, sourceIpu);
        }
        for (auto &output : out) {
          if (self.getTensors().contains(output.second)) {
            Tensor *t = self.getTensors().get(output.second);
            if (t->hasProducer()) {
              t->getProducer()->disconnectOutTensor(t);
            }
            op->connectOutTensor(output.first, output.second);
          } else {
            op->createAndConnectOutTensor(output.first, output.second);
          }
        }

        op->setup();

        return op;
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("sourceIpu"),
      py::arg("destIpu"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulateBaseOp
  g.def(
      "createConnectedOp_AccumulateBaseOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         OperatorIdentifier opid,
         AccumulationType type,
         OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createConnectedOp<op::PyVarUpdateOp<AccumulateBaseOp>>(
            in, out, opid, type, factor, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("accumulationType"),
      py::arg("optimizer_value"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulateOp
  g.def(
      "createConnectedOp_AccumulateOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         AccumulationType type,
         OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createConnectedOp<AccumulateOp>(
            in, out, type, factor, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("accumulationType"),
      py::arg("factor"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // VarUpdateOp
  g.def(
      "createConnectedOp_VarUpdateOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         OperatorIdentifier opid,
         const Op::Settings &settings) {
        return self.createConnectedOp<op::PyVarUpdateOp<>>(
            in, out, opid, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // VarUpdateWithUpdaterOp
  g.def(
      "createConnectedOp_VarUpdateWithUpdaterOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         OperatorIdentifier opid,
         const Op::Settings &settings) {
        return self
            .createConnectedOp<op::PyVarUpdateOp<VarUpdateWithUpdaterOp>>(
                in, out, opid, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // SparseAccumulateOp
  g.def(
      "createConnectedOp_SparseAccumulateOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         AccumulationType accumulationType,
         const OptimizerValue &optimizer_value,
         unsigned axis,
         const Op::Settings &settings) {
        return self.createConnectedOp<op::PyVarUpdateOp<SparseAccumulateOp>>(
            in, out, accumulationType, optimizer_value, axis, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("accumulationType"),
      py::arg("optimizer_value"),
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulatorScaleOp
  g.def(
      "createConnectedOp_AccumulatorScaleOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         const OptimizerValue factor,
         const Op::Settings &settings) {
        return self.createConnectedOp<op::PyVarUpdateOp<AccumulatorScaleOp>>(
            in, out, factor, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("factor"),
      py::arg("settings"),
      py::return_value_policy::reference);

  // AccumulatorZeroOp
  g.def(
      "createConnectedOp_AccumulatorZeroOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,

         const Op::Settings &settings) {
        return self.createConnectedOp<op::PyVarUpdateOp<AccumulatorZeroOp>>(
            in, out, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("settings"),
      py::return_value_policy::reference);
}

} // namespace ir
} // namespace _internal
} // namespace popart