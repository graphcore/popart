// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <utility>
#include <vector>
#include <popart/op/accumulate.hpp>
#include <popart/op/accumulatorscale.hpp>
#include <popart/op/accumulatorzero.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/argmax.hpp>
#include <popart/op/argmin.hpp>
#include <popart/op/averagepool.hpp>
#include <popart/op/call.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/conv.hpp>              // IWYU pragma: keep
#include <popart/op/exchange/codecopy.hpp> // IWYU pragma: keep
#include <popart/op/ipucopy.hpp>           // IWYU pragma: keep
#include <popart/op/loop.hpp>              // IWYU pragma: keep
#include <popart/op/matmul.hpp>            // IWYU pragma: keep
#include <popart/op/maxpool.hpp>           // IWYU pragma: keep
#include <popart/op/resize.hpp>
#include <popart/op/roialign.hpp> // IWYU pragma: keep

#include "bindings/basicoptionals.hpp"
#include "bindings/op/argminmax.hpp"
#include "bindings/op/manualbindops.hpp"
#include "bindings/op/matmul.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/op/pool.hpp"
#include "bindings/op/resize.hpp"
#include "bindings/op/roialign.hpp"
#include "bindings/op/varupdate.hpp"
#include "popart/adam.hpp"
#include "popart/alias/aliasmodel.hpp" // IWYU pragma: keep
#include "popart/datatype.hpp"
#include "popart/graph.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/receptive.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class MultiConvOptions;
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

  // RoiAlignOp
  g.def("createOp_RoiAlignOp",
        &Graph::createOp<RoiAlignOp,
                         const OperatorIdentifier &,
                         const Op::Settings &,
                         const float,
                         const uint64_t,
                         const uint64_t,
                         const uint64_t>,
        py::arg("opid"),
        py::arg("settings"),
        py::arg("spatialScale"),
        py::arg("samplingRatio"),
        py::arg("alignedHeight"),
        py::arg("alignedWidth"),
        py::return_value_policy::reference);

  // ConvOp
  g.def("createOp_ConvOp",
        &Graph::createOp<ConvOp,
                         const OperatorIdentifier &,
                         const Op::Settings &,
                         std::vector<int64_t> &,
                         std::vector<int64_t> &,
                         std::vector<int64_t> &,
                         int64_t,
                         const AutoPad &,
                         const MultiConvOptions &>,
        py::arg("opid"),
        py::arg("settings"),
        py::arg("strides"),
        py::arg("pads"),
        py::arg("dilations"),
        py::arg("group"),
        py::arg("padType"),
        py::arg("convOpts"),
        py::return_value_policy::reference);

  // AveragePool
  g.def("createOp_AveragePoolOp",
        &Graph::createOp<AveragePoolOp,
                         const OperatorIdentifier &,
                         int64_t,
                         const std::vector<int64_t> &,
                         const HasReceptiveFieldOp::ReceptiveOpAttributes &,
                         const Op::Settings &>,
        py::arg("opid"),
        py::arg("countIncludePad"),
        py::arg("kernelShape"),
        py::arg("attributes"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // MaxPool
  g.def("createOp_MaxPoolOp",
        &Graph::createOp<MaxPoolOp,
                         const OperatorIdentifier &,
                         const std::vector<int64_t> &,
                         int64_t,
                         const HasReceptiveFieldOp::ReceptiveOpAttributes &,
                         const Op::Settings &>,
        py::arg("opid"),
        py::arg("kernelShape"),
        py::arg("storageOrder"),
        py::arg("attributes"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // ArgMax
  g.def("createOp_ArgMaxOp",
        &Graph::createOp<ArgMaxOp,
                         const OperatorIdentifier &,
                         int64_t,
                         int64_t,
                         const Op::Settings &>,
        py::arg("opid"),
        py::arg("axis"),
        py::arg("keepdims"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // ArgMin
  g.def("createOp_ArgMinOp",
        &Graph::createOp<ArgMinOp,
                         const OperatorIdentifier &,
                         int64_t,
                         int64_t,
                         const Op::Settings &>,
        py::arg("opid"),
        py::arg("axis"),
        py::arg("keepdims"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // Resize
  g.def("createOp_ResizeOp",
        py::overload_cast<const OperatorIdentifier &,
                          const Op::Settings &,
                          ResizeMode &,
                          const std::vector<float> &>(
            &Graph::createOp<ResizeOp,
                             const OperatorIdentifier &,
                             const Op::Settings &,
                             ResizeMode &,
                             const std::vector<float> &>),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("mode"),
        py::arg("scales"),
        py::return_value_policy::reference)
      .def("createOp_ResizeOp",
           py::overload_cast<const OperatorIdentifier &,
                             const Op::Settings &,
                             ResizeMode &,
                             const std::vector<float> &,
                             ResizeNearestMode &,
                             ResizeCoordinateTransformationMode &>(
               &Graph::createOp<ResizeOp,
                                const OperatorIdentifier &,
                                const Op::Settings &,
                                ResizeMode &,
                                const std::vector<float> &,
                                ResizeNearestMode &,
                                ResizeCoordinateTransformationMode &>),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("mode"),
           py::arg("scales"),
           py::arg("nearestMode"),
           py::arg("coordinateTransformationMode"),
           py::return_value_policy::reference);

  // LoopOp
  g.def("createOp_LoopOp",
        py::overload_cast<const popart::OperatorIdentifier &,
                          const Op::Settings &,
                          popart::Graph &>(
            &Graph::createOp<LoopOp,
                             const popart::OperatorIdentifier &,
                             const Op::Settings &,
                             popart::Graph &>),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("callee_"),
        py::return_value_policy::reference);
  g.def("createOp_LoopOp",
        py::overload_cast<const popart::OperatorIdentifier &,
                          const Op::Settings &,
                          popart::Graph &,
                          int &>(
            &Graph::createOp<LoopOp,
                             const popart::OperatorIdentifier &,
                             const Op::Settings &,
                             popart::Graph &,
                             int &>),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("callee_"),
        py::arg("numImplicitScanOutputs_"),
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
  // AdamUpdaterOp
  g.def(
      "createOp_AdamUpdaterOp",
      [](Graph &self,
         AdamMode mode_,
         OptimizerValue wd,
         OptimizerValue b1,
         OptimizerValue b2,
         OptimizerValue eps,
         const Op::Settings &settings) {
        return self.createOp<AdamUpdaterOp>(mode_, wd, b1, b2, eps, settings);
      },
      py::arg("mode_"),
      py::arg("wd"),
      py::arg("b1"),
      py::arg("b2"),
      py::arg("eps"),
      py::arg("settings"),
      py::return_value_policy::reference);
  // ConcatOp
  g.def(
      "createOp_ConcatOp",
      [](Graph &self,
         const popart::OperatorIdentifier &opid,
         int64_t axis,
         const Op::Settings &settings) {
        return self.createOp<ConcatOp>(opid, axis, settings);
      },
      py::arg("opid"),
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);
  g.def(
      "createOp_ConcatInplaceOp",
      [](Graph &self, int64_t axis, const Op::Settings &settings) {
        return self.createOp<ConcatInplaceOp>(axis, settings);
      },
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);
  g.def(
      "createOp_ExternalCodeCopyOp",
      [](Graph &self,
         const popart::OperatorIdentifier opid,
         const GraphId &gid,
         const CodeMemoryType destinationType_,
         const Op::Settings &settings) {
        return self.createOp<ExternalCodeCopyOp>(
            opid, gid, destinationType_, settings);
      },
      py::arg("opid"),
      py::arg("graphid"),
      py::arg("destinationType"),
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

  // RoiAlignOp
  g.def("createConnectedOp_RoiAlignOp",
        &Graph::createConnectedOp<RoiAlignOp,
                                  const OperatorIdentifier &,
                                  const Op::Settings &,
                                  const float,
                                  const uint64_t,
                                  const uint64_t,
                                  const uint64_t>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("spatialScale"),
        py::arg("samplingRatio"),
        py::arg("alignedHeight"),
        py::arg("alignedWidth"),
        py::return_value_policy::reference);

  // ConvOp
  g.def("createConnectedOp_ConvOp",
        &Graph::createConnectedOp<ConvOp,
                                  const OperatorIdentifier &,
                                  const Op::Settings &,
                                  std::vector<int64_t> &,
                                  std::vector<int64_t> &,
                                  std::vector<int64_t> &,
                                  int64_t,
                                  const AutoPad &,
                                  const MultiConvOptions &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("strides"),
        py::arg("pads"),
        py::arg("dilations"),
        py::arg("group"),
        py::arg("padType"),
        py::arg("convOpts"),
        py::return_value_policy::reference);

  // AveragePool
  g.def("createConnectedOp_AveragePoolOp",
        &Graph::createConnectedOp<
            AveragePoolOp,
            const OperatorIdentifier &,
            int64_t,
            const std::vector<int64_t> &,
            const HasReceptiveFieldOp::ReceptiveOpAttributes &,
            const Op::Settings &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("countIncludePad"),
        py::arg("kernelShape"),
        py::arg("attributes"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // MaxPool
  g.def("createConnectedOp_MaxPoolOp",
        &Graph::createConnectedOp<
            MaxPoolOp,
            const OperatorIdentifier &,
            const std::vector<int64_t> &,
            int64_t,
            const HasReceptiveFieldOp::ReceptiveOpAttributes &,
            const Op::Settings &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("kernelShape"),
        py::arg("storageOrder"),
        py::arg("attributes"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // ArgMax
  g.def("createConnectedOp_ArgMaxOp",
        &Graph::createConnectedOp<ArgMaxOp,
                                  const OperatorIdentifier &,
                                  int64_t,
                                  int64_t,
                                  const Op::Settings &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("axis"),
        py::arg("keepdims"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // ArgMin
  g.def("createConnectedOp_ArgMinOp",
        &Graph::createConnectedOp<ArgMinOp,
                                  const OperatorIdentifier &,
                                  int64_t,
                                  int64_t,
                                  const Op::Settings &>,
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("axis"),
        py::arg("keepdims"),
        py::arg("settings"),
        py::return_value_policy::reference);

  // Resize
  g.def("createConnectedOp_ResizeOp",
        py::overload_cast<const std::map<InIndex, TensorId> &,
                          const std::map<OutIndex, TensorId> &,
                          const OperatorIdentifier &,
                          const Op::Settings &,
                          ResizeMode &,
                          const std::vector<float> &>(
            &Graph::createConnectedOp<ResizeOp,
                                      const OperatorIdentifier &,
                                      const Op::Settings &,
                                      ResizeMode &,
                                      const std::vector<float> &>),
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("mode"),
        py::arg("scales"),
        py::return_value_policy::reference)
      .def("createConnectedOp_ResizeOp",
           py::overload_cast<const std::map<InIndex, TensorId> &,
                             const std::map<OutIndex, TensorId> &,
                             const OperatorIdentifier &,
                             const Op::Settings &,
                             ResizeMode &,
                             const std::vector<float> &,
                             ResizeNearestMode &,
                             ResizeCoordinateTransformationMode &>(
               &Graph::createConnectedOp<ResizeOp,
                                         const OperatorIdentifier &,
                                         const Op::Settings &,
                                         ResizeMode &,
                                         const std::vector<float> &,
                                         ResizeNearestMode &,
                                         ResizeCoordinateTransformationMode &>),
           py::arg("in"),
           py::arg("out"),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("mode"),
           py::arg("scales"),
           py::arg("nearestMode"),
           py::arg("coordinateTransformationMode"),
           py::return_value_policy::reference);

  // LoopOp
  g.def("createConnectedOp_LoopOp",
        py::overload_cast<const std::map<InIndex, TensorId> &,
                          const std::map<OutIndex, TensorId> &,
                          const popart::OperatorIdentifier &,
                          const Op::Settings &,
                          popart::Graph &>(
            &Graph::createConnectedOp<LoopOp,
                                      const popart::OperatorIdentifier &,
                                      const Op::Settings &,
                                      popart::Graph &>),
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("callee_"),
        py::return_value_policy::reference);
  g.def("createConnectedOp_LoopOp",
        py::overload_cast<const std::map<InIndex, TensorId> &,
                          const std::map<OutIndex, TensorId> &,
                          const popart::OperatorIdentifier &,
                          const Op::Settings &,
                          popart::Graph &,
                          int &>(
            &Graph::createConnectedOp<LoopOp,
                                      const popart::OperatorIdentifier &,
                                      const Op::Settings &,
                                      popart::Graph &,
                                      int &>),
        py::arg("in"),
        py::arg("out"),
        py::arg("opid"),
        py::arg("settings"),
        py::arg("callee_"),
        py::arg("numImplicitScanOutputs_"),
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

  // AdamUpdaterOp
  g.def(
      "createConnectedOp_AdamUpdaterOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         AdamMode mode_,
         OptimizerValue wd,
         OptimizerValue b1,
         OptimizerValue b2,
         OptimizerValue eps,
         const Op::Settings &settings) {
        return self.createConnectedOp<AdamUpdaterOp>(
            in, out, mode_, wd, b1, b2, eps, settings);
      },

      py::arg("in"),
      py::arg("out"),
      py::arg("mode_"),
      py::arg("wd"),
      py::arg("b1"),
      py::arg("b2"),
      py::arg("eps"),
      py::arg("settings"),
      py::return_value_policy::reference);
  // ConcatOp
  g.def(
      "createConnectedOp_ConcatOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         const popart::OperatorIdentifier &opid,
         int64_t axis,
         const Op::Settings &settings) {
        return self.createConnectedOp<ConcatOp>(in, out, opid, axis, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);
  g.def(
      "createConnectedOp_ConcatInplaceOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         int64_t axis,
         const Op::Settings &settings) {
        return self.createConnectedOp<ConcatInplaceOp>(in, out, axis, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("axis"),
      py::arg("settings"),
      py::return_value_policy::reference);
  g.def(
      "createConnectedOp_ExternalCodeCopyOp",
      [](Graph &self,
         const std::map<InIndex, TensorId> &in,
         const std::map<OutIndex, TensorId> &out,
         const OperatorIdentifier &opid,
         const GraphId &gid,
         const CodeMemoryType destinationType_,
         const Op::Settings &settings) {
        return self.createConnectedOp<ExternalCodeCopyOp>(
            in, out, opid, gid, destinationType_, settings);
      },
      py::arg("in"),
      py::arg("out"),
      py::arg("opid"),
      py::arg("graphid"),
      py::arg("destinationType"),
      py::arg("settings"),
      py::return_value_policy::reference);
}

} // namespace ir
} // namespace _internal
} // namespace popart
