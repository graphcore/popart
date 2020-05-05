// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <pybind11/functional.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/np_utils.hpp>
#include <popart/numerics.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/loss.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/pyarray_accessor.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio_generic.hpp>
#include <popart/stepio_size_assertion.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/version.hpp>

#include <popart/popx/devicex.hpp>

#include <stdexcept>
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <onnx/onnx_pb.h>

namespace py = pybind11;
using namespace popart;

// The following code attempts to convert the python dictionary
// (py::dict) into a map of strings for keys and values. The default
// pybind will attempt to match types
// TODO : This is not very elegant code is there a better way to do
// this?
std::map<std::string, std::string> getDictionary(py::dict pydict) {

  std::map<std::string, std::string> dictionary;
  for (auto element : pydict) {
    std::stringstream key;
    key << element.first;

    std::stringstream value;
    value << element.second;

    dictionary.insert(std::make_pair(key.str(), value.str()));
  }
  return dictionary;
}

std::map<std::string, std::pair<float, bool>>
getOptimizerValueDictionary(py::dict e) {
  std::map<std::string, std::pair<float, bool>> cpm;
  for (auto element : e) {
    if (!py::isinstance<py::str>(element.first)) {
      throw error("A key in the optimizer map input must be a py::str (in "
                  "getOptimizerValueDictionary)");
    }
    auto key = py::str(element.first);
    if (!py::isinstance<py::tuple>(element.second)) {
      throw error("A value in the optimizer map must be a py::tuple (in "
                  "getOptimizerValueDictionary)");
    }
    std::pair<float, bool> p = element.second.cast<std::pair<float, bool>>();
    cpm.insert({key, p});
  }
  return cpm;
}

std::map<std::string, boost::any> getDictionaryVar(py::dict pydict) {
  // This attempts to convert the py::dict to a map of string, boost::any. Since
  // we do not know the python types given by the user until runtime, we have to
  // account for each type. See attributes.hpp for a description of possible
  // attribute types.

  std::map<std::string, boost::any> dictionary;
  for (auto element : pydict) {
    auto key = py::str(element.first);
    auto val = element.second;
    if (py::isinstance<py::str>(val)) {
      // String
      dictionary.insert(std::make_pair(key, val.cast<std::string>()));
    } else if (py::isinstance<py::int_>(val)) {
      // Int
      dictionary.insert(std::make_pair(key, val.cast<int64_t>()));
    } else if (py::isinstance<py::list>(val)) {
      // Ints
      std::vector<int64_t> vec;
      for (auto subval : val) {
        vec.push_back(subval.cast<int64_t>());
      }
      dictionary.insert(std::make_pair(key, vec));
    } else if (py::isinstance<py::float_>(val)) {
      // Float
      dictionary.insert(std::make_pair(key, val.cast<float>()));
    } else {
      throw error("Invalid type provided in custom op attribute '{}'", key);
    }
  }
  return dictionary;
}

class PyStepIO
    : public StepIOGeneric<py::array, StepIONS::PyArrayAccessor, py::array> {
public:
  PyStepIO(std::map<TensorId, py::array> inputs,
           std::map<TensorId, py::array> outputs) {
    for (auto p : inputs) {
      inputsInfo.insert({p.first, {p.second, 0}});
    }

    for (auto p : outputs) {
      outputsInfo.insert({p.first, {p.second, 0}});
    }
  }
};

class PyStepIOCallback : public IStepIO {
public:
  using InputCallback          = std::function<py::array(std::string, bool)>;
  using InputCompleteCallback  = std::function<void(std::string)>;
  using OutputCallback         = std::function<py::array(std::string)>;
  using OutputCompleteCallback = std::function<void(std::string)>;

  // inputCb The call back to get input data
  // inputCompleteCb_ The call back to indicate that input had been consumed
  // outputCb_ The call back to get out data
  // outputCompleteCb_ The call back to indicate that output had been written
  PyStepIOCallback(InputCallback inputCb_,
                   InputCompleteCallback inputCompleteCb_,
                   OutputCallback outputCb_,
                   OutputCompleteCallback outputCompleteCb_)
      : inputCb(inputCb_), inputCompleteCb(inputCompleteCb_),
        outputCb(outputCb_), outputCompleteCb(outputCompleteCb_) {}

  void assertNumElements(const Ir &) const final {}

  ConstVoidData in(TensorId id, int64_t, bool prefetch)final {
    py::array a = inputCb(id, prefetch);

    ConstVoidData data;

    // If a None object has been returned ndim will be 0
    if (a.ndim() > 0) {
      data.data = a.request().ptr;
      data.info = getTensorInfo(a);
    }

    return data;
  }

  void inComplete(TensorId id, int64_t) final { inputCompleteCb(id); }

  MutableVoidData out(TensorId id, int64_t) final {
    py::array a = outputCb(id);

    MutableVoidData data;
    data.data = a.request().ptr;
    data.info = getTensorInfo(a);
    return data;
  }

  void outComplete(TensorId id) final { outputCompleteCb(id); }

private:
  // user land callbacks
  InputCallback inputCb;
  InputCompleteCallback inputCompleteCb;
  OutputCallback outputCb;
  OutputCompleteCallback outputCompleteCb;
};

class PyWeightsIO : public IWeightsIO {
public:
  PyWeightsIO(std::map<TensorId, py::array> weights_) : weights(weights_) {}

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, py::array> &M,
        std::string mapName) const {
    auto found = M.find(id);
    if (found == M.end()) {
      throw error("No tensor {} provided in PyWeightsIO's {}", id, mapName);
    }
    py::array npArr = found->second;
    T stepData;
    stepData.data = npArr.request().ptr;
    stepData.info = getTensorInfo(npArr);
    return stepData;
  }

  bool contains(TensorId id) const final {
    return weights.find(id) != weights.end();
  }

  MutableVoidData weight(TensorId id) const final {
    return get<MutableVoidData>(id, weights, "weights");
  }

private:
  std::map<TensorId, py::array> weights;
};

class AttributeContextManager {
  Builder &builder;
  std::string attribute;
  boost::any value;
  std::vector<boost::any> prevValue;

public:
  AttributeContextManager(Builder &_builder,
                          const std::string &_attribute,
                          boost::any value_)
      : builder(_builder), attribute(_attribute), value(value_) {}

  void enter() {
    if (builder.hasAttribute(attribute)) {
      // Backup previous attribute value
      prevValue.push_back(
          boost::any_cast<int64_t>(builder.getAttribute(attribute)));
      builder.clearAttribute(attribute);
    }
    builder.setAttribute(attribute, value);
  }
  void exit() {
    builder.clearAttribute(attribute);
    if (prevValue.size() > 0) {
      // Restore previous attribute value
      builder.setAttribute(attribute, prevValue.back());
      prevValue.pop_back();
    }
  }
};

struct PrepareDeviceError {
  bool success = true;
  std::unique_ptr<popart::memory_allocation_err> exception;

  virtual ~PrepareDeviceError() {}

  virtual bool isSuccessful() const { return success; }
  std::string what() const { return exception->what(); }
  std::string getSummaryReport() const { return exception->getSummaryReport(); }
  std::string getGraphReport(bool useCbor) const {
    return exception->getGraphReport(useCbor);
  }
};

class NameContextManager {
  Builder &builder;
  std::string name;

public:
  NameContextManager(Builder &_builder, const std::string &_name)
      : builder(_builder), name(_name) {}

  void enter() { builder.pushNameScope(name); }
  void exit() { builder.popNameScope(); }
};

// Create a logging interface for popart that is similar to python logging
// module

class Logger {

  std::string name;

  Logger(const std::string &name_) : name(name_) {}

public:
  static Logger getLogger(const std::string &name = "all") {
    return Logger(name);
  }

  void setLevel(const std::string &level) {
    logging::configure({{name, level}});
  }

  void debug(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Debug, std::move(info));
  }

  void info(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Info, std::move(info));
  }

  void warn(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Warn, std::move(info));
  }

  void error(const std::string &info) {
    logging::log(logging::Module::python, logging::Level::Err, std::move(info));
  }

  void critical(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Critical, std::move(info));
  }
};

// The following code allow boost optional to be used in the C++ interface and
// map to python types
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(popart_core, m) {
  m.doc() = "binding for C++ popart library";

  m.def("getTensorInfo", &getTensorInfo);

  m.def("getLogger", &Logger::getLogger, py::arg("name") = "all");

  m.def("versionString", &popart::core::versionString);
  m.def("packageHash", &popart::core::packageHash);

  {
    py::class_<Logger> cls(m, "Logger");
    cls.def("setLevel", &Logger::setLevel);
    cls.def("debug", &Logger::debug);
    cls.def("info", &Logger::info);
    cls.def("warn", &Logger::warn);
    cls.def("error", &Logger::error);
    cls.def("critical", &Logger::critical);
  }
  {
    py::class_<OperatorIdentifier> cls(m, "OperatorIdentifier");
    cls.def(py::init<const std::string &, const std::string &, unsigned>(),
            py::arg("domain"),
            py::arg("type"),
            py::arg("version"));
    cls.def_readonly("domain", &OperatorIdentifier::domain);
    cls.def_readonly("type", &OperatorIdentifier::type);
    cls.def_readonly("version", &OperatorIdentifier::version);
  }
  m.def("getSupportedOperations",
        &OpManager::getSupportedOperations,
        py::arg("includeInternal"));
  {
    py::enum_<DataType> en(m, "DataType");
    en.value("UINT8", DataType::UINT8);
    en.value("INT8", DataType::INT8);
    en.value("UINT16", DataType::UINT16);
    en.value("INT16", DataType::INT16);
    en.value("INT32", DataType::INT32);
    en.value("INT64", DataType::INT64);
    en.value("UINT32", DataType::UINT32);
    en.value("UINT64", DataType::UINT64);
    en.value("BOOL", DataType::BOOL);
    en.value("FLOAT", DataType::FLOAT);
    en.value("FLOAT16", DataType::FLOAT16);
    en.value("BFLOAT16", DataType::BFLOAT16);
    en.value("DOUBLE", DataType::DOUBLE);
    en.value("COMPLEX64", DataType::COMPLEX64);
    en.value("COMPLEX128", DataType::COMPLEX128);
    en.value("STRING", DataType::STRING);
    en.value("UNDEFINED", DataType::UNDEFINED);
  }
  {
    py::enum_<InitType> en(m, "InitType");
    en.value("NONE", InitType::NONE);
    en.value("ZERO", InitType::ZERO);
  }
  {
    py::class_<OpDefinition::Input> cls(m, "OpDefinition::Input");
    cls.def_readonly("name", &OpDefinition::Input::name);
    cls.def_readonly("supportedTensors",
                     &OpDefinition::Input::supportedTensors);
    cls.def_readonly("constant", &OpDefinition::Input::constant);
  }
  {
    py::class_<OpDefinition::Output> cls(m, "OpDefinition::Output");
    cls.def_readonly("name", &OpDefinition::Output::name);
    cls.def_readonly("supportedTensors",
                     &OpDefinition::Output::supportedTensors);
  }
  {
    py::class_<OpDefinition::Attribute> cls(m, "OpDefinition::Attribute");
    cls.def_readonly("supportedValuesRegex",
                     &OpDefinition::Attribute::supportedValuesRegex);
  }
  {
    py::class_<OpDefinition> cls(m, "OpDefinition");
    cls.def_readonly("inputs", &OpDefinition::inputs);
    cls.def_readonly("outputs", &OpDefinition::outputs);
    cls.def_readonly("attributes", &OpDefinition::attributes);
  }

  m.def("getSupportedOperationsDefinition",
        &OpManager::getSupportedOperationsDefinition,
        py::arg("includeInternal"));

  {
    py::class_<IStepIO> stepio(m, "IStepIO");
    py::class_<IWeightsIO> weightsio(m, "IWeightsIO");

    py::enum_<AnchorReturnTypeId> en(m, "AnchorReturnTypeId");
    en.value("FINAL", AnchorReturnTypeId::FINAL);
    en.value("EVERYN", AnchorReturnTypeId::EVERYN);
    en.value("ALL", AnchorReturnTypeId::ALL);
    en.value("SUM", AnchorReturnTypeId::SUM);

    {
      py::class_<PyStepIO> cls(m, "PyStepIO", stepio);
      cls.def(py::init<std::map<TensorId, py::array>,
                       std::map<TensorId, py::array>>(),
              py::arg("inputs"),
              py::arg("outputs"));
      cls.def("enableRuntimeAsserts", &PyStepIO::enableRuntimeAsserts);
    }
    {
      py::class_<PyStepIOCallback> cls(m, "PyStepIOCallback", stepio);
      cls.def(py::init<std::function<py::array(std::string, bool)>,
                       std::function<void(std::string)>,
                       std::function<py::array(std::string)>,
                       std::function<void(std::string)>>(),
              py::arg("input_callback"),
              py::arg("input_complete_callback"),
              py::arg("output_callback"),
              py::arg("output_complete_callback"));
    }
    {
      py::class_<PyWeightsIO> cls(m, "PyWeightsIO", weightsio);
      cls.def(py::init<std::map<TensorId, py::array>>(), py::arg("weights"));
    }
  }
  {
    py::class_<AnchorReturnType> cls(m, "AnchorReturnType");
    cls.def(py::init<std::string>(), py::arg("anchorReturnTypeString"));
    cls.def(py::init<std::string, int>(),
            py::arg("anchorReturnTypeString"),
            py::arg("returnPeriod"));
    cls.def("id", &AnchorReturnType::id);
    cls.def("rp", &AnchorReturnType::rp);
  }
  {
    py::class_<DataFlow> cls(m, "DataFlow");
    cls.def(py::init<int, const std::map<TensorId, AnchorReturnType> &>(),
            py::arg("batchesPerStep"),
            py::arg("anchorTensors"));
    cls.def(py::init<int,
                     const std::vector<TensorId> &,
                     const AnchorReturnType &>(),
            py::arg("batchesPerStep"),
            py::arg("anchorIds"),
            py::arg("anchorReturnType") = AnchorReturnType("ALL"));
    cls.def("isAnchored", &DataFlow::isAnchored);
    cls.def("nAnchors", &DataFlow::nAnchors);
    cls.def("batchesPerStep", &DataFlow::batchesPerStep);
    cls.def("anchors", &DataFlow::anchors, pybind11::return_value_policy::copy);
    cls.def("art", &DataFlow::art);
  }
  {
    py::class_<TensorInfo> cls(m, "_TensorInfoCore");
    cls.def(py::init<std::string, const std::vector<int64_t> &>(),
            py::arg("dataType"),
            py::arg("shape"));
    cls.def("data_type_lcase", &TensorInfo::data_type_lcase);
    cls.def("shape", &TensorInfo::shape);
  }
  {
    py::class_<numerics::NumericsReport> cls(m, "NumericsReport");
    cls.def(py::init<std::string, std::string, std::string, std::string>(),
            py::arg("A0"),
            py::arg("A1"),
            py::arg("B0"),
            py::arg("B1"));
    cls.def("report", &numerics::NumericsReport::report);
    cls.def("fullReport", &numerics::NumericsReport::fullReport);
    cls.def("getRelativeErrors", &numerics::NumericsReport::getRelativeErrors);
  }
  {
    py::class_<InputShapeInfo> cls(m, "InputShapeInfo");
    cls.def(py::init<>());
    cls.def("add", &InputShapeInfo::add);
    cls.def("get", &InputShapeInfo::get);
    cls.def("has", &InputShapeInfo::has);
  }
  {
    py::class_<Loss> loss(m, "Loss");
    loss.def("input", &Loss::input);
    loss.def("output", &Loss::output);

    py::enum_<ReductionType> en(m, "ReductionType");
    en.value("Sum", ReductionType::SUM);
    en.value("Mean", ReductionType::MEAN);

    {
      py::class_<NllLoss> cls(m, "NllLoss", loss);
      cls.def(py::init<TensorId, TensorId, TensorId, ReductionType>(),
              py::arg("probabilities"),
              py::arg("labels"),
              py::arg("output"),
              py::arg("reduction") = ReductionType::SUM);
      cls.def(py::init<TensorId, TensorId, TensorId, int, ReductionType>(),
              py::arg("probabilities"),
              py::arg("labels"),
              py::arg("output"),
              py::arg("ignore_index"),
              py::arg("reduction") = ReductionType::SUM);
      cls.def("probsTensorId", &NllLoss::probsTensorId);
      cls.def("labelTensorId", &NllLoss::labelTensorId);
      cls.def("pipelineStage", &NllLoss::pipelineStage);
      cls.def("virtualGraph", &NllLoss::virtualGraph);
    }

    {
      py::class_<L1Loss> cls(m, "L1Loss", loss);
      cls.def(py::init<TensorId, TensorId, float, ReductionType>(),
              py::arg("input"),
              py::arg("output"),
              py::arg("lambda"),
              py::arg("reduction") = ReductionType::SUM);
      cls.def("getInputId", &L1Loss::getInputId);
      cls.def("getLambda", &L1Loss::getLambda);
      cls.def("pipelineStage", &L1Loss::pipelineStage);
      cls.def("virtualGraph", &L1Loss::virtualGraph);
    }
    {
      py::class_<IdentityLoss> cls(m, "IdentityLoss", loss);
      cls.def(py::init<TensorId, TensorId, ReductionType>(),
              py::arg("input"),
              py::arg("output"),
              py::arg("reduction") = ReductionType::SUM);
      cls.def("getInputId", &IdentityLoss::getInputId);
      cls.def("pipelineStage", &IdentityLoss::pipelineStage);
      cls.def("virtualGraph", &IdentityLoss::virtualGraph);
    }
  }
  {
    py::class_<OptimizerValue> optimizerValue(m, "OptimizerValue");
    optimizerValue.def(
        py::init<float, bool>(), py::arg("val"), py::arg("isConst"));
    optimizerValue.def(py::init<float>(), py::arg("val"));
    optimizerValue.def(py::init<>());
    optimizerValue.def(py::init<std::pair<float, bool>>());

    optimizerValue.def("val", &OptimizerValue::val);
    optimizerValue.def("isConst", &OptimizerValue::isConst);

    py::class_<OptimizerValueMap> optimizerValueMap(m, "OptimizerValueMap");
    optimizerValueMap.def("getDefault", &OptimizerValueMap::getDefault);
  }
  {
    py::class_<Optimizer> optimizer(m, "Optimizer");
    optimizer.def("getLossScalingVal", &Optimizer::getLossScalingVal);

    {
      py::class_<SGD> sgd(m, "SGD", optimizer);
      sgd.def(py::init([](py::dict pyd) {
        auto cppm = getOptimizerValueDictionary(pyd);
        return SGD(cppm);
      }));
      sgd.def("insertSpecific", [](SGD &self, TensorId id, py::dict pyd) {
        self.insertSpecific(id, getOptimizerValueDictionary(pyd));
      });

      sgd.def("learningRates", &SGD::learningRates);
      sgd.def("weightDecays", &SGD::weightDecays);
      sgd.def("momentums", &SGD::momentums);
      sgd.def("dampenings", &SGD::dampenings);
      sgd.def("velocityScalings", &SGD::velocityScalings);

      { // This class is deprecated, and SGD should be preferred
        py::class_<ConstSGD> cls(m, "ConstSGD", sgd);
        cls.def(py::init<float, float, float>(),
                py::arg("learning_rate"),
                py::arg("weight_decay") = 0.0f,
                py::arg("loss_scaling") = 1.0f);
      }
    }
  }
  {
    py::class_<SessionOptions> cls(m, "SessionOptions");
    cls.def(py::init<>());
    cls.def_readwrite("logDir", &SessionOptions::logDir);
    cls.def_readwrite("exportPoplarComputationGraph",
                      &SessionOptions::exportPoplarComputationGraph);
    cls.def_readwrite("exportPoplarVertexGraph",
                      &SessionOptions::exportPoplarVertexGraph);
    cls.def_readwrite("ignoreData", &SessionOptions::ignoreData);
    cls.def_readwrite("syntheticDataMode", &SessionOptions::syntheticDataMode);
    cls.def_readwrite("instrumentWithHardwareCycleCounter",
                      &SessionOptions::instrumentWithHardwareCycleCounter);
    cls.def_readwrite("disableGradAccumulationTensorStreams",
                      &SessionOptions::disableGradAccumulationTensorStreams);
    cls.def_readwrite("enableOutlining", &SessionOptions::enableOutlining);
    cls.def_readwrite("enableOutliningCopyCostPruning",
                      &SessionOptions::enableOutliningCopyCostPruning);
    cls.def_readwrite("outlineThreshold", &SessionOptions::outlineThreshold);
    cls.def_readwrite("accumulationFactor",
                      &SessionOptions::accumulationFactor);
    cls.def_readwrite("enableGradientAccumulation",
                      &SessionOptions::enableGradientAccumulation);
    cls.def_readwrite("enableNonStableSoftmax",
                      &SessionOptions::enableNonStableSoftmax);
    cls.def_readwrite("enablePipelining", &SessionOptions::enablePipelining);
    cls.def_readwrite("autoRecomputation", &SessionOptions::autoRecomputation);
    cls.def_readwrite("mergeVarUpdate", &SessionOptions::mergeVarUpdate);
    cls.def_readwrite("mergeVarUpdateMemThreshold",
                      &SessionOptions::mergeVarUpdateMemThreshold);
    cls.def_readwrite("rearrangeAnchorsOnHost",
                      &SessionOptions::rearrangeAnchorsOnHost);
    cls.def_readwrite("pingPongPhases", &SessionOptions::pingPongPhases);
    cls.def_readwrite("explicitRecomputation",
                      &SessionOptions::explicitRecomputation);
    cls.def_readwrite("batchSerializationFactor",
                      &SessionOptions::batchSerializationFactor);
    cls.def_readwrite("enablePrefetchDatastreams",
                      &SessionOptions::enablePrefetchDatastreams);
    cls.def_readwrite("enableVirtualGraphs",
                      &SessionOptions::enableVirtualGraphs);
    cls.def_readwrite("autoVirtualGraph", &SessionOptions::autoVirtualGraph);
    cls.def_readwrite("virtualGraphMode", &SessionOptions::virtualGraphMode);
    cls.def_readwrite("enableReplicatedGraphs",
                      &SessionOptions::enableReplicatedGraphs);
    cls.def_readwrite("replicatedGraphCount",
                      &SessionOptions::replicatedGraphCount);
    cls.def_readwrite("compileEngine", &SessionOptions::compileEngine);
    cls.def_readwrite("_engineOptions", &SessionOptions::engineOptions);
    cls.def_readwrite("_convolutionOptions",
                      &SessionOptions::convolutionOptions);
    cls.def_readwrite("_reportOptions", &SessionOptions::reportOptions);
    cls.def_readwrite("dotOpNames", &SessionOptions::dotOpNames);
    cls.def_readwrite("separateCallOpPdfs",
                      &SessionOptions::separateCallOpPdfs);
    cls.def_readwrite("finalDotOp", &SessionOptions::finalDotOp);
    cls.def_readwrite("firstDotOp", &SessionOptions::firstDotOp);
    cls.def_readwrite("constantWeights", &SessionOptions::constantWeights);
    cls.def_readwrite("cachePath", &SessionOptions::cachePath);
    cls.def_readwrite("enableEngineCaching",
                      &SessionOptions::enableEngineCaching);
    cls.def_readwrite("enableFloatingPointChecks",
                      &SessionOptions::enableFloatingPointChecks);
    cls.def_readwrite("enableStochasticRounding",
                      &SessionOptions::enableStochasticRounding);
    cls.def_readwrite("enableFullyConnectedPass",
                      &SessionOptions::enableFullyConnectedPass);
    cls.def_readwrite("enableGroupedMatmuls",
                      &SessionOptions::enableGroupedMatmuls);
    cls.def_readwrite("enableStableNorm", &SessionOptions::enableStableNorm);
    // set in python use the python set constructor, so something like
    // mySessionOptions.dotChecks = {popart.DotCheck.FINAL}
    cls.def_readwrite("dotChecks", &SessionOptions::dotChecks);
    cls.def_readwrite("customCodelets", &SessionOptions::customCodelets);
    cls.def_readwrite("customCodeletCompileFlags",
                      &SessionOptions::customCodeletCompileFlags);
    cls.def_readwrite("hostAllReduce", &SessionOptions::hostAllReduce);
    cls.def_readwrite("hostWeightUpdate", &SessionOptions::hostWeightUpdate);
    cls.def_readwrite("hostAllReduceRemoteBuffer",
                      &SessionOptions::hostAllReduceRemoteBuffer);
    cls.def_readwrite("hostWeightUpdate", &SessionOptions::hostWeightUpdate);

    cls.def_readwrite("kahnTieBreaker", &SessionOptions::kahnTieBreaker);
    cls.def_readwrite("timeLimitScheduler",
                      &SessionOptions::timeLimitScheduler);
    cls.def_readwrite("swapLimitScheduler",
                      &SessionOptions::swapLimitScheduler);
    cls.def_readwrite("decomposeGradSum", &SessionOptions::decomposeGradSum);
    cls.def_readwrite("serializedPoprithmsAnnealGraphsDir",
                      &SessionOptions::serializedPoprithmsAnnealGraphsDir);
    cls.def_readwrite("enableDistributedReplicatedGraphs",
                      &SessionOptions::enableDistributedReplicatedGraphs);
    cls.def_readwrite("globalReplicationFactor",
                      &SessionOptions::globalReplicationFactor);
    cls.def_readwrite("globalReplicaOffset",
                      &SessionOptions::globalReplicaOffset);
    cls.def_readwrite("globalNumIpus", &SessionOptions::globalNumIpus);
    cls.def_readwrite("ipuSystemType", &SessionOptions::ipuSystemType);
    cls.def_readwrite("groupHostSync", &SessionOptions::groupHostSync);
  }
  {
    py::enum_<PatternsLevel> en(m, "PatternsLevel");
    en.value("ALL", PatternsLevel::ALL);
    en.value("DEFAULT", PatternsLevel::DEFAULT);
    en.value("NONE", PatternsLevel::NONE);
  }
  {
    py::enum_<DotCheck> en(m, "DotCheck");
    en.value("FWD0", DotCheck::FWD0);
    en.value("FWD1", DotCheck::FWD1);
    en.value("BWD0", DotCheck::BWD0);
    en.value("PREALIAS", DotCheck::PREALIAS);
    en.value("FINAL", DotCheck::FINAL);
  }
  {
    py::enum_<RecomputationType> en(m, "RecomputationType");
    en.value("NoRecompute", RecomputationType::None);
    en.value("Standard", RecomputationType::Standard);
    en.value("NormOnly", RecomputationType::NormOnly);
    en.value("Pipeline", RecomputationType::Pipeline);
  }
  {
    py::enum_<RecomputeType> en(m, "RecomputeType");
    en.value("Undefined", RecomputeType::UNDEFINED);
    en.value("Checkpoint", RecomputeType::CHECKPOINT);
    en.value("Recompute", RecomputeType::RECOMPUTE);
    en.value("Recomputed", RecomputeType::RECOMPUTED);
  }
  {
    py::enum_<CacheType> en(m, "CacheType");
    en.value("Undefined", CacheType::UNDEFINED);
    en.value("Uncached", CacheType::UNCACHED);
    en.value("Cached", CacheType::CACHED);
  }
  {
    py::enum_<SyncPattern> en(m, "SyncPattern");
    en.value("Full", SyncPattern::Full);
    en.value("SinglePipeline", SyncPattern::SinglePipeline);
    en.value("PingPong", SyncPattern::PingPong);
  }
  {
    py::enum_<MergeVarUpdateType> en(m, "MergeVarUpdateType");
    en.value("Off", MergeVarUpdateType::None);
    en.value("All", MergeVarUpdateType::All);
    en.value("AutoTight", MergeVarUpdateType::AutoTight);
    en.value("AutoLoose", MergeVarUpdateType::AutoLoose);
  }
  {
    py::enum_<VirtualGraphMode> en(m, "VirtualGraphMode");
    en.value("Off", VirtualGraphMode::Off);
    en.value("Manual", VirtualGraphMode::Manual);
    en.value("Auto", VirtualGraphMode::Auto);
    en.value("PingPong", VirtualGraphMode::PingPong);
  }
  {
    py::enum_<SyntheticDataMode> en(m, "SyntheticDataMode");
    en.value("Off", SyntheticDataMode::Off);
    en.value("Zeros", SyntheticDataMode::Zeros);
    en.value("RandomNormal", SyntheticDataMode::RandomNormal);
  }
  {
    py::enum_<IrSerializationFormat> en(m, "IrSerializationFormat");
    en.value("JSON", IrSerializationFormat::JSON);
  }
  {
    py::enum_<PreAliasPatternType> en(m, "PreAliasPatternType");
    en.value("PREUNIREPL", PreAliasPatternType::PREUNIREPL);
    en.value("POSTNREPL", PreAliasPatternType::POSTNREPL);
    en.value("SOFTMAXGRADDIRECT", PreAliasPatternType::SOFTMAXGRADDIRECT);
    en.value("NLLLWITHSOFTMAXGRADDIRECT",
             PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT);
    en.value("SPLITCONVBIAS", PreAliasPatternType::SPLITCONVBIAS);
    en.value("OPTOIDENTITY", PreAliasPatternType::OPTOIDENTITY);
    en.value("SUBTRACTARG1GRADOP", PreAliasPatternType::SUBTRACTARG1GRADOP);
    en.value("MULARGGRADOP", PreAliasPatternType::MULARGGRADOP);
    en.value("RECIPROCALGRADOP", PreAliasPatternType::RECIPROCALGRADOP);
    en.value("SINGRADOP", PreAliasPatternType::SINGRADOP);
    en.value("COSGRADOP", PreAliasPatternType::COSGRADOP);
    en.value("TANTOSINOVERCOS", PreAliasPatternType::TANTOSINOVERCOS);
    en.value("DIVARG0GRADOP", PreAliasPatternType::DIVARG0GRADOP);
    en.value("DIVARG1GRADOP", PreAliasPatternType::DIVARG1GRADOP);
    en.value("POWARG0GRADOP", PreAliasPatternType::POWARG0GRADOP);
    en.value("POWARG1GRADOP", PreAliasPatternType::POWARG1GRADOP);
    en.value("SQRTGRADOP", PreAliasPatternType::SQRTGRADOP);
    en.value("EXPGRADOP", PreAliasPatternType::EXPGRADOP);
    en.value("GEMMDECOMPOSITION", PreAliasPatternType::GEMMDECOMPOSITION);
    en.value("NEGATIVEONESCALE", PreAliasPatternType::NEGATIVEONESCALE);
    en.value("MATMULOP", PreAliasPatternType::MATMULOP);
    en.value("MATMULLHSGRADOP", PreAliasPatternType::MATMULLHSGRADOP);
    en.value("MATMULRHSGRADOP", PreAliasPatternType::MATMULRHSGRADOP);
  }
  {
    py::class_<Patterns> cls(m, "Patterns");
    cls.def(py::init<>());
    cls.def(py::init<PatternsLevel>());
    cls.def(py::init<std::vector<PreAliasPatternType>>());
    cls.def(py::init(
        [](std::vector<std::string> l) { return Patterns::create(l); }));
    cls.def_property("PreUniRepl",
                     &Patterns::isPreUniReplEnabled,
                     &Patterns::enablePreUniRepl);
    cls.def_property(
        "PostNRepl", &Patterns::isPostNReplEnabled, &Patterns::enablePostNRepl);
    cls.def_property("SoftMaxGradDirect",
                     &Patterns::isSoftMaxGradDirectEnabled,
                     &Patterns::enableSoftMaxGradDirect);
    cls.def_property("NlllWithSoftMaxGradDirect",
                     &Patterns::isNlllWithSoftMaxGradDirectEnabled,
                     &Patterns::enableNlllWithSoftMaxGradDirect);
    cls.def_property("SplitConvBias",
                     &Patterns::isSplitConvBiasEnabled,
                     &Patterns::enableSplitConvBias);
    cls.def_property("OpToIdentity",
                     &Patterns::isOpToIdentityEnabled,
                     &Patterns::enableOpToIdentity);
    cls.def_property("SubtractArg1GradOp",
                     &Patterns::isSubtractArg1GradOpEnabled,
                     &Patterns::enableSubtractArg1GradOp);
    cls.def_property("MulArgGradOp",
                     &Patterns::isMulArgGradOpEnabled,
                     &Patterns::enableMulArgGradOp);
    cls.def_property(
        "MatMulOp", &Patterns::isMatMulOpEnabled, &Patterns::enableMatMulOp);
    cls.def_property("MatMulLhsGradOp",
                     &Patterns::isMatMulLhsGradOpEnabled,
                     &Patterns::enableMatMulLhsGradOp);
    cls.def_property("MatMulRhsGradOp",
                     &Patterns::isMatMulRhsGradOpEnabled,
                     &Patterns::enableMatMulRhsGradOp);
    cls.def_property(
        "InPlace", &Patterns::isInPlaceEnabled, &Patterns::enableInPlace);
    cls.def("isPatternEnabled",
            static_cast<bool (Patterns::*)(const std::string &)>(
                &Patterns::isPatternEnabled));
    cls.def("enablePattern",
            static_cast<Patterns &(Patterns::*)(const std::string &, bool)>(
                &Patterns::enablePattern));
    cls.def("__repr__", [](const Patterns &p) {
      std::stringstream ss;
      ss << p;
      return ss.str();
    });
  }
  {
    py::class_<PrepareDeviceError> cls(m, "PrepareDeviceError");
    cls.def(py::init<>());
    cls.def("__repr__", &PrepareDeviceError::what);
    cls.def("isSuccessful", &PrepareDeviceError::isSuccessful);
    cls.def("getSummaryReport", &PrepareDeviceError::getSummaryReport);
    cls.def(
        "getGraphReport",
        [](const PrepareDeviceError &error, bool useCbor) {
          auto report = error.getGraphReport(useCbor);
          return py::bytes(report);
        },
        py::arg("useCbor") = false);
  }
  {
    py::class_<InferenceSession> cls(m, "_InferenceSessionCore");
    cls.def(py::init(&InferenceSession::createFromOnnxModel),
            py::arg("model"),
            py::arg("dataFlow").none(),
            py::arg("deviceInfo"),
            py::arg("losses"),
            py::arg("inputShapeInfo"),
            py::arg("userOptions"),
            py::arg("passes"));
    cls.def(
        "prepareDevice",
        [](InferenceSession &session, PrepareDeviceError *status) {
          try {
            session.prepareDevice();
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
              status->success   = false;
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("err").none());
    cls.def("setRandomSeed",
            &InferenceSession::setRandomSeed,
            py::arg("seedValue"));
    cls.def("getCycleCount", &InferenceSession::getCycleCount);
    cls.def("weightsFromHost", &InferenceSession::weightsFromHost);
    cls.def("writeWeights", &TrainingSession::writeWeights);
    cls.def("run", &InferenceSession::run);
    cls.def("modelToHost", &InferenceSession::modelToHost);
    cls.def("getInfo", &InferenceSession::getInfo);
    cls.def("getSummaryReport",
            &InferenceSession::getSummaryReport,
            py::arg("resetProfile") = true);
    cls.def(
        "getGraphReport",
        [](const InferenceSession &session, bool useCbor) {
          auto report = session.getGraphReport(useCbor);
          return py::bytes(report);
        },
        py::arg("useCbor") = false);
    cls.def(
        "getExecutionReport",
        [](const InferenceSession &session, bool useCbor, bool resetProfile) {
          auto report = session.getExecutionReport(useCbor, resetProfile);
          return py::bytes(report);
        },
        py::arg("useCbor")      = false,
        py::arg("resetProfile") = true);
    cls.def("getSerializedGraph", [](const InferenceSession &session) {
      auto report = session.getSerializedGraph();
      return py::bytes(report);
    });
    cls.def("getTensorTileMap", &InferenceSession::getTensorTileMap);
    cls.def("resetHostWeights",
            &InferenceSession::resetHostWeights,
            py::arg("modelProtoOrFilename"),
            py::arg("ignoreWeightsInModelWithoutCorrespondingHostWeight") =
                false);
    // Special test method to write serialise ir for analysis
    cls.def("_serializeIr", &InferenceSession::serializeIr, py::arg("format"));
  }
  {
    py::class_<TrainingSession> cls(m, "_TrainingSessionCore");
    cls.def(py::init(&TrainingSession::createFromOnnxModel),
            py::arg("model"),
            py::arg("dataFlow").none(),
            py::arg("losses"),
            py::arg("optimizer"),
            py::arg("deviceInfo"),
            py::arg("inputShapeInfo"),
            py::arg("userOptions"),
            py::arg("passes"));
    cls.def("updateOptimizer", &TrainingSession::updateOptimizer);
    cls.def(
        "prepareDevice",
        [](TrainingSession &session, PrepareDeviceError *status) {
          try {
            session.prepareDevice();
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
              status->success   = false;
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("err").none());
    cls.def(
        "setRandomSeed", &TrainingSession::setRandomSeed, py::arg("seedValue"));
    cls.def("getCycleCount", &TrainingSession::getCycleCount);
    cls.def("weightsToHost", &TrainingSession::weightsToHost);
    cls.def("weightsFromHost", &TrainingSession::weightsFromHost);
    cls.def("readWeights", &TrainingSession::readWeights);
    cls.def("writeWeights", &TrainingSession::writeWeights);
    cls.def("optimizerFromHost", &TrainingSession::optimizerFromHost);
    cls.def("run", &TrainingSession::run);
    cls.def("modelToHost", &TrainingSession::modelToHost);
    cls.def("getInfo", &TrainingSession::getInfo);
    cls.def("getSummaryReport",
            &TrainingSession::getSummaryReport,
            py::arg("resetProfile") = true);
    cls.def(
        "getGraphReport",
        [](const TrainingSession &session, bool useCbor) {
          auto report = session.getGraphReport(useCbor);
          return py::bytes(report);
        },
        py::arg("useCbor") = false);
    cls.def(
        "getExecutionReport",
        [](const TrainingSession &session, bool useCbor, bool resetProfile) {
          auto report = session.getExecutionReport(useCbor, resetProfile);
          return py::bytes(report);
        },
        py::arg("useCbor")      = false,
        py::arg("resetProfile") = true);
    cls.def("getSerializedGraph", [](const TrainingSession &session) {
      auto report = session.getSerializedGraph();
      return py::bytes(report);
    });
    cls.def("getTensorTileMap", &TrainingSession::getTensorTileMap);
    cls.def("resetHostWeights",
            &TrainingSession::resetHostWeights,
            py::arg("modelProtoOrFilename"),
            py::arg("ignoreWeightsInModelWithoutCorrespondingHostWeight") =
                false);
    // Special test method to write serialise ir for analysis
    cls.def("_serializeIr", &TrainingSession::serializeIr, py::arg("format"));
    // Accessor for internal objects
    cls.def("getIr", &TrainingSession::getIr);
    cls.def("getHostReduceStreamIds", &TrainingSession::getHostReduceStreamIds);
    cls.def("connectStreamToCallback",
            &TrainingSession::connectStreamToCallback);
  }
  {
    py::class_<GraphTransformer> cls(m, "GraphTransformer");
    cls.def(py::init<const std::string &>(), py::arg("modelProtoOrFilename"));
    cls.def("getModelProto", [](const GraphTransformer &graphtransformer) {
      return py::bytes(graphtransformer.getModelProto());
    });
    cls.def("removeUnusedInputs", &GraphTransformer::removeUnusedInputs);
    cls.def("prepareNodesForTraining",
            &GraphTransformer::prepareNodesForTraining);
    cls.def("convertFloatsToHalfs", &GraphTransformer::convertFloatsToHalfs);
    cls.def("convertInitializersToConstants",
            &GraphTransformer::convertInitializersToConstants,
            py::arg("ids"));
    cls.def("convertAllFixedPointInitializersToConstants",
            &GraphTransformer::convertAllFixedPointInitializersToConstants);
    cls.def("saveInitializersExternally",
            &GraphTransformer::saveInitializersExternally,
            py::arg("ids"),
            py::arg("filename"));
  }

// Include the generated poponx.cpp code
#include "popart.cpp.gen"
  {
    py::class_<AiGraphcoreOpset1> cls(m, "AiGraphcoreOpset1");
    cls.def("groupnormalization",
            &AiGraphcoreOpset1::groupnormalization,
            py::arg("args"),
            py::arg("num_groups"),
            py::arg("epsilon")     = 1e-05f,
            py::arg("debugPrefix") = std::string());
    cls.def("printtensor",
            &AiGraphcoreOpset1::printtensor,
            py::arg("args"),
            py::arg("print_gradient") = 1,
            py::arg("debugPrefix")    = std::string());
    cls.def("scale",
            &AiGraphcoreOpset1::scale,
            py::arg("args"),
            py::arg("scale"),
            py::arg("debugPrefix") = std::string());
    cls.def("lstm",
            &AiGraphcoreOpset1::lstm,
            py::arg("args"),
            py::arg("outputFullSequence") = 1,
            py::arg("debugPrefix")        = std::string());
    cls.def("subsample",
            &AiGraphcoreOpset1::subsample,
            py::arg("args"),
            py::arg("strides"),
            py::arg("debugPrefix") = std::string());
    cls.def("gelu",
            &AiGraphcoreOpset1::gelu,
            py::arg("args"),
            py::arg("debugPrefix") = std::string());
    cls.def("detach",
            &AiGraphcoreOpset1::detach,
            py::arg("args"),
            py::arg("pass_through_creation") = 0,
            py::arg("debugPrefix")           = std::string());
    cls.def("init",
            &AiGraphcoreOpset1::init,
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("init_type"),
            py::arg("debugPrefix") = std::string());
    cls.def("dynamicslice",
            &AiGraphcoreOpset1::dynamicslice,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")   = 0,
            py::arg("debugPrefix") = std::string());
    cls.def("dynamicupdate",
            &AiGraphcoreOpset1::dynamicupdate,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")   = 0,
            py::arg("debugPrefix") = std::string());
    cls.def("dynamiczero",
            &AiGraphcoreOpset1::dynamiczero,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugPrefix") = std::string());
    cls.def("dynamicadd",
            &AiGraphcoreOpset1::dynamicadd,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugPrefix") = std::string());
    cls.def("call",
            &AiGraphcoreOpset1::call,
            py::arg("args"),
            py::arg("num_outputs"),
            py::arg("callee"),
            py::arg("debugPrefix") = std::string());
    cls.def("replicatedallreduce",
            &AiGraphcoreOpset1::replicatedallreduce,
            py::arg("args"),
            py::arg("debugPrefix") = std::string());
  }
  {
    py::class_<Builder> cls(m, "_BuilderCore");
    cls.def(py::init(&Builder::create));
    cls.def(py::init(&Builder::createFromOnnxModel),
            py::arg("modelProtoOrFilename"));
    cls.def("setGraphName", &Builder::setGraphName, py::arg("name"));
    cls.def("addInputTensor",
            py::overload_cast<const TensorInfo &, const std::string &>(
                &Builder::addInputTensor),
            py::arg("tensorInfo"),
            py::arg("debugPrefix") = std::string());
    cls.def("addInputTensor",
            py::overload_cast<const std::string &,
                              const Shape &,
                              const std::string &>(&Builder::addInputTensor),
            py::arg("dataType"),
            py::arg("shape"),
            py::arg("debugPrefix") = std::string());
    cls.def("addUntypedInputTensor",
            &Builder::addUntypedInputTensor,
            py::arg("debugPrefix") = std::string());
    cls.def("addInputTensorFromParentGraph",
            &Builder::addInputTensorFromHigherScope,
            py::arg("tensorId"));
    cls.def(
        "addInitializedInputTensor",
        [](Builder &builder, py::array array, std::string &debugPrefix) {
          ConstVoidData initData;
          initData.data = array.request().ptr;
          initData.info = getTensorInfo(array);
          return builder.addInitializedInputTensor(initData, debugPrefix);
        },
        py::arg("initVal"),
        py::arg("debugPrefix") = std::string());
    cls.def(
        "addOutputTensor", &Builder::addOutputTensor, py::arg("outputName"));
    cls.def("_createSubgraphBuilder",
            &Builder::createSubgraphBuilder,
            pybind11::return_value_policy::reference);
    cls.def("saveModelProto", &Builder::saveModelProto, py::arg("filename"));
    cls.def("saveInitializersExternally",
            &Builder::saveInitializersExternally,
            py::arg("ids"),
            py::arg("filename"));

    // Accessors for the ai.onnx domain builder interface
    cls.def_property_readonly("aiOnnxOpset6", &Builder::aiOnnxOpset6);
    cls.def_property_readonly("aiOnnxOpset7", &Builder::aiOnnxOpset7);
    cls.def_property_readonly("aiOnnxOpset8", &Builder::aiOnnxOpset8);
    cls.def_property_readonly("aiOnnxOpset9", &Builder::aiOnnxOpset9);
    cls.def_property_readonly("aiOnnxOpset10", &Builder::aiOnnxOpset10);
    cls.def_property_readonly("aiOnnxOpset11", &Builder::aiOnnxOpset11);

    // Accessors for the ai.onnxml domain builder interface
    cls.def_property_readonly("aiOnnxMlOpset1", &Builder::aiOnnxMlOpset1);

    // Accessors for the ai.graphcore domain builder interface
    cls.def_property_readonly("aiGraphcoreOpset1", &Builder::aiGraphcoreOpset1);
    // Custom Op interface for separately compiled operations used in python.
    cls.def(
        "customOp",
        [](Builder &builder,
           const std::string &opName,
           const int &OpVersion,
           const std::string &domain,
           const py::list &inputs,
           const py::dict &attr,
           const unsigned &numOutputs,
           const std::string &name) {
          popart::OperatorIdentifier opId = {
              domain, opName, static_cast<popart::OpVersion>(OpVersion)};
          std::vector<TensorId> input_vector;
          for (auto item : inputs) {
            std::string str = py::cast<std::string>(item);
            TensorId t      = static_cast<TensorId>(str);
            input_vector.push_back(t);
          }
          return builder.customOp(
              opId, 1, input_vector, numOutputs, getDictionaryVar(attr), name);
        },
        py::arg("opName"),
        py::arg("opVersion"),
        py::arg("domain"),
        py::arg("inputs"),
        py::arg("attributes"),
        py::arg("numOutputs") = 1,
        py::arg("name")       = std::string());
    cls.def(
        "addNodeAttribute",
        static_cast<void (Builder::*)(
            const std::string &, const int64_t &, const std::set<TensorId> &)>(
            &Builder::addNodeAttribute),
        py::arg("attributeName"),
        py::arg("attributeValue"),
        py::arg("nodeOutputNames"));
    cls.def("addNodeAttribute",
            static_cast<void (Builder::*)(const std::string &,
                                          const std::vector<int64_t> &,
                                          const std::set<TensorId> &)>(
                &Builder::addNodeAttribute),
            py::arg("attributeName"),
            py::arg("attributeValue"),
            py::arg("nodeOutputNames"));
    cls.def(
        "addNodeAttribute",
        static_cast<void (Builder::*)(
            const std::string &, const float &, const std::set<TensorId> &)>(
            &Builder::addNodeAttribute),
        py::arg("attributeName"),
        py::arg("attributeValue"),
        py::arg("nodeOutputNames"));
    cls.def("addNodeAttribute",
            static_cast<void (Builder::*)(const std::string &,
                                          const std::vector<float> &,
                                          const std::set<TensorId> &)>(
                &Builder::addNodeAttribute),
            py::arg("attributeName"),
            py::arg("attributeValue"),
            py::arg("nodeOutputNames"));
    cls.def("addNodeAttribute",
            static_cast<void (Builder::*)(const std::string &,
                                          const std::string &,
                                          const std::set<TensorId> &)>(
                &Builder::addNodeAttribute),
            py::arg("attributeName"),
            py::arg("attributeValue"),
            py::arg("nodeOutputNames"));
    cls.def("addNodeAttribute",
            static_cast<void (Builder::*)(const std::string &,
                                          const std::vector<std::string> &,
                                          const std::set<TensorId> &)>(
                &Builder::addNodeAttribute),
            py::arg("attributeName"),
            py::arg("attributeValue"),
            py::arg("nodeOutputNames"));
    cls.def("nodeHasAttribute",
            &Builder::nodeHasAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getInt64NodeAttribute",
            &Builder::getInt64NodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getInt64VectorNodeAttribute",
            &Builder::getInt64VectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getFloatNodeAttribute",
            &Builder::getFloatNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getFloatVectorNodeAttribute",
            &Builder::getFloatVectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getStringNodeAttribute",
            &Builder::getStringNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getStringVectorNodeAttribute",
            &Builder::getStringVectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("removeNodeAttribute",
            &Builder::removeNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"));
    cls.def("getAllNodeAttributeNames",
            &Builder::getAllNodeAttributeNames,
            py::arg("nodeOutputNames"));
    cls.def("getModelProto", [](const Builder &builder) {
      return py::bytes(builder.getModelProto());
    });
    cls.def("getInputTensorIds", &Builder::getInputTensorIds);
    cls.def("getOutputTensorIds", &Builder::getOutputTensorIds);
    cls.def("getValueTensorIds", &Builder::getValueTensorIds);
    cls.def("getTensorShape", &Builder::getTensorShape, py::arg("id"));
    cls.def(
        "getTensorDtypeString", &Builder::getTensorDtypeString, py::arg("id"));
    cls.def("isInitializer", &Builder::isInitializer, py::arg("id"));
    cls.def("virtualGraph",
            static_cast<void (Builder::*)(const TensorId &, int64_t value)>(
                &Builder::virtualGraph),
            py::arg("nodeOutputNames"),
            py::arg("value") = 0);
    cls.def(
        "virtualGraph",
        [](Builder &self, int64_t index) -> AttributeContextManager {
          AttributeContextManager acm(self, sVirtualGraphAttribute, index);
          return acm;
        },
        py::arg("value"));
    cls.def("pingPongPhase",
            static_cast<void (Builder::*)(const TensorId &, int64_t phase)>(
                &Builder::pingPongPhase),
            py::arg("nodeOutputNames"),
            py::arg("value") = 0);
    cls.def(
        "pingPongPhase",
        [](Builder &self, int64_t phase) -> AttributeContextManager {
          AttributeContextManager acm(self, sPingPongPhaseAttribute, phase);
          return acm;
        },
        py::arg("value") = 0);
    cls.def(
        "getPingPongPhase",
        static_cast<int64_t (Builder::*)() const>(&Builder::getPingPongPhase));
    cls.def("hasPingPongPhase", [](Builder &self) -> bool {
      return self.hasAttribute(sPingPongPhaseAttribute);
    });
    cls.def(
        "recomputeOutput",
        static_cast<void (Builder::*)(const TensorId &, RecomputeType value)>(
            &Builder::recomputeOutput),
        py::arg("nodeOutputNames"),
        py::arg("value") = RecomputeType::UNDEFINED);
    cls.def(
        "recomputeOutput",
        [](Builder &self, RecomputeType value) -> AttributeContextManager {
          AttributeContextManager acm(
              self, sRecomputeOutputAttribute, static_cast<int64_t>(value));
          return acm;
        },
        py::arg("value") = RecomputeType::UNDEFINED);
    cls.def("cacheOutput",
            static_cast<void (Builder::*)(const TensorId &, CacheType value)>(
                &Builder::cacheOutput),
            py::arg("nodeOutputNames"),
            py::arg("value") = CacheType::UNDEFINED);
    cls.def(
        "cacheOutput",
        [](Builder &self, CacheType value) -> AttributeContextManager {
          AttributeContextManager acm(
              self, sCacheOutputAttribute, static_cast<int64_t>(value));
          return acm;
        },
        py::arg("value") = CacheType::UNDEFINED);
    cls.def("pipelineStage",
            static_cast<void (Builder::*)(const TensorId &, int64_t value)>(
                &Builder::pipelineStage),
            py::arg("nodeOutputNames"),
            py::arg("value") = 0);
    cls.def(
        "pipelineStage",
        [](Builder &self, int64_t index) -> AttributeContextManager {
          AttributeContextManager acm(self, sPipelineStageAttribute, index);
          return acm;
        },
        py::arg("value"));
    cls.def(
        "schedulePriority",
        [](Builder &self, float priority) -> AttributeContextManager {
          AttributeContextManager acm(self, sSchedulePriority, priority);
          return acm;
        },
        py::arg("value"));
    cls.def("excludePatterns",
            static_cast<void (Builder::*)(
                const TensorId &, const std::vector<std::string> &value)>(
                &Builder::excludePatterns),
            py::arg("nodeOutputName"),
            py::arg("patternNames"));
    cls.def("getPipelineStage", &Builder::getPipelineStage);
    cls.def("hasPipelineStage", [](Builder &self) -> bool {
      return self.hasAttribute(sPipelineStageAttribute);
    });
    cls.def(
        "getVirtualGraph",
        static_cast<int64_t (Builder::*)() const>(&Builder::getVirtualGraph));
    cls.def("hasVirtualGraph", [](Builder &self) -> bool {
      return self.hasAttribute(sVirtualGraphAttribute);
    });
    cls.def("setPartialsType",
            &Builder::setPartialsType,
            py::arg("nodeOutputName"),
            py::arg("partialsType"));
    cls.def("getPartialsType",
            &Builder::getPartialsType,
            py::arg("nodeOutputName"));
    cls.def("setAvailableMemoryProportion",
            &Builder::setAvailableMemoryProportion,
            py::arg("nodeOutputName"),
            py::arg("availableMemoryProportion"));
    cls.def(
        "setSerializeMatMul",
        [](Builder &self,
           const std::set<TensorId> &nodeOutputNames,
           std::string mode,
           int64_t factor,
           bool keep_precision) {
          self.setSerializeMatMul(
              nodeOutputNames, mode, factor, keep_precision);
        },
        py::arg("nodeOutputName"),
        py::arg("mode"),
        py::arg("factor")         = 0,
        py::arg("keep_precision") = false);
    cls.def(
        "nameScope",
        [](Builder &self, const std::string &name) -> NameContextManager {
          NameContextManager ncm(self, name);
          return ncm;
        },
        py::arg("name"));
    cls.def(
        "getNameScope",
        [](Builder &self, std::string &name) {
          return self.getNameScope(name);
        },
        py::arg("name") = "");
    cls.def("getVirtualGraph",
            static_cast<int64_t (Builder::*)(const TensorId &)>(
                &Builder::getVirtualGraph),
            py::arg("nodeOutputNames"));

    cls.def(
        "recomputeOutputInBackwardPass",
        static_cast<void (Builder::*)(const TensorId &, RecomputeType value)>(
            &Builder::recomputeOutputInBackwardPass),
        py::arg("nodeOutputName"),
        py::arg("value") = RecomputeType::RECOMPUTE);
    cls.def("recomputeOutputInBackwardPass",
            static_cast<void (Builder::*)(const std::set<TensorId> &,
                                          RecomputeType value)>(
                &Builder::recomputeOutputInBackwardPass),
            py::arg("nodeOutputNames"),
            py::arg("value") = RecomputeType::RECOMPUTE);

    cls.def("getRecomputeOutputInBackwardPass",
            static_cast<bool (Builder::*)(const TensorId &)>(
                &Builder::getRecomputeOutputInBackwardPass),
            py::arg("nodeOutputName"));

    cls.def("getRecomputeOutputInBackwardPass",
            static_cast<bool (Builder::*)(const std::set<TensorId> &)>(
                &Builder::getRecomputeOutputInBackwardPass),
            py::arg("nodeOutputNames"));

    cls.def("setInplacePreferences",
            static_cast<void (Builder::*)(const TensorId &,
                                          const std::map<OpType, float> &)>(
                &Builder::setInplacePreferences),
            py::arg("nodeOutputName"),
            py::arg("prefs"));
  }
  {
    py::class_<AttributeContextManager> cls(m, "AttributeContextManager");
    cls.def("__enter__", &AttributeContextManager::enter);
    cls.def("__exit__",
            [](AttributeContextManager &self,
               py::object &,
               py::object &,
               py::object &) { self.exit(); });
  }
  {
    py::class_<NameContextManager> cls(m, "NameContextManager");
    cls.def("__enter__", &NameContextManager::enter);
    cls.def(
        "__exit__",
        [](NameContextManager &self, py::object &, py::object &, py::object &) {
          self.exit();
        });
  }
  {
    py::enum_<DeviceType> en(m, "DeviceType");
    en.value("IpuModel", DeviceType::IpuModel);
    en.value("Cpu", DeviceType::Cpu);
    en.value("Ipu", DeviceType::Ipu);
    en.value("Sim", DeviceType::Sim);
  }
  {
    py::enum_<DeviceConnectionType> en(m, "DeviceConnectionType");
    en.value("ALWAYS", DeviceConnectionType::ALWAYS);
    en.value("ON_DEMAND", DeviceConnectionType::ON_DEMAND);
    en.value("NEVER", DeviceConnectionType::NEVER);
  }

  {
    // PyBinding to a singleton
    py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>> cls(
        m, "DeviceManager");
    cls.def(py::init([]() {
      return std::unique_ptr<DeviceManager, py::nodelete>(
          &DeviceManager::createDeviceManager());
    }));
    cls.def("acquireAvailableDevice",
            static_cast<std::shared_ptr<DeviceInfo> (DeviceManager::*)(
                int, int, SyncPattern, uint32_t, DeviceConnectionType)>(
                &DeviceManager::acquireAvailableDevice),
            py::arg("numIpus")           = 1,
            py::arg("tilesPerIpu")       = 0,
            py::arg("pattern")           = SyncPattern::Full,
            py::arg("replicationFactor") = 1,
            py::arg("connectionType")    = DeviceConnectionType::ALWAYS);
    cls.def("acquireDeviceById",
            &DeviceManager::acquireDeviceById,
            py::arg("id"),
            py::arg("pattern")           = SyncPattern::Full,
            py::arg("replicationFactor") = 1,
            py::arg("connectionType")    = DeviceConnectionType::ALWAYS);
    cls.def("createCpuDevice", &DeviceManager::createCpuDevice);
    cls.def("createIpuModelDevice", [](DeviceManager &dm, py::dict e) {
      std::map<std::string, std::string> options = getDictionary(e);
      return dm.createIpuModelDevice(options);
    });
    cls.def("createSimDevice", [](DeviceManager &dm, py::dict e) {
      std::map<std::string, std::string> options = getDictionary(e);
      return dm.createSimDevice(options);
    });
    cls.def("enumerateDevices",
            &DeviceManager::enumerateDevices,
            py::arg("pattern")           = SyncPattern::Full,
            py::arg("replicationFactor") = 1,
            py::arg("numIpus")           = 1,
            py::arg("deviceType")        = DeviceType::Ipu,
            py::arg("connectionType")    = DeviceConnectionType::ALWAYS);
  }
  {
    py::class_<DeviceInfo, std::shared_ptr<DeviceInfo>> cls(m, "DeviceInfo");
    cls.def("attach", &DeviceInfo::attach);
    cls.def("detach", &DeviceInfo::detach);
    cls.def_property_readonly("type", &DeviceInfo::getType);
    cls.def_property_readonly("connectionType", &DeviceInfo::getConnectionType);
    cls.def_property_readonly("version", &DeviceInfo::getVersion);
    cls.def_property_readonly("id", &DeviceInfo::getId);
    cls.def_property_readonly("numIpus", &DeviceInfo::getNumIpus);
    cls.def_property_readonly("tilesPerIpu", &DeviceInfo::getTilesPerIpu);
    cls.def_property_readonly("driverIds", &DeviceInfo::getDriverIds);

    cls.def_property_readonly("numWorkerContexts",
                              &DeviceInfo::getNumWorkerContexts);
    cls.def("__repr__", [](const DeviceInfo &di) {
      std::stringstream ss;
      ss << di;
      return ss.str();
    });
  }
  m.def("reservedGradientPrefix", &reservedGradientPrefix);
  m.def("reservedUpdatedVarPrefix", &reservedUpdatedVarPrefix);

  m.def("reservedAcclToAccumulatorPrefix", &reservedAcclToAccumulatorPrefix);
  m.def("reservedAcclToReducePrefix", &reservedAcclToReducePrefix);
  m.def("reservedAcclToUpdatePrefix", &reservedAcclToUpdatePrefix);
  m.def("reservedAcclFinalOutPrefix", &reservedAcclFinalOutPrefix);

  m.def("reservedStashedPrefix", &reservedStashedPrefix);
  m.def("reservedRestoredPrefix", &reservedRestoredPrefix);
  m.def("reservedLossScalingPrefix", &reservedLossScalingPrefix);
  m.def("reservedDefaultScaledLearningRate0Prefix",
        &reservedDefaultScaledLearningRate0Prefix);
  m.def("reservedDefaultWeightDecayScaleFactor0Prefix",
        &reservedDefaultWeightDecayScaleFactor0Prefix);
  m.def("reservedSpecificScaledLearningRate0Prefix",
        &reservedSpecificScaledLearningRate0Prefix);
  m.def("reservedSpecificWeightDecayScaleFactor0Prefix",
        &reservedSpecificWeightDecayScaleFactor0Prefix);

  // Exceptions are processed explicitly to allow the main dynamic library
  // to do the type inference.  This prevents some inter dynamic library type
  // inference issues on OS/X
  static py::exception<popart::error> ePopart(m, "popart_exception");
  static py::exception<popart::internal_error> ePopartInternal(
      m, "popart_internal_exception");
  static py::exception<poplar::poplar_error> ePoplar(m, "poplar_exception");
  static py::exception<poputil::poplibs_error> ePoplibs(m, "poplibs_exception");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      std::rethrow_exception(p);
    } catch (std::exception &e) {
      switch (popart::getErrorSource(e)) {
      case popart::ErrorSource::popart:
        ePopart(e.what());
        return;
      case popart::ErrorSource::popart_internal:
        ePopartInternal(e.what());
        return;
      case popart::ErrorSource::poplar:
        ePoplar(e.what());
        return;
      case popart::ErrorSource::poplibs:
        ePoplibs(e.what());
        return;
      case popart::ErrorSource::unknown:
        throw;
      }
    }
  });
}
