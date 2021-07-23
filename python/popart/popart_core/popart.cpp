// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <pybind11/functional.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../shared_cpp/np_utils.hpp"
#include "../shared_cpp/pyarray_accessor.hpp"

#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/builder.hpp>
#include <popart/debugcontext.hpp>
#include <popart/devicemanager.hpp>
#include <popart/docs/pydocs_popart_core.hpp>
#include <popart/docs/pydocs_popart_custom.hpp>
#include <popart/error.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/numerics.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/sgd.hpp>
#include <popart/stepio_generic.hpp>
#include <popart/stepio_size_assertion.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/vendored/optional.hpp>
#include <popart/version.hpp>

#include <stdexcept>
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace py = pybind11;
using namespace popart;

namespace popart {
namespace popx {
class Executablex;
}
} // namespace popart

namespace {
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

std::map<std::string, popart::any> getDictionaryVar(py::dict pydict) {
  // This attempts to convert the py::dict to a map of string, popart::any.
  // Since we do not know the python types given by the user until runtime, we
  // have to account for each type. See attributes.hpp for a description of
  // possible attribute types.

  std::map<std::string, popart::any> dictionary;
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
} // namespace

class PyStepIO : public StepIOGeneric<StepIONS::PyArray,
                                      StepIONS::PyArrayAccessor,
                                      StepIONS::PyArray> {
public:
  PyStepIO(std::map<TensorId, py::array> inputs,
           std::map<TensorId, py::array> outputs) {
    for (auto p : inputs) {
      if (isContiguous(p.second)) {
        inputsInfo.insert({p.first, {p.second, 0}});
      } else {
        throw error("PyStepIO is unable to use the numpy input array for "
                    "tensor '{}' as it is not c-contiguous (a data conversion "
                    "here could have a significant impact on performance and "
                    "hence is not allowed)",
                    p.first);
      }
    }

    for (auto p : outputs) {
      if (isContiguous(p.second)) {
        outputsInfo.insert({p.first, {p.second, 0}});
      } else {
        throw error("PyStepIO is unable to use the numpy output array for "
                    "tensor '{}' as it is not c-contiguous (a data conversion "
                    "here could have a significant impact on performance and "
                    "hence is not allowed)",
                    p.first);
      }
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

  void assertNumElements(const popx::Executablex &) const final {}

  ConstVoidData in(TensorId id, int64_t, bool prefetch) final {
    py::gil_scoped_acquire acquire;
    py::array a = inputCb(id, prefetch);
    if (!isContiguous(a)) {
      throw error(
          "PyStepIO is unable to use the numpy input array for tensor "
          "'{}' as it is not c-contiguous (a data conversion here could have a "
          "significant impact on performance and hence is not allowed)",
          id);
    }
    // To ensure that array is persisted until complete is called
    inDict[py::str(id)] = a;

    ConstVoidData data;

    // If a None object has been returned ndim will be 0
    if (a.ndim() > 0) {
      data.data = a.request().ptr;
      data.info = getTensorInfo(a);
    }

    return data;
  }

  void inComplete(TensorId id, int64_t) final {
    py::gil_scoped_acquire acquire;
    inputCompleteCb(id);
    inDict[py::str(id)] = py::none();
  }

  MutableVoidData out(TensorId id, int64_t) final {
    py::gil_scoped_acquire acquire;
    py::array a = outputCb(id);
    if (!isContiguous(a)) {
      throw error(
          "PyStepIO is unable to use the numpy output array for tensor "
          "'{}' as it is not c-contiguous (a data conversion here could have a "
          "significant impact on performance and hence is not allowed)",
          id);
    }

    // To ensure that array is persisted until complete is called
    outDict[py::str(id)] = a;

    MutableVoidData data;
    data.data = a.request().ptr;
    data.info = getTensorInfo(a);
    return data;
  }

  void outComplete(TensorId id) final {
    py::gil_scoped_acquire acquire;
    outputCompleteCb(id);
    outDict[py::str(id)] = py::none();
  }

private:
  // user land callbacks
  InputCallback inputCb;
  InputCompleteCallback inputCompleteCb;
  OutputCallback outputCb;
  OutputCompleteCallback outputCompleteCb;
  py::dict inDict  = py::dict();
  py::dict outDict = py::dict();
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
  popart::any value;
  std::vector<popart::any> prevValue;

public:
  AttributeContextManager(Builder &_builder,
                          const std::string &_attribute,
                          popart::any value_)
      : builder(_builder), attribute(_attribute), value(value_) {}

  void enter() {
    if (builder.hasAttribute(attribute)) {
      // Backup previous attribute value
      prevValue.push_back(
          popart::any_cast<int64_t>(builder.getAttribute(attribute)));
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

class KeyValueContextManager {
  Builder &builder;
  std::string attribute;
  std::map<std::string, std::string> value;
  std::vector<std::map<std::string, std::string>> prevValue;

public:
  KeyValueContextManager(Builder &_builder,
                         const std::string &_attribute,
                         std::map<std::string, std::string> value_)
      : builder(_builder), attribute(_attribute), value(value_) {}

  void enter() {
    if (builder.hasAttribute(attribute)) {
      auto prevKeyValuePairs = popart::any_cast<std::vector<std::string>>(
          builder.getAttribute(attribute));
      // Backup previous attribute values
      prevValue.emplace_back();
      for (size_t i = 0; i < prevKeyValuePairs.size(); i += 2) {
        prevValue.back().insert(
            {prevKeyValuePairs[i], prevKeyValuePairs[i + 1]});
      }
      builder.clearAttribute(attribute);
    }
    std::map<std::string, std::string> map;
    if (!prevValue.empty()) {
      map.insert(prevValue.back().begin(), prevValue.back().end());
    }
    map.insert(value.begin(), value.end());

    std::vector<std::string> keyValuePairs;
    for (auto &kv : map) {
      keyValuePairs.push_back(kv.first);
      keyValuePairs.push_back(kv.second);
    }

    builder.setAttribute(attribute, keyValuePairs);
  }
  void exit() {
    builder.clearAttribute(attribute);
    if (!prevValue.empty()) {
      // Restore previous attribute value
      std::vector<std::string> keyValuePairs;
      for (auto &kv : prevValue.back()) {
        keyValuePairs.push_back(kv.first);
        keyValuePairs.push_back(kv.second);
      }
      builder.setAttribute(attribute, keyValuePairs);
      prevValue.pop_back();
    }
  }
};

struct OutOfMemoryError {
  std::unique_ptr<popart::memory_allocation_err> exception;

  virtual ~OutOfMemoryError() {}

  virtual bool isSuccessful() const { return exception.get() == nullptr; }
  std::string what() const {
    if (exception) {
      return exception->what();
    } else {
      return "No memory_allocation_err raised";
    }
  }
  std::string getSummaryReport() const { return exception->getSummaryReport(); }
  std::string getProfilePath() const { return exception->getProfilePath(); }
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

PYBIND11_MODULE(popart_core, m) {
  // Import the popart._internal.ir module to reuse some bindings.
  py::module popart_internal_ir = py::module::import("popart._internal.ir");

  m.doc() = "binding for C++ popart library";

  m.attr("defaultAiOnnxOpset")      = defaultAiOnnxOpset;
  m.attr("defaultAiOnnxMlOpset")    = defaultAiOnnxMlOpset;
  m.attr("defaultAiGraphcoreOpset") = defaultAiGraphcoreOpset;

  m.def("getTensorInfo", &getTensorInfo);

  m.def("syncPatternFromString", &syncPatternFromString);
  m.def("syncPatternToString", &syncPatternToString);

  m.def("getLogger", &Logger::getLogger, py::arg("name") = "all");

  m.def("versionString", &popart::core::versionString);
  m.def("versionNumber", []() {
    auto version = popart::core::versionNumber();
    return py::make_tuple(version.major, version.minor, version.point);
  });
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
    // Reuse SourceLocation from the popart._internal.ir module.
    m.attr("SourceLocation") = popart_internal_ir.attr("SourceLocation");
  }

  {
    // Reuse DebugContext from the popart._internal.ir module.
    m.attr("DebugContext") = popart_internal_ir.attr("DebugContext");
  }

  {
    // Reuse DebugInfo from the popart._internal.ir module.
    m.attr("DebugInfo") = popart_internal_ir.attr("DebugInfo");
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
  m.def("getUnsupportedOperations",
        &OpManager::getUnsupportedOperations,
        py::arg("opsetVersion"));
  {
    // Reuse DataType from the popart._internal.ir module.
    m.attr("DataType") = popart_internal_ir.attr("DataType");
  }
  {
    py::enum_<InitType> en(m, "InitType", DOC(popart, InitType));
    en.value("NoInit", InitType::NoInit, DOC(popart, InitType, NoInit));
    en.value("Zero", InitType::Zero, DOC(popart, InitType, Zero));
  }
  {
    py::enum_<TileSet> en(m, "TileSet", DOC(popart, TileSet));
    en.value("Compute", TileSet::Compute, DOC(popart, TileSet, Compute));
    en.value("IO", TileSet::IO, DOC(popart, TileSet, IO));
  }
  {
    py::enum_<ExchangeStrategy> en(
        m, "ExchangeStrategy", DOC(popart, ExchangeStrategy));
    en.value("JustInTime",
             ExchangeStrategy::JustInTime,
             DOC(popart, ExchangeStrategy, JustInTime));
    en.value("OverlapInnerLoop",
             ExchangeStrategy::OverlapInnerLoop,
             DOC(popart, ExchangeStrategy, OverlapInnerLoop));
    en.value("OverlapLoops",
             ExchangeStrategy::OverlapLoops,
             DOC(popart, ExchangeStrategy, OverlapLoops));
    en.value("OverlapStep",
             ExchangeStrategy::OverlapStep,
             DOC(popart, ExchangeStrategy, OverlapStep));
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
    en.value("Final",
             AnchorReturnTypeId::Final,
             DOC(popart, AnchorReturnTypeId, Final));
    en.value("EveryN",
             AnchorReturnTypeId::EveryN,
             DOC(popart, AnchorReturnTypeId, EveryN));
    en.value(
        "All", AnchorReturnTypeId::All, DOC(popart, AnchorReturnTypeId, All));
    en.value(
        "Sum", AnchorReturnTypeId::Sum, DOC(popart, AnchorReturnTypeId, Sum));

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
    cls.def(py::init<std::string, TileSet, ExchangeStrategy>(),
            py::arg("anchorReturnTypeString"),
            py::arg("tileSet")          = TileSet::Compute,
            py::arg("exchangeStrategy") = ExchangeStrategy::JustInTime);
    cls.def(py::init<std::string, int, TileSet, ExchangeStrategy>(),
            py::arg("anchorReturnTypeString"),
            py::arg("returnPeriod"),
            py::arg("tileSet")          = TileSet::Compute,
            py::arg("exchangeStrategy") = ExchangeStrategy::JustInTime);
    cls.def("id", &AnchorReturnType::id);
    cls.def("rp", &AnchorReturnType::rp);
    cls.def("tileSet", &AnchorReturnType::tileSet);
    cls.def("exchangeStrategy", &AnchorReturnType::exchangeStrategy);
  }
  {
    py::class_<DataFlow> cls(m, "DataFlow");
    cls.def(py::init<int, const AnchorReturnTypeMap &>(),
            py::arg("batchesPerStep"),
            py::arg("anchorTensors"));
    cls.def(py::init<int, const AnchorReturnTypeMap &>(),
            py::arg("batchesPerStep"),
            py::arg("anchorTensors"));
    cls.def(py::init<int,
                     const std::vector<TensorId> &,
                     const AnchorReturnType &>(),
            py::arg("batchesPerStep"),
            py::arg("anchorIds"),
            py::arg("anchorReturnType") = AnchorReturnType("All"));
    cls.def("isAnchored", &DataFlow::isAnchored);
    cls.def("nAnchors", &DataFlow::nAnchors);
    cls.def("batchesPerStep", &DataFlow::batchesPerStep);
    cls.def("anchors",
            &DataFlow::anchors,
            pybind11::return_value_policy::copy,
            DOC(popart, DataFlow, anchors));
    cls.def("art", &DataFlow::art);
  }
  {
    py::class_<InputSettings> cls(m, "InputSettings");
    cls.def(py::init<TileSet, ExchangeStrategy>(),
            py::arg("tileSet")          = TileSet::Compute,
            py::arg("exchangeStrategy") = ExchangeStrategy::JustInTime);
    cls.def("tileSet", &InputSettings::tileSet);
    cls.def("exchangeStrategy", &InputSettings::exchangeStrategy);
  }
  {
    // Reuse TensorInfo from the popart._internal.ir module.
    m.attr("_TensorInfoCore") = popart_internal_ir.attr("TensorInfo");
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
    py::enum_<ReductionType> en(m, "ReductionType");
    en.value("Mean", ReductionType::Mean, DOC(popart, ReductionType, Mean));
    en.value("NoReduction",
             ReductionType::NoReduction,
             DOC(popart, ReductionType, NoReduction));
    en.value("Sum", ReductionType::Sum, DOC(popart, ReductionType, Sum));
  }
  {
    // T42271
    py::enum_<MeanReductionStrategy> en(m, "MeanReductionStrategy");
    en.value("Running", MeanReductionStrategy::Running);
    en.value("Post", MeanReductionStrategy::Post);
    en.value("PostAndLoss", MeanReductionStrategy::PostAndLoss);
  }
  {
    py::enum_<ScatterReduction> en(m, "ScatterReduction");
    en.value("Sum", ScatterReduction::Sum);
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
      py::enum_<WeightDecayMode> en(m, "WeightDecayMode");
      en.value(
          "Decay", WeightDecayMode::Decay, DOC(popart, WeightDecayMode, Decay));
      en.value("L2Regularization",
               WeightDecayMode::L2Regularization,
               DOC(popart, WeightDecayMode, L2Regularization));
    }

    {
      py::enum_<SGDAccumulatorAndMomentum> en(m, "SGDAccumulatorAndMomentum");
      en.value("Combined",
               SGDAccumulatorAndMomentum::Combined /*,
               DOC(popart, SGDAccumulatorAndMomentum, Combined)*/);
      en.value("Separate",
               SGDAccumulatorAndMomentum::Separate /*,
               DOC(popart, SGDAccumulatorAndMomentum, Separate) */);
    }

    {
      py::class_<SGD> sgd(m, "SGD", optimizer, DOC(popart, SGD));
      sgd.def(py::init([](py::dict pyd,
                          std::vector<ClipNormSettings> clipNormSettings,
                          SGDAccumulatorAndMomentum accumulatorAndMomentum,
                          DataType accumType,
                          DataType accl1Type) {
                auto cppm = getOptimizerValueDictionary(pyd);
                return SGD(cppm,
                           clipNormSettings,
                           accumulatorAndMomentum,
                           accumType,
                           accl1Type);
              }),
              py::arg("pyd"),
              py::arg("clip_norm_settings") = std::vector<ClipNormSettings>{},
              py::arg("accumulatorAndMomentum") =
                  SGDAccumulatorAndMomentum::Combined,
              py::arg("accumType") = DataType::UNDEFINED,
              py::arg("accl1Type") = DataType::UNDEFINED);
      sgd.def("insertSpecific", [](SGD &self, TensorId id, py::dict pyd) {
        self.insertSpecific(id, getOptimizerValueDictionary(pyd));
      });

      sgd.def("learningRates", &SGD::learningRates);
      sgd.def("weightDecays", &SGD::weightDecays);
      sgd.def("momentums", &SGD::momentums);
      sgd.def("dampenings", &SGD::dampenings);
      sgd.def("velocityScalings", &SGD::velocityScalings);

      { // This class is deprecated, and SGD should be preferred
        py::class_<ConstSGD> cls(m, "ConstSGD", sgd, DOC(popart, ConstSGD));
        cls.def(py::init<float, float, float, std::vector<ClipNormSettings>>(),
                py::arg("learning_rate"),
                py::arg("weight_decay") = 0.0f,
                py::arg("loss_scaling") = 1.0f,
                py::arg("clip_norm_settings") =
                    std::vector<ClipNormSettings>{});
      }
    }
    {
      {
        py::enum_<AdamMode> en(m, "AdamMode");
        en.value("Adam", AdamMode::Adam, DOC(popart, AdamMode, Adam));
        en.value("AdamNoBias",
                 AdamMode::AdamNoBias,
                 DOC(popart, AdamMode, AdamNoBias));
        en.value("Lamb", AdamMode::Lamb, DOC(popart, AdamMode, Lamb));
        en.value("LambNoBias",
                 AdamMode::LambNoBias,
                 DOC(popart, AdamMode, LambNoBias));
        en.value("AdaMax", AdamMode::AdaMax, DOC(popart, AdamMode, AdaMax));
      }
      py::class_<Adam> adam(m, "Adam", optimizer, DOC(popart, Adam));
      adam.def(py::init([](py::dict pyd,
                           AdamMode mode,
                           WeightDecayMode wdmode,
                           DataType accumType,
                           DataType accl1Type,
                           DataType accl2Type,
                           std::vector<ClipNormSettings> clipNormSettings,
                           bool scaledOptimizerState) {
                 auto cppm = getOptimizerValueDictionary(pyd);
                 return Adam(cppm,
                             mode,
                             wdmode,
                             accumType,
                             accl1Type,
                             accl2Type,
                             clipNormSettings,
                             scaledOptimizerState);
               }),
               py::arg("values"),
               py::arg("mode")              = AdamMode::Adam,
               py::arg("weight_decay_mode") = WeightDecayMode::Decay,
               // Choose same data type as weight for the accumulator by default
               py::arg("accum_type") = DataType::UNDEFINED,
               // Momentums in FP32 by default
               py::arg("accl1_type")         = DataType::FLOAT,
               py::arg("accl2_type")         = DataType::FLOAT,
               py::arg("clip_norm_settings") = std::vector<ClipNormSettings>{},
               py::arg("scaled_optimizer_state") = false);

      adam.def("insertSpecific", [](Adam &self, TensorId id, py::dict pyd) {
        self.insertSpecific(id, getOptimizerValueDictionary(pyd));
      });

      adam.def("learningRates", &Adam::learningRates);
      adam.def("weightDecays", &Adam::weightDecays);
      adam.def("beta1s", &Adam::beta1s);
      adam.def("beta2s", &Adam::beta1s);
      adam.def("epss", &Adam::epss);
      adam.def("maxWeightNorms", &Adam::maxWeightNorms);
    }
    {
      {
        py::enum_<AdaptiveMode> en(m, "AdaptiveMode");
        en.value("AdaGrad", AdaptiveMode::AdaGrad);
        en.value("RMSProp", AdaptiveMode::RMSProp);
        en.value("CenteredRMSProp", AdaptiveMode::CenteredRMSProp);
        en.value("AdaDelta", AdaptiveMode::AdaDelta);
      }
      py::class_<Adaptive> adaptive(m, "Adaptive", optimizer);
      adaptive.def(
          py::init([](py::dict pyd,
                      AdaptiveMode mode,
                      WeightDecayMode wdmode,
                      DataType accumType,
                      DataType accl1Type,
                      DataType accl2Type,
                      DataType accl3Type,
                      bool rmspropTFVariant) {
            auto cppm = getOptimizerValueDictionary(pyd);
            return Adaptive(cppm,
                            mode,
                            wdmode,
                            accumType,
                            accl1Type,
                            accl2Type,
                            accl3Type,
                            rmspropTFVariant);
          }),
          py::arg("values"),
          py::arg("mode")              = AdaptiveMode::RMSProp,
          py::arg("weight_decay_mode") = WeightDecayMode::Decay,
          // Choose same data type as weight for the accumulator by default
          py::arg("accum_type") = DataType::UNDEFINED,
          // Accl1 / Accl2 in FP32 by default
          py::arg("accl1_type") = DataType::FLOAT,
          py::arg("accl2_type") = DataType::FLOAT,
          // Choose same data type as weight for the Accl3 by default
          py::arg("accl3_type")         = DataType::UNDEFINED,
          py::arg("rmsprop_tf_variant") = false);

      adaptive.def("insertSpecific",
                   [](Adaptive &self, TensorId id, py::dict pyd) {
                     self.insertSpecific(id, getOptimizerValueDictionary(pyd));
                   });

      adaptive.def("learningRates", &Adaptive::learningRates);
      adaptive.def("weightDecays", &Adaptive::weightDecays);
      adaptive.def("alphas", &Adaptive::alphas);
      adaptive.def("momentums", &Adaptive::momentums);
      adaptive.def("epss", &Adaptive::epss);
    }
  }
  {
    py::class_<TensorLocation> cls(m, "TensorLocation");
    cls.def(py::init<>());
    cls.def(py::init<TensorStorage>(), py::arg("storage"));
    cls.def(py::init<TensorStorage, ReplicatedTensorSharding>(),
            py::arg("storage"),
            py::arg("replicatedTensorSharding"));
    cls.def(
        py::init<TensorStorage, TileSet, TileSet, ReplicatedTensorSharding>(),
        py::arg("storage"),
        py::arg("loadTileSet"),
        py::arg("storageTileSet"),
        py::arg("replicatedTensorSharding"));
    cls.def_readwrite("storage",
                      &TensorLocation::storage,
                      DOC(popart, TensorLocation, storage));
    cls.def_readwrite("loadTileSet",
                      &TensorLocation::loadTileSet,
                      DOC(popart, TensorLocation, loadTileSet));
    cls.def_readwrite("storageTileSet",
                      &TensorLocation::storageTileSet,
                      DOC(popart, TensorLocation, storageTileSet));
    cls.def_readwrite("replicatedTensorSharding",
                      &TensorLocation::replicatedTensorSharding,
                      DOC(popart, TensorLocation, replicatedTensorSharding));
  }
  {
    py::class_<AutomaticLossScalingSettings> cls(
        m, "AutomaticLossScalingSettings");
    cls.def(py::init<>());
    cls.def(py::init<bool,
                     const nonstd::optional<std::vector<TensorId>> &,
                     float,
                     float>(),
            py::arg("enabled"),
            py::arg("toTrackTensors") =
                nonstd::optional<std::vector<TensorId>>(),
            py::arg("binEdgeLocation")               = 1.0,
            py::arg("thresholdUpperCountProportion") = 1e-7);
    cls.def_readwrite("enabled", &AutomaticLossScalingSettings::enabled);
    cls.def_readwrite("binEdgeLocation",
                      &AutomaticLossScalingSettings::binEdgeLocation);
    cls.def_readwrite(
        "thresholdUpperCountProportion",
        &AutomaticLossScalingSettings::thresholdUpperCountProportion);
    cls.def_readwrite("toTrackTensors",
                      &AutomaticLossScalingSettings::toTrackTensors);
  }
  {
    py::class_<TensorLocationSettings> cls(m, "TensorLocationSettings");
    cls.def(py::init<>());
    cls.def(py::init<TensorLocation, int, int>(),
            py::arg("location"),
            py::arg("minElementsForOffChip")                  = 2,
            py::arg("minElementsForReplicatedTensorSharding") = 8192);
    cls.def(py::init<TensorStorage, int, int>(),
            py::arg("storage"),
            py::arg("minElementsForOffChip")                  = 2,
            py::arg("minElementsForReplicatedTensorSharding") = 8192);
    cls.def_readwrite("location",
                      &TensorLocationSettings::location,
                      DOC(popart, TensorLocationSettings, location));
    cls.def_readwrite(
        "minElementsForOffChip",
        &TensorLocationSettings::minElementsForOffChip,
        DOC(popart, TensorLocationSettings, minElementsForOffChip));
    cls.def_readwrite(
        "minElementsForReplicatedTensorSharding",
        &TensorLocationSettings::minElementsForReplicatedTensorSharding,
        DOC(popart,
            TensorLocationSettings,
            minElementsForReplicatedTensorSharding));
  }
  {
    // This setting is experimental and may change.
    py::enum_<BatchSerializationTransformContext> en(
        m,
        "BatchSerializationTransformContext",
        DOC(popart, BatchSerializationTransformContext));
    en.value("Forward",
             BatchSerializationTransformContext::Fwd,
             DOC(popart, BatchSerializationTransformContext, Fwd));
    en.value("Backward",
             BatchSerializationTransformContext::Bwd,
             DOC(popart, BatchSerializationTransformContext, Bwd));
    en.value("Fwd",
             BatchSerializationTransformContext::Fwd,
             DOC(popart, BatchSerializationTransformContext, Fwd));
    en.value("Bwd",
             BatchSerializationTransformContext::Bwd,
             DOC(popart, BatchSerializationTransformContext, Bwd));
  }
  {
    // This setting is experimental and may change.
    py::enum_<BatchSerializationMethod> en(
        m, "BatchSerializationMethod", DOC(popart, BatchSerializationMethod));
    en.value("UnrollDynamic",
             BatchSerializationMethod::UnrollDynamic,
             DOC(popart, BatchSerializationMethod, UnrollDynamic));
    en.value("UnrollStatic",
             BatchSerializationMethod::UnrollStatic,
             DOC(popart, BatchSerializationMethod, UnrollStatic));
    en.value("Loop",
             BatchSerializationMethod::Loop,
             DOC(popart, BatchSerializationMethod, Loop));
  }
  {
    // This setting is experimental and may change.
    py::enum_<BatchSerializationBatchSchedule> en(
        m,
        "BatchSerializationBatchSchedule",
        DOC(popart, BatchSerializationBatchSchedule));
    en.value("Scheduler",
             BatchSerializationBatchSchedule::Scheduler,
             DOC(popart, BatchSerializationBatchSchedule, Scheduler));
    en.value("Isomorphic",
             BatchSerializationBatchSchedule::Isomorphic,
             DOC(popart, BatchSerializationBatchSchedule, Isomorphic));
    en.value("OverlapOnIo",
             BatchSerializationBatchSchedule::OverlapOnIo,
             DOC(popart, BatchSerializationBatchSchedule, OverlapOnIo));
    en.value("OverlapOnCompute",
             BatchSerializationBatchSchedule::OverlapOnCompute,
             DOC(popart, BatchSerializationBatchSchedule, OverlapOnCompute));
  }
  {
    py::class_<BatchSerializationSettings> cls(
        m,
        "BatchSerializationSettings",
        DOC(popart, BatchSerializationSettings));
    cls.def(py::init<>());
    cls.def(
        py::init<int,
                 bool,
                 bool,
                 bool,
                 BatchSerializationTransformContext,
                 BatchSerializationMethod,
                 BatchSerializationBatchSchedule>(),
        py::arg("factor"),
        py::arg("concatOnVirtualGraphChange"),
        py::arg("concatOnExecutionPhaseChange"),
        py::arg("concatOnPipelineStageChange"),
        py::arg("transformContext") = BatchSerializationTransformContext::Fwd,
        py::arg("method")           = BatchSerializationMethod::UnrollDynamic,
        py::arg("batchSchedule") = BatchSerializationBatchSchedule::Isomorphic);
    cls.def_readwrite("factor",
                      &BatchSerializationSettings::factor,
                      DOC(popart, BatchSerializationSettings, factor));
    cls.def_readwrite(
        "concatOnVirtualGraphChange",
        &BatchSerializationSettings::concatOnVirtualGraphChange,
        DOC(popart, BatchSerializationSettings, concatOnVirtualGraphChange));
    cls.def_readwrite(
        "concatOnExecutionPhaseChange",
        &BatchSerializationSettings::concatOnExecutionPhaseChange,
        DOC(popart, BatchSerializationSettings, concatOnExecutionPhaseChange));
    cls.def_readwrite(
        "concatOnPipelineStageChange",
        &BatchSerializationSettings::concatOnPipelineStageChange,
        DOC(popart, BatchSerializationSettings, concatOnPipelineStageChange));
    cls.def_readwrite(
        "transformContext",
        &BatchSerializationSettings::transformContext,
        DOC(popart, BatchSerializationSettings, transformContext));
    cls.def_readwrite("method",
                      &BatchSerializationSettings::method,
                      DOC(popart, BatchSerializationSettings, method));
    // This setting is experimental and may change.
    cls.def_readwrite("batchSchedule",
                      &BatchSerializationSettings::batchSchedule,
                      DOC(popart, BatchSerializationSettings, batchSchedule));
  }
  {
    // This setting is experimental and may change.
    py::enum_<AutodiffStitchStrategy> en(m,
                                         "AutodiffStitchStrategy" //,
                                         // DOC(popart, AutodiffStitchStrategy)
    );
    en.value("RecomputeMinimal",
             AutodiffStitchStrategy::RecomputeMinimal //,
             // DOC(popart, AutodiffStitchStrategy, RecomputeMinimal)
    );
    en.value("RecomputeAllNonInputs",
             AutodiffStitchStrategy::RecomputeAllNonInputs //,
             // DOC(popart, AutodiffStitchStrategy, RecomputeAllNonInputs)
    );
    en.value("AddFwdOutputs",
             AutodiffStitchStrategy::AddFwdOutputs //,
             // DOC(popart, AutodiffStitchStrategy, AddFwdOutputs)
    );
    en.value("SafeAddFwdOutputs",
             AutodiffStitchStrategy::SafeAddFwdOutputs //,
             // DOC(popart, AutodiffStitchStrategy, SafeAddFwdOutputs)
    );
  }
  {
    py::class_<AutodiffSettings> cls(m,
                                     "AutodiffSettings" //,
                                     // DOC(popart, AutodiffSettings)
    );
    cls.def(py::init<>());
    cls.def(py::init<AutodiffStitchStrategy>(),
            py::arg("stitchStrategy") =
                AutodiffStitchStrategy::RecomputeAllNonInputs);
    cls.def_readwrite("stitchStrategy",
                      &AutodiffSettings::stitchStrategy //,
                      // DOC(popart, AutodiffSettings, stitchStrategy)
    );
  }
  {
    py::class_<ExecutionPhaseSettings> cls(m, "ExecutionPhaseSettings");
    cls.def(py::init<>());
    cls.def(py::init<int,
                     int,
                     ExecutionPhaseIOSchedule,
                     ExecutionPhaseIOSchedule,
                     ExecutionPhaseIOSchedule,
                     ExecutionPhaseIOSchedule,
                     ExecutionPhaseSchedule>(),
            py::arg("phases"),
            py::arg("stages"),
            py::arg("weightIOSchedule"),
            py::arg("activationIOSchedule"),
            py::arg("optimizerStateIOSchedule"),
            py::arg("accumulatorIOSchedule"),
            py::arg("optimizerSchedule"));
    cls.def_readwrite("phases",
                      &ExecutionPhaseSettings::phases,
                      DOC(popart, ExecutionPhaseSettings, phases));
    cls.def_readwrite("stages",
                      &ExecutionPhaseSettings::stages,
                      DOC(popart, ExecutionPhaseSettings, stages));
    cls.def_readwrite("weightIOSchedule",
                      &ExecutionPhaseSettings::weightIOSchedule,
                      DOC(popart, ExecutionPhaseSettings, weightIOSchedule));
    cls.def_readwrite(
        "activationIOSchedule",
        &ExecutionPhaseSettings::activationIOSchedule,
        DOC(popart, ExecutionPhaseSettings, activationIOSchedule));
    cls.def_readwrite(
        "optimizerStateIOSchedule",
        &ExecutionPhaseSettings::optimizerStateIOSchedule,
        DOC(popart, ExecutionPhaseSettings, optimizerStateIOSchedule));
    cls.def_readwrite(
        "accumulatorIOSchedule",
        &ExecutionPhaseSettings::accumulatorIOSchedule,
        DOC(popart, ExecutionPhaseSettings, accumulatorIOSchedule));
    cls.def_readwrite("schedule",
                      &ExecutionPhaseSettings::schedule,
                      DOC(popart, ExecutionPhaseSettings, schedule));
  }
  {
    py::class_<AccumulateOuterFragmentSettings> cls(
        m, "AccumulateOuterFragmentSettings");
    cls.def(py::init<>());
    cls.def(py::init<AccumulateOuterFragmentSchedule, std::vector<int>>(),
            py::arg("schedule"),
            py::arg("excludedVirtualGraphs") = std::vector<int>());
    cls.def_readwrite("schedule",
                      &AccumulateOuterFragmentSettings::schedule,
                      DOC(popart, AccumulateOuterFragmentSettings, schedule));
    cls.def_readwrite(
        "excludedVirtualGraphs",
        &AccumulateOuterFragmentSettings::excludedVirtualGraphs,
        DOC(popart, AccumulateOuterFragmentSettings, excludedVirtualGraphs));
  }
  {
    // This setting is experimental and may change.
    py::enum_<SubgraphCopyingStrategy> en(m, "SubgraphCopyingStrategy");
    en.value("OnEnterAndExit",
             SubgraphCopyingStrategy::OnEnterAndExit,
             DOC(popart, SubgraphCopyingStrategy, OnEnterAndExit));
    en.value("JustInTime",
             SubgraphCopyingStrategy::JustInTime,
             DOC(popart, SubgraphCopyingStrategy, JustInTime));
  }
  {
    py::class_<ClipNormSettings> cls(m, "ClipNormSettings");
    cls.def(py::init<std::vector<TensorId>, float>(),
            py::arg("weightIds"),
            py::arg("maxNorm"));
    cls.def_static("clipWeights", &ClipNormSettings::clipWeights);
    cls.def_static("clipAllWeights", &ClipNormSettings::clipAllWeights);
    cls.def_readwrite("weightIds",
                      &ClipNormSettings::weightIds,
                      DOC(popart, ClipNormSettings, weightIds));
    cls.def_readwrite("maxNorm",
                      &ClipNormSettings::maxNorm,
                      DOC(popart, ClipNormSettings, maxNorm));
  }
  {
    py::class_<SessionOptions::NumIOTiles> cls(m, "NumIOTiles");
    cls.def(py::init<>());
    cls.def(py::init<int>());
  }
  {
    py::class_<SessionOptions> cls(m, "SessionOptions");
    cls.def(py::init<>());
    cls.def_readwrite(
        "logDir", &SessionOptions::logDir, DOC(popart, SessionOptions, logDir));
    cls.def_readwrite(
        "exportPoplarComputationGraph",
        &SessionOptions::exportPoplarComputationGraph,
        DOC(popart, SessionOptions, exportPoplarComputationGraph));
    cls.def_readwrite("exportPoplarVertexGraph",
                      &SessionOptions::exportPoplarVertexGraph,
                      DOC(popart, SessionOptions, exportPoplarVertexGraph));
    cls.def_readwrite("syntheticDataMode",
                      &SessionOptions::syntheticDataMode,
                      DOC(popart, SessionOptions, syntheticDataMode));
    cls.def_readwrite(
        "instrumentWithHardwareCycleCounter",
        &SessionOptions::instrumentWithHardwareCycleCounter,
        DOC(popart, SessionOptions, instrumentWithHardwareCycleCounter));
    cls.def_readwrite("hardwareInstrumentations",
                      &SessionOptions::hardwareInstrumentations,
                      DOC(popart, SessionOptions, hardwareInstrumentations));
    cls.def_readwrite(
        "disableGradAccumulationTensorStreams",
        &SessionOptions::disableGradAccumulationTensorStreams,
        DOC(popart, SessionOptions, disableGradAccumulationTensorStreams));
    cls.def_readwrite("enableOutlining",
                      &SessionOptions::enableOutlining,
                      DOC(popart, SessionOptions, enableOutlining));
    cls.def_readwrite(
        "enableOutliningCopyCostPruning",
        &SessionOptions::enableOutliningCopyCostPruning,
        DOC(popart, SessionOptions, enableOutliningCopyCostPruning));
    cls.def_readwrite("outlineThreshold",
                      &SessionOptions::outlineThreshold,
                      DOC(popart, SessionOptions, outlineThreshold));
    cls.def_readwrite("outlineSequenceBreakCost",
                      &SessionOptions::outlineSequenceBreakCost,
                      DOC(popart, SessionOptions, outlineSequenceBreakCost));
    cls.def_readwrite("accumulationFactor",
                      &SessionOptions::accumulationFactor,
                      DOC(popart, SessionOptions, accumulationFactor));
    cls.def_readwrite("enableGradientAccumulation",
                      &SessionOptions::enableGradientAccumulation,
                      DOC(popart, SessionOptions, enableGradientAccumulation));
    cls.def_readwrite(
        "accumulationAndReplicationReductionType",
        &SessionOptions::accumulationAndReplicationReductionType,
        DOC(popart, SessionOptions, accumulationAndReplicationReductionType));
    cls.def_readwrite(
        "meanAccumulationAndReplicationReductionStrategy",
        &SessionOptions::meanAccumulationAndReplicationReductionStrategy
        /* T42271 DOC(popart, SessionOptions, meanAccumulationAndReplicationReductionStrategy)*/);
    cls.def_readwrite("enableNonStableSoftmax",
                      &SessionOptions::enableNonStableSoftmax,
                      DOC(popart, SessionOptions, enableNonStableSoftmax));
    cls.def_readwrite("enablePipelining",
                      &SessionOptions::enablePipelining,
                      DOC(popart, SessionOptions, enablePipelining));
    cls.def_readwrite("subgraphCopyingStrategy",
                      &SessionOptions::subgraphCopyingStrategy,
                      DOC(popart, SessionOptions, subgraphCopyingStrategy));
    cls.def_readwrite("autoRecomputation",
                      &SessionOptions::autoRecomputation,
                      DOC(popart, SessionOptions, autoRecomputation));
    cls.def_readwrite("mergeVarUpdate",
                      &SessionOptions::mergeVarUpdate,
                      DOC(popart, SessionOptions, mergeVarUpdate));
    cls.def_readwrite("mergeVarUpdateMemThreshold",
                      &SessionOptions::mergeVarUpdateMemThreshold,
                      DOC(popart, SessionOptions, mergeVarUpdateMemThreshold));
    cls.def_readwrite("rearrangeAnchorsOnHost",
                      &SessionOptions::rearrangeAnchorsOnHost,
                      DOC(popart, SessionOptions, rearrangeAnchorsOnHost));
    cls.def_readwrite("executionPhaseSettings",
                      &SessionOptions::executionPhaseSettings,
                      DOC(popart, SessionOptions, executionPhaseSettings));
    cls.def_property(
        "numIOTiles",
        [](const SessionOptions &s) -> int { return s.numIOTiles; },
        [](SessionOptions &s, int numIOTiles) -> void {
          s.numIOTiles = numIOTiles;
        });
    cls.def_readwrite("explicitRecomputation",
                      &SessionOptions::explicitRecomputation,
                      DOC(popart, SessionOptions, explicitRecomputation));
    cls.def_readwrite("explicitPipelining",
                      &SessionOptions::explicitPipelining);
    cls.def_readwrite("batchSerializationSettings",
                      &SessionOptions::batchSerializationSettings,
                      DOC(popart, SessionOptions, batchSerializationSettings));
    cls.def_readwrite("aliasZeroCopy",
                      &SessionOptions::aliasZeroCopy,
                      DOC(popart, SessionOptions, aliasZeroCopy));
    cls.def_readwrite("enablePrefetchDatastreams",
                      &SessionOptions::enablePrefetchDatastreams);
    cls.def_readwrite("defaultPrefetchBufferingDepth",
                      &SessionOptions::defaultPrefetchBufferingDepth);
    cls.def_readwrite("prefetchBufferingDepthMap",
                      &SessionOptions::prefetchBufferingDepthMap);
    cls.def_readwrite("virtualGraphMode", &SessionOptions::virtualGraphMode);
    cls.def_readwrite("enableReplicatedGraphs",
                      &SessionOptions::enableReplicatedGraphs,
                      DOC(popart, SessionOptions, enableReplicatedGraphs));
    cls.def_readwrite("replicatedGraphCount",
                      &SessionOptions::replicatedGraphCount,
                      DOC(popart, SessionOptions, replicatedGraphCount));
    cls.def_readwrite("compileEngine",
                      &SessionOptions::compileEngine,
                      DOC(popart, SessionOptions, compileEngine));
    cls.def_readwrite("_engineOptions",
                      &SessionOptions::engineOptions,
                      DOC(popart, SessionOptions, engineOptions));
    cls.def_readwrite("_convolutionOptions",
                      &SessionOptions::convolutionOptions,
                      DOC(popart, SessionOptions, convolutionOptions));
    cls.def_readwrite("_lstmOptions",
                      &SessionOptions::lstmOptions,
                      DOC(popart, SessionOptions, lstmOptions));
    cls.def_readwrite("_matmulOptions",
                      &SessionOptions::matmulOptions,
                      // TODO see T42271
                      "");
    cls.def_readwrite("_reportOptions",
                      &SessionOptions::reportOptions,
                      DOC(popart, SessionOptions, reportOptions));
    cls.def_readwrite("_gclOptions",
                      &SessionOptions::gclOptions,
                      DOC(popart, SessionOptions, gclOptions));
    cls.def_readwrite("dotOpNames",
                      &SessionOptions::dotOpNames,
                      DOC(popart, SessionOptions, dotOpNames));
    cls.def_readwrite("separateCallOpPdfs",
                      &SessionOptions::separateCallOpPdfs,
                      DOC(popart, SessionOptions, separateCallOpPdfs));
    cls.def_readwrite("finalDotOp",
                      &SessionOptions::finalDotOp,
                      DOC(popart, SessionOptions, finalDotOp));
    cls.def_readwrite("firstDotOp",
                      &SessionOptions::firstDotOp,
                      DOC(popart, SessionOptions, firstDotOp));
    cls.def_readwrite("constantWeights",
                      &SessionOptions::constantWeights,
                      DOC(popart, SessionOptions, constantWeights));
    cls.def_readwrite("cachePath",
                      &SessionOptions::cachePath,
                      DOC(popart, SessionOptions, cachePath));
    cls.def_readwrite("enableEngineCaching",
                      &SessionOptions::enableEngineCaching,
                      DOC(popart, SessionOptions, enableEngineCaching));
    cls.def_readwrite("enableFloatingPointChecks",
                      &SessionOptions::enableFloatingPointChecks,
                      DOC(popart, SessionOptions, enableFloatingPointChecks));
    cls.def_readwrite("enableStochasticRounding",
                      &SessionOptions::enableStochasticRounding,
                      DOC(popart, SessionOptions, enableStochasticRounding));
    cls.def_readwrite("enableFullyConnectedPass",
                      &SessionOptions::enableFullyConnectedPass,
                      DOC(popart, SessionOptions, enableFullyConnectedPass));
    cls.def_readwrite("enableGroupedMatmuls",
                      &SessionOptions::enableGroupedMatmuls,
                      DOC(popart, SessionOptions, enableGroupedMatmuls));
    cls.def_readwrite("partialsTypeMatMuls",
                      &SessionOptions::partialsTypeMatMuls,
                      DOC(popart, SessionOptions, partialsTypeMatMuls));
    cls.def_readwrite("enableStableNorm",
                      &SessionOptions::enableStableNorm,
                      DOC(popart, SessionOptions, enableStableNorm));
    cls.def_readwrite("groupNormStridedChannelGrouping",
                      &SessionOptions::groupNormStridedChannelGrouping);
    // set in python use the python set constructor, so something like
    // mySessionOptions.dotChecks = {popart.DotCheck.FINAL}
    cls.def_readwrite("dotChecks",
                      &SessionOptions::dotChecks,
                      DOC(popart, SessionOptions, dotChecks));
    cls.def_readwrite("customCodelets",
                      &SessionOptions::customCodelets,
                      DOC(popart, SessionOptions, customCodelets));
    cls.def_readwrite("customCodeletCompileFlags",
                      &SessionOptions::customCodeletCompileFlags,
                      DOC(popart, SessionOptions, customCodeletCompileFlags));
    cls.def_readwrite("hostAllReduce",
                      &SessionOptions::hostAllReduce,
                      DOC(popart, SessionOptions, hostAllReduce));
    cls.def_readwrite("hostWeightUpdate",
                      &SessionOptions::hostWeightUpdate,
                      DOC(popart, SessionOptions, hostWeightUpdate));
    cls.def_readwrite("hostAllReduceRemoteBuffer",
                      &SessionOptions::hostAllReduceRemoteBuffer,
                      DOC(popart, SessionOptions, hostAllReduceRemoteBuffer));
    cls.def_readwrite("hostWeightUpdate",
                      &SessionOptions::hostWeightUpdate,
                      DOC(popart, SessionOptions, hostWeightUpdate));

    cls.def_readwrite("kahnTieBreaker",
                      &SessionOptions::kahnTieBreaker,
                      DOC(popart, SessionOptions, kahnTieBreaker));
    cls.def_readwrite("timeLimitScheduler",
                      &SessionOptions::timeLimitScheduler,
                      DOC(popart, SessionOptions, timeLimitScheduler));
    cls.def_readwrite("swapLimitScheduler",
                      &SessionOptions::swapLimitScheduler,
                      DOC(popart, SessionOptions, swapLimitScheduler));
    cls.def_readwrite(
        "transitiveClosureOptimizationThreshold",
        &SessionOptions::transitiveClosureOptimizationThreshold
        // TODO(matthewha)
        // DOC(popart, SessionOptions, transitiveClosureOptimizationThreshold)
    );
    cls.def_readwrite("decomposeGradSum",
                      &SessionOptions::decomposeGradSum,
                      DOC(popart, SessionOptions, decomposeGradSum));
    cls.def_readwrite(
        "serializedPoprithmsShiftGraphsDir",
        &SessionOptions::serializedPoprithmsShiftGraphsDir,
        DOC(popart, SessionOptions, serializedPoprithmsShiftGraphsDir));
    // To be deprecated in favor of the Shift version.
    cls.def_readwrite(
        "serializedPoprithmsAnnealGraphsDir",
        &SessionOptions::serializedPoprithmsShiftGraphsDir,
        DOC(popart, SessionOptions, serializedPoprithmsShiftGraphsDir));
    cls.def_readwrite(
        "enableDistributedReplicatedGraphs",
        &SessionOptions::enableDistributedReplicatedGraphs,
        DOC(popart, SessionOptions, enableDistributedReplicatedGraphs));
    cls.def_readwrite("globalReplicationFactor",
                      &SessionOptions::globalReplicationFactor,
                      DOC(popart, SessionOptions, globalReplicationFactor));
    cls.def_readwrite("globalReplicaOffset",
                      &SessionOptions::globalReplicaOffset,
                      DOC(popart, SessionOptions, globalReplicaOffset));
    cls.def_readwrite("groupHostSync",
                      &SessionOptions::groupHostSync,
                      DOC(popart, SessionOptions, groupHostSync));
    cls.def_readwrite("strictOpVersions",
                      &SessionOptions::strictOpVersions,
                      DOC(popart, SessionOptions, strictOpVersions));
    cls.def_readwrite("opxAliasChecking",
                      &SessionOptions::opxAliasChecking,
                      DOC(popart, SessionOptions, opxAliasChecking));
    cls.def_readwrite("opxModifyChecking",
                      &SessionOptions::opxModifyChecking,
                      DOC(popart, SessionOptions, opxModifyChecking));
    cls.def_readwrite(
        "activationTensorLocationSettings",
        &SessionOptions::activationTensorLocationSettings,
        DOC(popart, SessionOptions, activationTensorLocationSettings));
    cls.def_readwrite(
        "weightTensorLocationSettings",
        &SessionOptions::weightTensorLocationSettings,
        DOC(popart, SessionOptions, weightTensorLocationSettings));
    cls.def_readwrite(
        "optimizerStateTensorLocationSettings",
        &SessionOptions::optimizerStateTensorLocationSettings,
        DOC(popart, SessionOptions, optimizerStateTensorLocationSettings));
    cls.def_readwrite(
        "accumulatorTensorLocationSettings",
        &SessionOptions::accumulatorTensorLocationSettings,
        DOC(popart, SessionOptions, accumulatorTensorLocationSettings));
    cls.def_readwrite(
        "_tensorLocationSettingsOverride",
        &SessionOptions::tensorLocationSettingsOverride,
        DOC(popart, SessionOptions, tensorLocationSettingsOverride));
    cls.def_readwrite(
        "accumulateOuterFragmentSettings",
        &SessionOptions::accumulateOuterFragmentSettings,
        DOC(popart, SessionOptions, accumulateOuterFragmentSettings));
    cls.def_readwrite(
        "enableLoadAndOffloadRNGState",
        &SessionOptions::enableLoadAndOffloadRNGState,
        DOC(popart, SessionOptions, enableLoadAndOffloadRNGState));
    cls.def_readwrite("automaticLossScalingSettings",
                      &SessionOptions::automaticLossScalingSettings);
    cls.def_readwrite("useHostCopyOps", &SessionOptions::useHostCopyOps);
    cls.def_readwrite(
        "enableSupportedDataTypeCasting",
        &SessionOptions::enableSupportedDataTypeCasting,
        DOC(popart, SessionOptions, enableSupportedDataTypeCasting));
    cls.def_readwrite("enableExplicitMainLoops",
                      &SessionOptions::enableExplicitMainLoops,
                      DOC(popart, SessionOptions, enableExplicitMainLoops));
    cls.def_readwrite("enableMergeExchange",
                      &SessionOptions::enableMergeExchange,
                      DOC(popart, SessionOptions, enableMergeExchange));
    cls.def_readwrite("delayVarUpdates",
                      &SessionOptions::delayVarUpdates,
                      DOC(popart, SessionOptions, delayVarUpdates));
    cls.def_readwrite(
        "scheduleNonWeightUpdateGradientConsumersEarly",
        &SessionOptions::scheduleNonWeightUpdateGradientConsumersEarly
        /*, TODO(T43312):
        DOC(popart,
        SessionOptions,
        scheduleNonWeightUpdateGradientConsumersEarly) */);
  }
  {
    py::enum_<PatternsLevel> en(m, "PatternsLevel", DOC(popart, PatternsLevel));
    en.value("All", PatternsLevel::All, DOC(popart, PatternsLevel, All));
    en.value(
        "Default", PatternsLevel::Default, DOC(popart, PatternsLevel, Default));
    en.value(
        "Minimal", PatternsLevel::Minimal, DOC(popart, PatternsLevel, Minimal));
    en.value("NoPatterns",
             PatternsLevel::NoPatterns,
             DOC(popart, PatternsLevel, NoPatterns));
  }
  {
    py::enum_<DotCheck> en(m, "DotCheck", DOC(popart, DotCheck));
    en.value("Fwd0", DotCheck::Fwd0, DOC(popart, DotCheck, Fwd0));
    en.value("Fwd1", DotCheck::Fwd1, DOC(popart, DotCheck, Fwd1));
    en.value("Bwd0", DotCheck::Bwd0, DOC(popart, DotCheck, Bwd0));
    en.value("PreAlias", DotCheck::PreAlias, DOC(popart, DotCheck, PreAlias));
    en.value("Final", DotCheck::Final, DOC(popart, DotCheck, Final));
  }
  {
    py::enum_<RecomputationType> en(
        m, "RecomputationType", DOC(popart, RecomputationType));
    en.value("NoRecompute",
             RecomputationType::None,
             DOC(popart, RecomputationType, None));
    en.value("Standard",
             RecomputationType::Standard,
             DOC(popart, RecomputationType, Standard));
    en.value("NormOnly",
             RecomputationType::NormOnly,
             DOC(popart, RecomputationType, NormOnly));
    en.value("Pipeline",
             RecomputationType::Pipeline,
             DOC(popart, RecomputationType, Pipeline));
  }
  {
    py::enum_<RecomputeType> en(m, "RecomputeType", DOC(popart, RecomputeType));
    en.value("Undefined",
             RecomputeType::Undefined,
             DOC(popart, RecomputeType, Undefined));
    en.value("Checkpoint",
             RecomputeType::Checkpoint,
             DOC(popart, RecomputeType, Checkpoint));
    en.value("Recompute",
             RecomputeType::Recompute,
             DOC(popart, RecomputeType, Recompute));
    en.value("Recomputed",
             RecomputeType::Recomputed,
             DOC(popart, RecomputeType, Recomputed));
  }
  {
    py::enum_<TensorStorage> en(m, "TensorStorage", DOC(popart, TensorStorage));
    en.value(
        "OnChip", TensorStorage::OnChip, DOC(popart, TensorStorage, OnChip));
    en.value(
        "OffChip", TensorStorage::OffChip, DOC(popart, TensorStorage, OffChip));
  }
  {
    py::enum_<ReplicatedTensorSharding> en(
        m, "ReplicatedTensorSharding", DOC(popart, ReplicatedTensorSharding));
    en.value("Off",
             ReplicatedTensorSharding::Off,
             DOC(popart, ReplicatedTensorSharding, Off));
    en.value("On",
             ReplicatedTensorSharding::On,
             DOC(popart, ReplicatedTensorSharding, On));
  }
  {
    py::enum_<ExecutionPhaseIOSchedule> en(
        m, "ExecutionPhaseIOSchedule", DOC(popart, ExecutionPhaseIOSchedule));
    en.value("Preload",
             ExecutionPhaseIOSchedule::Preload,
             DOC(popart, ExecutionPhaseIOSchedule, Preload));
    en.value("OnDemand",
             ExecutionPhaseIOSchedule::OnDemand,
             DOC(popart, ExecutionPhaseIOSchedule, OnDemand));
  }
  {
    py::enum_<ExecutionPhaseSchedule> en(
        m, "ExecutionPhaseSchedule", DOC(popart, ExecutionPhaseSchedule));
    en.value("Interleaving",
             ExecutionPhaseSchedule::Interleaving,
             DOC(popart, ExecutionPhaseSchedule, Interleaving));
    en.value("Batch",
             ExecutionPhaseSchedule::Batch,
             DOC(popart, ExecutionPhaseSchedule, Batch));
    en.value("BatchClusteredIO",
             ExecutionPhaseSchedule::BatchClusteredIO,
             DOC(popart, ExecutionPhaseSchedule, BatchClusteredIO));
  }
  {
    py::enum_<AccumulateOuterFragmentSchedule> en(
        m,
        "AccumulateOuterFragmentSchedule",
        DOC(popart, AccumulateOuterFragmentSchedule));
    en.value("Scheduler",
             AccumulateOuterFragmentSchedule::Scheduler,
             DOC(popart, AccumulateOuterFragmentSchedule, Scheduler));
    en.value("Serial",
             AccumulateOuterFragmentSchedule::Serial,
             DOC(popart, AccumulateOuterFragmentSchedule, Serial));
    en.value(
        "OverlapCycleOptimized",
        AccumulateOuterFragmentSchedule::OverlapCycleOptimized,
        DOC(popart, AccumulateOuterFragmentSchedule, OverlapCycleOptimized));
    en.value(
        "OverlapMemoryOptimized",
        AccumulateOuterFragmentSchedule::OverlapMemoryOptimized,
        DOC(popart, AccumulateOuterFragmentSchedule, OverlapMemoryOptimized));
  }
  {
    py::enum_<SyncPattern> en(m, "SyncPattern", DOC(popart, SyncPattern));
    en.value("Full", SyncPattern::Full, DOC(popart, SyncPattern, Full));
    en.value("SinglePipeline",
             SyncPattern::SinglePipeline,
             DOC(popart, SyncPattern, SinglePipeline));
    en.value("ReplicaAndLadder",
             SyncPattern::ReplicaAndLadder,
             DOC(popart, SyncPattern, ReplicaAndLadder));
    en.attr("__str__") = py::cpp_function(
        [](const SyncPattern &sp) {
          std::stringstream ss;
          ss << sp;
          return ss.str();
        },
        py::name("__str__"),
        py::is_method(en));
  }
  {
    py::enum_<MergeVarUpdateType> en(
        m, "MergeVarUpdateType", DOC(popart, MergeVarUpdateType));
    en.value(
        "Off", MergeVarUpdateType::None, DOC(popart, MergeVarUpdateType, None));
    en.value(
        "All", MergeVarUpdateType::All, DOC(popart, MergeVarUpdateType, All));
    en.value("AutoTight",
             MergeVarUpdateType::AutoTight,
             DOC(popart, MergeVarUpdateType, AutoTight));
    en.value("AutoLoose",
             MergeVarUpdateType::AutoLoose,
             DOC(popart, MergeVarUpdateType, AutoLoose));
  }
  {
    py::enum_<VirtualGraphMode> en(m, "VirtualGraphMode");
    en.value("Off", VirtualGraphMode::Off, DOC(popart, VirtualGraphMode, Off));
    en.value("Manual",
             VirtualGraphMode::Manual,
             DOC(popart, VirtualGraphMode, Manual));
    en.value(
        "Auto", VirtualGraphMode::Auto, DOC(popart, VirtualGraphMode, Auto));
    en.value("ExecutionPhases",
             VirtualGraphMode::ExecutionPhases,
             DOC(popart, VirtualGraphMode, ExecutionPhases));
  }
  {
    py::enum_<SyntheticDataMode> en(m, "SyntheticDataMode");
    en.value(
        "Off", SyntheticDataMode::Off, DOC(popart, SyntheticDataMode, Off));
    en.value("Zeros",
             SyntheticDataMode::Zeros,
             DOC(popart, SyntheticDataMode, Zeros));
    en.value("RandomNormal",
             SyntheticDataMode::RandomNormal,
             DOC(popart, SyntheticDataMode, RandomNormal));
  }
  {
    py::enum_<IrSerializationFormat> en(m, "IrSerializationFormat");
    en.value("JSON",
             IrSerializationFormat::JSON,
             DOC(popart, IrSerializationFormat, JSON));
  }
  {
    py::enum_<Instrumentation> en(m, "Instrumentation");
    en.value(
        "Outer", Instrumentation::Outer, DOC(popart, Instrumentation, Outer));
    en.value(
        "Inner", Instrumentation::Inner, DOC(popart, Instrumentation, Inner));
  }
  {
    py::enum_<PreAliasPatternType> en(m, "PreAliasPatternType");
    en.value("PreUniRepl",
             PreAliasPatternType::PreUniRepl,
             DOC(popart, Patterns, enablePreUniRepl));
    en.value("PostNRepl",
             PreAliasPatternType::PostNRepl,
             DOC(popart, Patterns, enablePostNRepl));
    en.value("SoftmaxGradDirect", PreAliasPatternType::SoftmaxGradDirect);
    en.value("NLLLWithSoftmaxGradDirect",
             PreAliasPatternType::NLLLWithSoftmaxGradDirect);
    en.value("OptoIdentity", PreAliasPatternType::OptoIdentity);
    en.value("SubtractArg1GradOp",
             PreAliasPatternType::SubtractArg1GradOp,
             DOC(popart, Patterns, enableSubtractArg1GradOp));
    en.value("MulArgGradOp",
             PreAliasPatternType::MulArgGradOp,
             DOC(popart, Patterns, enableMulArgGradOp));
    en.value("ReciprocalGradOp",
             PreAliasPatternType::ReciprocalGradOp,
             DOC(popart, Patterns, enableReciprocalGradOp));
    en.value("SinGradOp",
             PreAliasPatternType::SinGradOp,
             DOC(popart, Patterns, enableSinGradOp));
    en.value("CosGradOp",
             PreAliasPatternType::CosGradOp,
             DOC(popart, Patterns, enableCosGradOp));
    en.value("DivArg0GradOp",
             PreAliasPatternType::DivArg0GradOp,
             DOC(popart, Patterns, enableDivArg0GradOp));
    en.value("DivArg1GradOp",
             PreAliasPatternType::DivArg1GradOp,
             DOC(popart, Patterns, enableDivArg1GradOp));
    en.value("PowArg0GradOp",
             PreAliasPatternType::PowArg0GradOp,
             DOC(popart, Patterns, enablePowArg0GradOp));
    en.value("PowArg1GradOp",
             PreAliasPatternType::PowArg1GradOp,
             DOC(popart, Patterns, enablePowArg1GradOp));
    en.value("Atan2Arg0GradOp",
             PreAliasPatternType::Atan2Arg0GradOp,
             DOC(popart, Patterns, enableAtan2Arg0GradOp));
    en.value("Atan2Arg1GradOp",
             PreAliasPatternType::Atan2Arg1GradOp,
             DOC(popart, Patterns, enableAtan2Arg1GradOp));
    en.value("SqrtGradOp",
             PreAliasPatternType::SqrtGradOp,
             DOC(popart, Patterns, enableSqrtGradOp));
    en.value("ExpGradOp",
             PreAliasPatternType::ExpGradOp,
             DOC(popart, Patterns, enableExpGradOp));
    en.value("Expm1GradOp",
             PreAliasPatternType::Expm1GradOp,
             DOC(popart, Patterns, enableExpm1GradOp));
    en.value("NegativeOneScale",
             PreAliasPatternType::NegativeOneScale,
             DOC(popart, Patterns, enableNegativeOneScale));
    en.value("MatMulOp",
             PreAliasPatternType::MatMulOp,
             DOC(popart, Patterns, enableMatMulOp));
    en.value("MatMulLHSGradOp", PreAliasPatternType::MatMulLHSGradOp);
    en.value("MatMulRHSGradOp", PreAliasPatternType::MatMulRHSGradOp);
    en.value("Log1pGradOp",
             PreAliasPatternType::Log1pGradOp,
             DOC(popart, Patterns, enableLog1pGradOp));
    en.value("FmodArg0GradOp", PreAliasPatternType::FmodArg0GradOp);
  }
  {
    py::class_<Patterns> cls(m, "Patterns");
    cls.def(py::init<>());
    cls.def(py::init<PatternsLevel>(), py::arg("level"));
    cls.def(py::init<std::vector<PreAliasPatternType>>(), py::arg("types"));
    cls.def(py::init<std::vector<std::string>>(), py::arg("patterns"));
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
    cls.def("enableRuntimeAsserts", &Patterns::enableRuntimeAsserts);
  }
  {
    py::class_<OutOfMemoryError> cls(m, "OutOfMemoryError");
    cls.def(py::init<>());
    cls.def("__repr__", [](const OutOfMemoryError &err) {
      return "popart.OutOfMemoryError: " + err.what();
    });
    cls.def("__str__", &OutOfMemoryError::what);
    cls.def("isSuccessful", &OutOfMemoryError::isSuccessful);
    cls.def("getSummaryReport", &OutOfMemoryError::getSummaryReport);
    cls.def("getProfilePath", &OutOfMemoryError::getProfilePath);
  }
  {
    py::class_<InferenceSession> cls(m, "_InferenceSessionCore");
    cls.def(py::init(&InferenceSession::createFromOnnxModel),
            py::arg("model"),
            py::arg("dataFlow").none(),
            py::arg("deviceInfo"),
            py::arg("inputShapeInfo"),
            py::arg("userOptions"),
            py::arg("patterns"),
            py::arg("name") = "inference",
            DOC(popart, InferenceSession, createFromOnnxModel));
    cls.def(py::init(&InferenceSession::createFromIr),
            py::arg("ir"),
            py::arg("deviceInfo"),
            py::arg("name") = "inference");
    cls.def(
        "compileAndExport",
        [](InferenceSession &session,
           const std::string &filename,
           OutOfMemoryError *status) {
          try {
            session.compileAndExport(filename);
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("filename"),
        py::arg("err").none());
    cls.def(
        "loadExecutable",
        [](InferenceSession &session, const std::string &filename) {
          session.loadExecutableFromFile(filename);
        },
        py::arg("filename"),
        DOC(popart, Session, loadExecutableFromFile));
    cls.def(
        "prepareDevice",
        [](InferenceSession &session,
           bool loadEngine,
           OutOfMemoryError *status) {
          try {
            session.prepareDevice(loadEngine);
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("loadEngine") = true,
        py::arg("err").none(),
        DOC(popart, Session, prepareDevice));
    cls.def("setRandomSeed",
            &InferenceSession::setRandomSeed,
            py::arg("seedValue"),
            DOC(popart, Session, setRandomSeed));
    cls.def("getRNGState", &InferenceSession::getRNGState);
    cls.def("setRNGState", &InferenceSession::setRNGState, py::arg("rngValue"));

    cls.def("getCycleCount",
            &InferenceSession::getCycleCount,
            py::arg("id") = "",
            DOC(popart, Session, getCycleCount));
    cls.def("loadEngineAndConnectStreams",
            &InferenceSession::loadEngineAndConnectStreams,
            DOC(popart, Session, loadEngineAndConnectStreams));
    cls.def("weightsFromHost",
            &InferenceSession::weightsFromHost,
            DOC(popart, Session, weightsFromHost));
    cls.def("writeWeights",
            &InferenceSession::writeWeights,
            DOC(popart, Session, writeWeights));
    cls.def("run",
            &InferenceSession::run,
            py::arg("stepio"),
            py::arg("debugName") = "",
            DOC(popart, Session, run),
            py::call_guard<py::gil_scoped_release>());
    cls.def("modelToHost",
            &InferenceSession::modelToHost,
            DOC(popart, Session, modelToHost));
    cls.def("updateExternallySavedTensorLocations",
            &InferenceSession::updateExternallySavedTensorLocations,
            DOC(popart, Session, updateExternallySavedTensorLocations));
    cls.def(
        "getInfo", &InferenceSession::getInfo, DOC(popart, Session, getInfo));
    cls.def("getSummaryReport",
            &InferenceSession::getSummaryReport,
            py::arg("resetProfile") = true,
            DOC(popart, Session, getSummaryReport));
    cls.def("getSerializedGraph", [](const InferenceSession &session) {
      auto report = session.getSerializedGraph();
      return py::bytes(report);
    });
    cls.def("getReport",
            &InferenceSession::getReport,
            DOC(popart, Session, getReport));
    cls.def("resetHostWeights",
            &InferenceSession::resetHostWeights,
            py::arg("modelProtoOrFilename"),
            py::arg("ignoreWeightsInModelWithoutCorrespondingHostWeight") =
                false,
            DOC(popart, Session, resetHostWeights));
    // Special test method to write serialise ir for analysis
    cls.def("_serializeIr",
            &InferenceSession::serializeIr,
            py::arg("format"),
            DOC(popart, Session, serializeIr));
  }
  {
    py::class_<TrainingSession> cls(m, "_TrainingSessionCore");
    cls.def(py::init(&TrainingSession::createFromOnnxModel),
            py::arg("model"),
            py::arg("dataFlow").none(),
            py::arg("loss"),
            py::arg("optimizer"),
            py::arg("deviceInfo"),
            py::arg("inputShapeInfo"),
            py::arg("userOptions"),
            py::arg("patterns"),
            py::arg("name") = "training",
            DOC(popart, TrainingSession, createFromOnnxModel));
    cls.def(
        "compileAndExport",
        [](TrainingSession &session,
           const std::string &filename,
           OutOfMemoryError *status) {
          try {
            session.compileAndExport(filename);
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("filename"),
        py::arg("err").none());
    cls.def(
        "loadExecutable",
        [](InferenceSession &session, const std::string &filename) {
          session.loadExecutableFromFile(filename);
        },
        py::arg("filename"),
        DOC(popart, Session, loadExecutableFromFile));
    cls.def(
        "prepareDevice",
        [](TrainingSession &session,
           bool loadEngine,
           OutOfMemoryError *status) {
          try {
            session.prepareDevice(loadEngine);
          } catch (const popart::memory_allocation_err &e) {
            if (status != nullptr) {
              status->exception = e.clone();
            } else {
              // rethrow the exception
              throw;
            }
          }
        },
        py::arg("loadEngine") = true,
        py::arg("err").none(),
        DOC(popart, Session, prepareDevice));
    cls.def("setRandomSeed",
            &TrainingSession::setRandomSeed,
            py::arg("seedValue"),
            DOC(popart, Session, setRandomSeed));
    cls.def("getRNGState", &TrainingSession::getRNGState);
    cls.def("setRNGState", &TrainingSession::setRNGState, py::arg("rngValue"));
    cls.def("getCycleCount",
            &TrainingSession::getCycleCount,
            py::arg("id") = "",
            DOC(popart, Session, getCycleCount));
    cls.def("loadEngineAndConnectStreams",
            &TrainingSession::loadEngineAndConnectStreams,
            DOC(popart, Session, loadEngineAndConnectStreams));
    cls.def("weightsToHost",
            &TrainingSession::weightsToHost,
            DOC(popart, Session, weightsToHost));
    cls.def("weightsFromHost",
            &TrainingSession::weightsFromHost,
            DOC(popart, Session, weightsFromHost));
    cls.def("readWeights",
            &TrainingSession::readWeights,
            DOC(popart, Session, readWeights));
    cls.def("writeWeights",
            &TrainingSession::writeWeights,
            DOC(popart, Session, writeWeights));
    cls.def("updateOptimizerFromHost",
            static_cast<void (TrainingSession::*)(const Optimizer *)>(
                &TrainingSession::updateOptimizerFromHost),
            DOC(popart, TrainingSession, updateOptimizerFromHost));
    cls.def("run",
            &TrainingSession::run,
            py::arg("stepio"),
            py::arg("debugName") = "",
            DOC(popart, Session, run),
            py::call_guard<py::gil_scoped_release>());
    cls.def("modelToHost",
            &TrainingSession::modelToHost,
            DOC(popart, Session, modelToHost));
    cls.def("updateExternallySavedTensorLocations",
            &TrainingSession::updateExternallySavedTensorLocations,
            DOC(popart, Session, updateExternallySavedTensorLocations));
    cls.def(
        "getInfo", &TrainingSession::getInfo, DOC(popart, Session, getInfo));
    cls.def("getSummaryReport",
            &TrainingSession::getSummaryReport,
            py::arg("resetProfile") = true,
            DOC(popart, Session, getSummaryReport));
    cls.def("getReport",
            &TrainingSession::getReport,
            DOC(popart, Session, getReport));
    cls.def(
        "getSerializedGraph",
        [](const TrainingSession &session) {
          auto report = session.getSerializedGraph();
          return py::bytes(report);
        },
        DOC(popart, Session, getSerializedGraph));
    cls.def("resetHostWeights",
            &TrainingSession::resetHostWeights,
            py::arg("modelProtoOrFilename"),
            py::arg("ignoreWeightsInModelWithoutCorrespondingHostWeight") =
                false,
            DOC(popart, Session, resetHostWeights));
    // Special test method to write serialise ir for analysis
    cls.def("_serializeIr", &TrainingSession::serializeIr, py::arg("format"));
    // Accessor for internal objects
    cls.def("getIr", &TrainingSession::getIr);
    cls.def("getHostReduceStreamIds",
            &TrainingSession::getHostReduceStreamIds,
            DOC(popart, TrainingSession, getHostReduceStreamIds));
    cls.def("connectStreamToCallback",
            &TrainingSession::connectStreamToCallback,
            DOC(popart, Session, connectStreamToCallback));
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
    cls.def("convertUINT8ToINT32", &GraphTransformer::convertUINT8ToINT32);
    cls.def("convertUINT16ToINT32", &GraphTransformer::convertUINT16ToINT32);
    cls.def("convertINT8ToINT32", &GraphTransformer::convertINT8ToINT32);
    cls.def("convertINT16ToINT32", &GraphTransformer::convertINT16ToINT32);
    cls.def("convertINT64ToINT32",
            &GraphTransformer::convertINT64ToINT32,
            py::arg("clip") = false,
            DOC(popart, GraphTransformer, convertINT64ToINT32));
    cls.def("convertDoublesToFloats",
            &GraphTransformer::convertDoublesToFloats);
    cls.def("convertDoublesToHalfs", &GraphTransformer::convertDoublesToHalfs);
    cls.def("convertBFloats16ToFloat32",
            &GraphTransformer::convertBFloats16ToFloat32);
    cls.def("convertInitializersToConstants",
            &GraphTransformer::convertInitializersToConstants,
            py::arg("ids"),
            DOC(popart, GraphTransformer, convertInitializersToConstants));
    cls.def("convertAllFixedPointInitializersToConstants",
            &GraphTransformer::convertAllFixedPointInitializersToConstants);
    cls.def("saveInitializersExternally",
            &GraphTransformer::saveInitializersExternally,
            py::arg("ids"),
            py::arg("filename"),
            DOC(popart, GraphTransformer, saveInitializersExternally));
  }
  {
    py::class_<AiGraphcoreOpset1> cls(m, "AiGraphcoreOpset1");
    cls.def("copyvarupdate",
            &AiGraphcoreOpset1::copyvarupdate,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, copyvarupdate));
    cls.def("copyvarupdate",
            &AiGraphcoreOpset1::copyvarupdate,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, copyvarupdate));
    cls.def("groupnormalization",
            &AiGraphcoreOpset1::groupnormalization,
            py::arg("args"),
            py::arg("num_groups"),
            py::arg("epsilon")     = 1e-05f,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, groupnormalization));
    cls.def("groupnormalization",
            &AiGraphcoreOpset1::groupnormalization,
            py::arg("args"),
            py::arg("num_groups"),
            py::arg("epsilon")      = 1e-05f,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, groupnormalization));
    cls.def("printtensor",
            &AiGraphcoreOpset1::printtensor,
            py::arg("args"),
            py::arg("print_gradient") = 1,
            py::arg("debugPrefix")    = std::string(),
            py::arg("title")          = std::string(),
            DOC(popart, AiGraphcoreOpset1, printtensor));
    cls.def("printtensor",
            &AiGraphcoreOpset1::printtensor,
            py::arg("args"),
            py::arg("print_gradient") = 1,
            py::arg("debugContext")   = std::string(),
            py::arg("title")          = std::string(),
            DOC(popart, AiGraphcoreOpset1, printtensor));
    cls.def("nop",
            &AiGraphcoreOpset1::nop,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, nop));
    cls.def("nop",
            &AiGraphcoreOpset1::nop,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, nop));
    cls.def("scale",
            &AiGraphcoreOpset1::scale,
            py::arg("args"),
            py::arg("scale"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, scale));
    cls.def("scale",
            &AiGraphcoreOpset1::scale,
            py::arg("args"),
            py::arg("scale"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, scale));
    cls.def("scaledadd",
            &AiGraphcoreOpset1::scaledadd,
            py::arg("args"),
            py::arg("scale0")      = 1.0,
            py::arg("scale1")      = 1.0,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, scaledadd));
    cls.def("scaledadd",
            &AiGraphcoreOpset1::scaledadd,
            py::arg("args"),
            py::arg("scale0")       = 1.0,
            py::arg("scale1")       = 1.0,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, scaledadd));
    cls.def("lstm",
            &AiGraphcoreOpset1::lstm,
            py::arg("args"),
            py::arg("outputFullSequence") = 1,
            py::arg("debugPrefix")        = std::string(),
            DOC(popart, AiGraphcoreOpset1, lstm));
    cls.def("lstm",
            &AiGraphcoreOpset1::lstm,
            py::arg("args"),
            py::arg("outputFullSequence") = 1,
            py::arg("debugContext")       = std::string(),
            DOC(popart, AiGraphcoreOpset1, lstm));
    cls.def("subsample",
            &AiGraphcoreOpset1::subsample,
            py::arg("args"),
            py::arg("strides"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, subsample));
    cls.def("subsample",
            &AiGraphcoreOpset1::subsample,
            py::arg("args"),
            py::arg("strides"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, subsample));
    cls.def("gelu",
            &AiGraphcoreOpset1::gelu,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, gelu));
    cls.def("gelu",
            &AiGraphcoreOpset1::gelu,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, gelu));
    cls.def("detach",
            &AiGraphcoreOpset1::detach,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, detach));
    cls.def("detach",
            &AiGraphcoreOpset1::detach,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, detach));
    cls.def("round",
            &AiGraphcoreOpset1::round,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, round));
    cls.def("round",
            &AiGraphcoreOpset1::round,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, round));
    cls.def("depthtospace",
            &AiGraphcoreOpset1::depthtospace,
            py::arg("args"),
            py::arg("blocksize"),
            py::arg("mode"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, depthtospace));
    cls.def("depthtospace",
            &AiGraphcoreOpset1::depthtospace,
            py::arg("args"),
            py::arg("blocksize"),
            py::arg("mode"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, depthtospace));
    cls.def("init",
            py::overload_cast<Attributes::Ints,
                              Attributes::Int,
                              Attributes::Int,
                              Attributes::Int,
                              const DebugContext &>(&AiGraphcoreOpset1::init),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("init_type"),
            py::arg("batch_axis"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, init));
    cls.def("init",
            py::overload_cast<Attributes::Ints,
                              Attributes::Int,
                              Attributes::Int,
                              Attributes::Int,
                              const DebugContext &>(&AiGraphcoreOpset1::init),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("init_type"),
            py::arg("batch_axis"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, init));
    cls.def("init",
            py::overload_cast<Attributes::Ints,
                              Attributes::Int,
                              Attributes::Int,
                              const DebugContext &>(&AiGraphcoreOpset1::init),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("init_type"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, init_2));
    cls.def("init",
            py::overload_cast<Attributes::Ints,
                              Attributes::Int,
                              Attributes::Int,
                              const DebugContext &>(&AiGraphcoreOpset1::init),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("init_type"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, init_2));
    cls.def("dynamicslice",
            &AiGraphcoreOpset1::dynamicslice,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")   = 0,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicslice));
    cls.def("dynamicslice",
            &AiGraphcoreOpset1::dynamicslice,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")    = 0,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicslice));
    cls.def("dynamicupdate",
            &AiGraphcoreOpset1::dynamicupdate,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")   = 0,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicupdate));
    cls.def("dynamicupdate",
            &AiGraphcoreOpset1::dynamicupdate,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("noOverlap")    = 0,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicupdate));
    cls.def("dynamiczero",
            &AiGraphcoreOpset1::dynamiczero,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamiczero));
    cls.def("dynamiczero",
            &AiGraphcoreOpset1::dynamiczero,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamiczero));
    cls.def("dynamicadd",
            &AiGraphcoreOpset1::dynamicadd,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicadd));
    cls.def("dynamicadd",
            &AiGraphcoreOpset1::dynamicadd,
            py::arg("args"),
            py::arg("axes"),
            py::arg("sizes"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, dynamicadd));
    cls.def("sequenceslice",
            &AiGraphcoreOpset1::sequenceslice,
            py::arg("args"),
            py::arg("zeroUnused")   = 0,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, sequenceslice));
    cls.def("packedDataBlock",
            &AiGraphcoreOpset1::packedDataBlock,
            py::arg("args"),
            py::arg("maxSequenceLengths"),
            py::arg("resultSize"),
            py::arg("callbackBatchSize"),
            py::arg("callback"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, call));
    cls.def("call",
            &AiGraphcoreOpset1::call,
            py::arg("args"),
            py::arg("num_outputs"),
            py::arg("callee"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, call));
    cls.def("call",
            &AiGraphcoreOpset1::call,
            py::arg("args"),
            py::arg("num_outputs"),
            py::arg("callee"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, call));
    cls.def("fmod",
            &AiGraphcoreOpset1::fmod,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, fmod));
    cls.def("fmod",
            &AiGraphcoreOpset1::fmod,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, fmod));
    cls.def("replicatedallreduce",
            &AiGraphcoreOpset1::replicatedallreduce,
            py::arg("args"),
            py::arg("commGroup")   = py::none(),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, replicatedallreduce));
    cls.def("replicatedallreduce",
            &AiGraphcoreOpset1::replicatedallreduce,
            py::arg("args"),
            py::arg("commGroup")    = py::none(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, replicatedallreduce));
    cls.def("l1loss",
            &AiGraphcoreOpset1::l1loss,
            py::arg("args"),
            py::arg("lambda"),
            py::arg("reduction")   = ReductionType::Mean,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, l1loss));
    cls.def("l1loss",
            &AiGraphcoreOpset1::l1loss,
            py::arg("args"),
            py::arg("lambda"),
            py::arg("reduction")    = ReductionType::Mean,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, l1loss));
    cls.def("nllloss",
            &AiGraphcoreOpset1::nllloss,
            py::arg("args"),
            py::arg("reduction")             = ReductionType::Mean,
            py::arg("ignoreIndex")           = pybind11::none(),
            py::arg("inputIsLogProbability") = false,
            py::arg("debugPrefix")           = std::string(),
            DOC(popart, AiGraphcoreOpset1, nllloss));
    cls.def("nllloss",
            &AiGraphcoreOpset1::nllloss,
            py::arg("args"),
            py::arg("reduction")             = ReductionType::Mean,
            py::arg("ignoreIndex")           = pybind11::none(),
            py::arg("inputIsLogProbability") = false,
            py::arg("debugContext")          = std::string(),
            DOC(popart, AiGraphcoreOpset1, nllloss));
    cls.def("identityloss",
            &AiGraphcoreOpset1::identityloss,
            py::arg("args"),
            py::arg("reduction")   = ReductionType::Mean,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, identityloss));
    cls.def("identityloss",
            &AiGraphcoreOpset1::identityloss,
            py::arg("args"),
            py::arg("reduction")    = ReductionType::Mean,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, identityloss));
    cls.def("ctcloss",
            &AiGraphcoreOpset1::ctcloss,
            py::arg("args"),
            py::arg("reduction")    = ReductionType::Mean,
            py::arg("blank")        = 0,
            py::arg("outDataType")  = "UNDEFINED",
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, ctcloss));
    cls.def("multiconv",
            &AiGraphcoreOpset1::multiconv,
            py::arg("args"),
            py::arg("dilations")                  = py::list(),
            py::arg("inDilations")                = py::list(),
            py::arg("pads")                       = py::list(),
            py::arg("outPads")                    = py::list(),
            py::arg("strides")                    = py::list(),
            py::arg("availableMemoryProportions") = py::list(),
            py::arg("partialsTypes")              = py::list(),
            py::arg("planType")                   = pybind11::none(),
            py::arg("perConvReservedTiles")       = pybind11::none(),
            py::arg("cycleBackOff")               = pybind11::none(),
            py::arg("debugPrefix")                = std::string(),
            DOC(popart, AiGraphcoreOpset1, multiconv));
    cls.def("multiconv",
            &AiGraphcoreOpset1::multiconv,
            py::arg("args"),
            py::arg("dilations")                  = py::list(),
            py::arg("inDilations")                = py::list(),
            py::arg("pads")                       = py::list(),
            py::arg("outPads")                    = py::list(),
            py::arg("strides")                    = py::list(),
            py::arg("availableMemoryProportions") = py::list(),
            py::arg("partialsTypes")              = py::list(),
            py::arg("planType")                   = pybind11::none(),
            py::arg("perConvReservedTiles")       = pybind11::none(),
            py::arg("cycleBackOff")               = pybind11::none(),
            py::arg("debugContext")               = std::string(),
            DOC(popart, AiGraphcoreOpset1, multiconv));
    cls.def("shapeddropout",
            &AiGraphcoreOpset1::shapeddropout,
            py::arg("args"),
            py::arg("shape"),
            py::arg("ratio")       = 0.5f,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, shapeddropout));
    cls.def("shapeddropout",
            &AiGraphcoreOpset1::shapeddropout,
            py::arg("args"),
            py::arg("shape"),
            py::arg("ratio")        = 0.5f,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, shapeddropout));
    cls.def("atan2",
            &AiGraphcoreOpset1::atan2,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, atan2));
    cls.def("atan2",
            &AiGraphcoreOpset1::atan2,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, atan2));
    cls.def("expm1",
            &AiGraphcoreOpset1::expm1,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, expm1));
    cls.def("expm1",
            &AiGraphcoreOpset1::expm1,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, expm1));
    cls.def("log1p",
            &AiGraphcoreOpset1::log1p,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, log1p));
    cls.def("log1p",
            &AiGraphcoreOpset1::log1p,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, log1p));
    cls.def("reshape",
            &AiGraphcoreOpset1::reshape,
            py::arg("args"),
            py::arg("shape"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, reshape));
    cls.def("reshape",
            &AiGraphcoreOpset1::reshape,
            py::arg("args"),
            py::arg("shape"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, reshape));
    cls.def("remainder",
            &AiGraphcoreOpset1::remainder,
            py::arg("args"),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, remainder));
    cls.def("remainder",
            &AiGraphcoreOpset1::remainder,
            py::arg("args"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, remainder));
    cls.def("reverse",
            &AiGraphcoreOpset1::reverse,
            py::arg("args"),
            py::arg("dimensions"),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, reverse));
    cls.def("abort",
            &AiGraphcoreOpset1::abort,
            py::arg("args")        = pybind11::list(),
            py::arg("debugPrefix") = std::string(),
            DOC(popart, AiGraphcoreOpset1, abort));
    cls.def("abort",
            &AiGraphcoreOpset1::abort,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, abort));
    cls.def("bitwiseand",
            &AiGraphcoreOpset1::bitwiseand,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, bitwiseand));
    cls.def("bitwisenot",
            &AiGraphcoreOpset1::bitwisenot,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, bitwisenot));
    cls.def("bitwiseor",
            &AiGraphcoreOpset1::bitwiseor,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, bitwiseor));
    cls.def("bitwisexor",
            &AiGraphcoreOpset1::bitwisexor,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, bitwisexor));
    cls.def("bitwisexnor",
            &AiGraphcoreOpset1::bitwisexnor,
            py::arg("args")         = pybind11::list(),
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, bitwisexnor));
    cls.def("reducemedian",
            &AiGraphcoreOpset1::reducemedian,
            py::arg("args"),
            py::arg("axes")         = pybind11::none(),
            py::arg("keepdims")     = 1,
            py::arg("debugContext") = std::string(),
            DOC(popart, AiGraphcoreOpset1, reducemedian));
    cls.def("scatterreduce",
            &AiGraphcoreOpset1::scatterreduce,
            py::arg("args") = pybind11::list(),
            py::arg("axis_size"),
            py::arg("axis")         = -1,
            py::arg("reduction")    = ScatterReduction::Sum,
            py::arg("debugContext") = std::string());
    cls.def("ctcbeamsearchdecoder",
            &AiGraphcoreOpset1::ctcbeamsearchdecoder,
            py::arg("args"),
            py::arg("blank")         = 0,
            py::arg("beam_width")    = 100,
            py::arg("top_paths")     = 1,
            py::arg("debug_context") = std::string()
            /*, DOC(popart, AiGraphcoreOpset1, ctcbeamsearchdecoder)*/);
  }
  {
    py::class_<Builder> cls(m, "_BuilderCore");
    cls.def(py::init(&Builder::create));
    cls.def(py::init(&Builder::createFromOnnxModel),
            py::arg("modelProtoOrFilename"));
    cls.def("setGraphName",
            &Builder::setGraphName,
            py::arg("name"),
            DOC(popart, Builder, setGraphName));
    cls.def("setGraphName", &Builder::setGraphName, py::arg("name"));
    cls.def(
        "addInputTensor",
        [](Builder &b, const TensorInfo &ti, const popart::DebugContext &dc)
            -> TensorId { return b.addInputTensor(ti, dc); },
        py::arg("tensorInfo"),
        py::arg("debugPrefix") = std::string());
    cls.def("addInputTensor",
            py::overload_cast<const TensorInfo &, const popart::DebugContext &>(
                &Builder::addInputTensor),
            py::arg("tensorInfo"),
            py::arg("debugContext") = std::string());
    cls.def(
        "addInputTensor",
        [](Builder &b,
           const std::string &dataType,
           const Shape &shape,
           const popart::DebugContext &dc) -> TensorId {
          return b.addInputTensor(dataType, shape, dc);
        },
        py::arg("dataType"),
        py::arg("shape"),
        py::arg("debugPrefix") = std::string());
    cls.def(
        "addInputTensor",
        [](Builder &b,
           const std::string &dataType,
           const Shape &shape,
           const popart::DebugContext &dc) -> TensorId {
          return b.addInputTensor(dataType, shape, dc);
        },
        py::arg("dataType"),
        py::arg("shape"),
        py::arg("debugContext") = "");
    cls.def(
        "addInputTensor",
        [](Builder &b, const TensorInfo &ti, const popart::DebugContext &dc)
            -> TensorId { return b.addInputTensor(ti, dc); },
        py::arg("tensorInfo"),
        py::arg("debugPrefix") = std::string());
    cls.def("addInputTensor",
            py::overload_cast<const TensorInfo &,
                              const InputSettings &,
                              const popart::DebugContext &>(
                &Builder::addInputTensor),
            py::arg("tensorInfo"),
            py::arg("settings")     = InputSettings(),
            py::arg("debugContext") = std::string());
    cls.def(
        "addInputTensor",
        [](Builder &b,
           const std::string &dataType,
           const Shape &shape,
           const InputSettings &settings,
           const popart::DebugContext &dc) -> TensorId {
          return b.addInputTensor(dataType, shape, dc);
        },
        py::arg("dataType"),
        py::arg("shape"),
        py::arg("settings")    = InputSettings(),
        py::arg("debugPrefix") = std::string());
    cls.def(
        "addInputTensor",
        [](Builder &b,
           const std::string &dataType,
           const Shape &shape,
           const InputSettings &settings,
           const popart::DebugContext &dc) -> TensorId {
          return b.addInputTensor(dataType, shape, dc);
        },
        py::arg("dataType"),
        py::arg("shape"),
        py::arg("settings")     = InputSettings(),
        py::arg("debugContext") = "");
    cls.def("addUntypedInputTensor",
            &Builder::addUntypedInputTensor,
            py::arg("debugPrefix") = std::string(),
            DOC(popart, Builder, addUntypedInputTensor));
    cls.def("addUntypedInputTensor",
            &Builder::addUntypedInputTensor,
            py::arg("debugContext") = "",
            DOC(popart, Builder, addUntypedInputTensor));
    cls.def("addInputTensorFromParentGraph",
            &Builder::addInputTensorFromParentGraph,
            py::arg("tensorId"),
            DOC(popart, Builder, addInputTensorFromParentGraph));
    cls.def(
        "addInitializedInputTensor",
        [](Builder &builder, py::array array, const popart::DebugContext &dc) {
          array = makeContiguous(array);
          ConstVoidData initData;
          initData.data = array.request().ptr;
          initData.info = getTensorInfo(array);
          return builder.addInitializedInputTensor(initData, dc);
        },
        py::arg("initVal"),
        py::arg("debugPrefix") = std::string());
    cls.def(
        "addInitializedInputTensor",
        [](Builder &builder, py::array array, popart::DebugContext &dc) {
          array = makeContiguous(array);
          ConstVoidData initData;
          initData.data = array.request().ptr;
          initData.info = getTensorInfo(array);
          return builder.addInitializedInputTensor(initData, dc);
        },
        py::arg("initVal"),
        py::arg("debugContext") = std::string());
    cls.def(
        "addOutputTensor", &Builder::addOutputTensor, py::arg("outputName"));
    cls.def("_createSubgraphBuilder",
            &Builder::createSubgraphBuilder,
            pybind11::return_value_policy::reference,
            DOC(popart, Builder, createSubgraphBuilder));
    cls.def("saveModelProto",
            &Builder::saveModelProto,
            py::arg("filename"),
            DOC(popart, Builder, saveModelProto));
    cls.def("saveInitializersExternally",
            &Builder::saveInitializersExternally,
            py::arg("ids"),
            py::arg("filename"),
            DOC(popart, Builder, saveInitializersExternally));

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
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, nodeHasAttribute));
    cls.def("getInt64NodeAttribute",
            &Builder::getInt64NodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getInt64NodeAttribute));
    cls.def("getInt64VectorNodeAttribute",
            &Builder::getInt64VectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getInt64VectorNodeAttribute));
    cls.def("getFloatNodeAttribute",
            &Builder::getFloatNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getFloatNodeAttribute));
    cls.def("getFloatVectorNodeAttribute",
            &Builder::getFloatVectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getFloatVectorNodeAttribute));
    cls.def("getStringNodeAttribute",
            &Builder::getStringNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getStringNodeAttribute));
    cls.def("getStringVectorNodeAttribute",
            &Builder::getStringVectorNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getStringVectorNodeAttribute));
    cls.def("removeNodeAttribute",
            &Builder::removeNodeAttribute,
            py::arg("attributeName"),
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, removeNodeAttribute));
    cls.def("getAllNodeAttributeNames",
            &Builder::getAllNodeAttributeNames,
            py::arg("nodeOutputNames"),
            DOC(popart, Builder, getAllNodeAttributeNames));
    cls.def("getModelProto", [](const Builder &builder) {
      return py::bytes(builder.getModelProto());
    });
    cls.def("getInputTensorIds", &Builder::getInputTensorIds);
    cls.def("getOutputTensorIds", &Builder::getOutputTensorIds);
    cls.def("getValueTensorIds", &Builder::getValueTensorIds);
    cls.def("getTrainableTensorIds", &Builder::getTrainableTensorIds);
    cls.def("getTensorShape",
            &Builder::getTensorShape,
            py::arg("id"),
            DOC(popart, Builder, getTensorShape));
    cls.def(
        "getTensorDtypeString", &Builder::getTensorDtypeString, py::arg("id"));
    cls.def("isInitializer",
            &Builder::isInitializer,
            py::arg("id"),
            DOC(popart, Builder, isInitializer));
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
    cls.def("executionPhase",
            static_cast<void (Builder::*)(const TensorId &, int64_t phase)>(
                &Builder::executionPhase),
            py::arg("nodeOutputNames"),
            py::arg("value") = 0);
    cls.def(
        "executionPhase",
        static_cast<void (Builder::*)(const std::set<TensorId> &,
                                      int64_t phase)>(&Builder::executionPhase),
        py::arg("nodeOutputNames"),
        py::arg("value") = 0);
    cls.def(
        "executionPhase",
        [](Builder &self, int64_t phase) -> AttributeContextManager {
          AttributeContextManager acm(self, sExecutionPhaseAttribute, phase);
          return acm;
        },
        py::arg("value") = 0);
    cls.def("outlineAttributes",
            [](Builder &self, py::dict pyd) -> KeyValueContextManager {
              KeyValueContextManager kvcm(
                  self, sOutlineAttribute, getDictionary(pyd));
              return kvcm;
            });
    cls.def(
        "getExecutionPhase",
        static_cast<int64_t (Builder::*)() const>(&Builder::getExecutionPhase));
    cls.def("hasExecutionPhase", [](Builder &self) -> bool {
      return self.hasAttribute(sExecutionPhaseAttribute);
    });
    cls.def(
        "recomputeOutput",
        static_cast<void (Builder::*)(const TensorId &, RecomputeType value)>(
            &Builder::recomputeOutput),
        py::arg("nodeOutputNames"),
        py::arg("value") = RecomputeType::Undefined);
    cls.def(
        "recomputeOutput",
        [](Builder &self, RecomputeType value) -> AttributeContextManager {
          AttributeContextManager acm(
              self, sRecomputeOutputAttribute, static_cast<int64_t>(value));
          return acm;
        },
        py::arg("value") = RecomputeType::Undefined);
    cls.def("checkpointOutput",
            &Builder::checkpointOutput,
            py::arg("nodeOutputNames"));
    cls.def(
        "outputTensorLocation",
        static_cast<void (Builder::*)(const TensorId &, TensorLocation value)>(
            &Builder::outputTensorLocation),
        py::arg("nodeOutputNames"),
        py::arg("value") = TensorLocation());
    cls.def(
        "outputTensorLocation",
        [](Builder &self, TensorLocation value) -> AttributeContextManager {
          AttributeContextManager acm(
              self, sOutputTensorLocationAttribute, value.serialize());
          return acm;
        },
        py::arg("value") = TensorLocation());
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
    cls.def(
        "commGroup",
        [](Builder &self,
           int64_t type,
           int64_t groupSize) -> AttributeContextManager {
          AttributeContextManager acm(self,
                                      sCollectiveCommGroup,
                                      std::vector<int64_t>{type, groupSize});
          return acm;
        },
        py::arg("type")      = 0,
        py::arg("groupSize") = 0);

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
            py::arg("partialsType"),
            DOC(popart, Builder, setPartialsType));
    cls.def("getPartialsType",
            &Builder::getPartialsType,
            py::arg("nodeOutputName"),
            DOC(popart, Builder, getPartialsType));
    cls.def("setAvailableMemoryProportion",
            &Builder::setAvailableMemoryProportion,
            py::arg("nodeOutputName"),
            py::arg("availableMemoryProportion"),
            DOC(popart, Builder, setAvailableMemoryProportion));
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
        py::arg("value") = RecomputeType::Recompute);
    cls.def("recomputeOutputInBackwardPass",
            static_cast<void (Builder::*)(const std::set<TensorId> &,
                                          RecomputeType value)>(
                &Builder::recomputeOutputInBackwardPass),
            py::arg("nodeOutputNames"),
            py::arg("value") = RecomputeType::Recompute);

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
    py::class_<KeyValueContextManager> cls(m, "KeyValueContextManager");
    cls.def("__enter__", &KeyValueContextManager::enter);
    cls.def("__exit__",
            [](KeyValueContextManager &self,
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
    en.value(
        "IpuModel", DeviceType::IpuModel, DOC(popart, DeviceType, IpuModel));
    en.value("Cpu", DeviceType::Cpu, DOC(popart, DeviceType, Cpu));
    en.value("Ipu", DeviceType::Ipu, DOC(popart, DeviceType, Ipu));
    en.value("Sim", DeviceType::Sim, DOC(popart, DeviceType, Sim));
  }
  {
    py::enum_<DeviceConnectionType> en(m, "DeviceConnectionType");
    en.value("Always",
             DeviceConnectionType::Always,
             DOC(popart, DeviceConnectionType, Always));
    en.value("OnDemand",
             DeviceConnectionType::OnDemand,
             DOC(popart, DeviceConnectionType, OnDemand));
    en.value("Never",
             DeviceConnectionType::Never,
             DOC(popart, DeviceConnectionType, Never));
  }
  {
    py::enum_<DeviceSelectionCriterion> en(m, "DeviceSelectionCriterion");
    en.value("First",
             DeviceSelectionCriterion::First,
             DOC(popart, DeviceSelectionCriterion, First));
    en.value("Random",
             DeviceSelectionCriterion::Random,
             DOC(popart, DeviceSelectionCriterion, Random));
  }
  {
    py::enum_<CommGroupType> en(m, "CommGroupType");
    en.value("All", CommGroupType::All, DOC(popart, CommGroupType, All));
    en.value("Consecutive",
             CommGroupType::Consecutive,
             DOC(popart, CommGroupType, Consecutive));
    en.value("Orthogonal",
             CommGroupType::Orthogonal,
             DOC(popart, CommGroupType, Orthogonal));
  }

  {
    // PyBinding to a singleton
    py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>> cls(
        m, "DeviceManager");
    cls.def(py::init([]() {
      return std::unique_ptr<DeviceManager, py::nodelete>(
          &DeviceManager::createDeviceManager());
    }));
    cls.def(
        "acquireAvailableDevice",
        static_cast<std::shared_ptr<DeviceInfo> (DeviceManager::*)(
            int,
            int,
            SyncPattern,
            DeviceConnectionType,
            DeviceSelectionCriterion)>(&DeviceManager::acquireAvailableDevice),
        py::arg("numIpus")            = 1,
        py::arg("tilesPerIpu")        = 0,
        py::arg("pattern")            = SyncPattern::Full,
        py::arg("connectionType")     = DeviceConnectionType::Always,
        py::arg("selectionCriterion") = DeviceSelectionCriterion::First);
    cls.def("acquireDeviceById",
            &DeviceManager::acquireDeviceById,
            py::arg("id"),
            py::arg("pattern")        = SyncPattern::Full,
            py::arg("connectionType") = DeviceConnectionType::Always,
            DOC(popart, DeviceManager, acquireDeviceById));
    cls.def("createCpuDevice", &DeviceManager::createCpuDevice);
    cls.def("createIpuModelDevice", [](DeviceManager &dm, py::dict e) {
      std::map<std::string, std::string> options = getDictionary(e);
      return dm.createIpuModelDevice(options);
    });
    cls.def("createSimDevice", [](DeviceManager &dm, py::dict e) {
      std::map<std::string, std::string> options = getDictionary(e);
      return dm.createSimDevice(options);
    });
    cls.def(
        "createOfflineIPUDevice",
        [](DeviceManager &dm, py::dict e) {
          std::map<std::string, std::string> options = getDictionary(e);
          return dm.createOfflineIPUDevice(options);
        },
        py::arg("opts"));
    cls.def("enumerateDevices",
            &DeviceManager::enumerateDevices,
            py::arg("pattern")        = SyncPattern::Full,
            py::arg("numIpus")        = 1,
            py::arg("deviceType")     = DeviceType::Ipu,
            py::arg("connectionType") = DeviceConnectionType::Always,
            py::arg("tilesPerIPU")    = 0,
            DOC(popart, DeviceManager, enumerateDevices));
    cls.def("setOnDemandAttachTimeout",
            &DeviceManager::setOnDemandAttachTimeout,
            py::arg("attachTimeout"),
            DOC(popart, DeviceManager, setOnDemandAttachTimeout));
  }
  {
    py::class_<DeviceInfo, std::shared_ptr<DeviceInfo>> cls(m, "DeviceInfo");
    cls.def("attach", &DeviceInfo::attach);
    cls.def("tryAttachUntilTimeout", &DeviceInfo::attach);
    cls.def("detach", &DeviceInfo::detach);
    cls.def_property_readonly("type", &DeviceInfo::getType);
    cls.def_property_readonly("connectionType", &DeviceInfo::getConnectionType);
    cls.def_property_readonly("version", &DeviceInfo::getVersion);
    cls.def_property_readonly("id", &DeviceInfo::getId);
    cls.def_property_readonly("numIpus", &DeviceInfo::getNumIpus);
    cls.def_property_readonly("tilesPerIPU", &DeviceInfo::getTilesPerIPU);
    cls.def_property_readonly("tilesPerIpu", &DeviceInfo::getTilesPerIPU);
    cls.def_property_readonly("driverIds", &DeviceInfo::getDriverIds);
    cls.def_property_readonly("isAttached", &DeviceInfo::isAttached);
    cls.def("__enter__", [&](DeviceInfo &r) -> DeviceInfo & { return r; });
    cls.def("__exit__",
            [&](DeviceInfo &r,
                pybind11::object exc_type,
                pybind11::object exc_value,
                pybind11::object traceback) { r.detach(); });

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

  m.def("reservedAccumPrefix", &reservedAccumPrefix);
  m.def("reservedAcclPrefix", &reservedAcclPrefix);
  m.def("reservedAccl1Prefix", &reservedAccl1Prefix);
  m.def("reservedAccl2Prefix", &reservedAccl2Prefix);
  m.def("reservedAccl3Prefix", &reservedAccl3Prefix);
  m.def("reservedStepPrefix", &reservedStepPrefix);
  m.def("reservedAcclToReducePrefix", &reservedAcclToReducePrefix);
  m.def("reservedAcclToUpdatePrefix", &reservedAcclToUpdatePrefix);
  m.def("reservedAcclFinalOutPrefix", &reservedAcclFinalOutPrefix);

  m.def("reservedAdamUpdaterPrefix", &reservedAdamUpdaterPrefix);
  m.def("reservedLambR1SqPrefix", &reservedLambR1SqPrefix);
  m.def("reservedLambR2SqPrefix", &reservedLambR2SqPrefix);

  m.def("reservedStashedPrefix", &reservedStashedPrefix);
  m.def("reservedRestoredPrefix", &reservedRestoredPrefix);
  m.def("reservedLossScalingPrefix", &reservedLossScalingPrefix);
  m.def("reservedRandomSeedPrefix", &reservedRandomSeedPrefix);
  m.def("reservedSeedModifierPrefix", &reservedSeedModifierPrefix);

  m.def("reservedRemoteArgPrefix", &reservedRemoteArgPrefix);

  m.def("reservedDefaultWeightDecayScaleFactor0Prefix",
        &reservedDefaultWeightDecayScaleFactor0Prefix);
  m.def("reservedSpecificWeightDecayScaleFactor0Prefix",
        &reservedSpecificWeightDecayScaleFactor0Prefix);
  m.def("reservedDefaultScaledLearningRate0Prefix",
        &reservedDefaultScaledLearningRate0Prefix);
  m.def("reservedSpecificScaledLearningRate0Prefix",
        &reservedSpecificScaledLearningRate0Prefix);

  m.def("reservedDefaultScaledWeightDecay1Prefix",
        &reservedDefaultScaledWeightDecay1Prefix);
  m.def("reservedSpecificScaledWeightDecay1Prefix",
        &reservedSpecificScaledWeightDecay1Prefix);
  m.def("reservedDefaultScaledLearningRate1Prefix",
        &reservedDefaultScaledLearningRate1Prefix);
  m.def("reservedSpecificScaledLearningRate1Prefix",
        &reservedSpecificScaledLearningRate1Prefix);
  m.def("reservedDefaultDampeningScaleFactor1Prefix",
        &reservedDefaultDampeningScaleFactor1Prefix);
  m.def("reservedSpecificDampeningScaleFactor1Prefix",
        &reservedSpecificDampeningScaleFactor1Prefix);
  m.def("reservedDefaultScaledMomentum1Prefix",
        &reservedDefaultScaledMomentum1Prefix);
  m.def("reservedSpecificScaledMomentum1Prefix",
        &reservedSpecificScaledMomentum1Prefix);
  m.def("reservedDefaultLearningRatePrefix",
        &reservedDefaultLearningRatePrefix);
  m.def("reservedSpecificLearningRatePrefix",
        &reservedSpecificLearningRatePrefix);
  m.def("reservedDefaultWeightDecayPrefix", &reservedDefaultWeightDecayPrefix);
  m.def("reservedSpecificWeightDecayPrefix",
        &reservedSpecificWeightDecayPrefix);
  m.def("reservedDefaultLossScalingPrefix", &reservedDefaultLossScalingPrefix);
  m.def("reservedSpecificLossScalingPrefix",
        &reservedSpecificLossScalingPrefix);
  m.def("reservedDefaultMaxWeightNormPrefix",
        &reservedDefaultMaxWeightNormPrefix);
  m.def("reservedSpecificMaxWeightNormPrefix",
        &reservedSpecificMaxWeightNormPrefix);
  m.def("reservedAutomaticLossScalePrefix", &reservedAutomaticLossScalePrefix);
  m.def("reservedDefaultAdamBeta1Prefix", &reservedDefaultAdamBeta1Prefix);
  m.def("reservedSpecificAdamBeta1Prefix", &reservedSpecificAdamBeta1Prefix);
  m.def("reservedDefaultAdamBeta2Prefix", &reservedDefaultAdamBeta2Prefix);
  m.def("reservedSpecificAdamBeta2Prefix", &reservedSpecificAdamBeta2Prefix);
  m.def("reservedDefaultAdamEpsPrefix", &reservedDefaultAdamEpsPrefix);
  m.def("reservedSpecificAdamEpsPrefix", &reservedSpecificAdamEpsPrefix);
  m.def("reservedDefaultStepPrefix", &reservedDefaultStepPrefix);
  m.def("reservedSpecificStepPrefix", &reservedSpecificStepPrefix);
  m.def("hostReduceGradCopyPrefix", &hostReduceGradCopyPrefix);
  m.def("hostReduceVarCopyPrefix", &hostReduceVarCopyPrefix);

  // These are helper methods to allow unit tests to start & stop
  // the debug info. They should move to poplar whenit has a python
  // interface.
  m.def(
      "initializePoplarDebugInfo",
      [&](std::string filename, std::string format) {
        poplar::DebugSerializationFormat sformat =
            poplar::DebugSerializationFormat::JSON;
        if (format == "json") {
          sformat = poplar::DebugSerializationFormat::JSON;
        } else if (format == "cbor") {
          sformat = poplar::DebugSerializationFormat::CBOR;
        }

        poplar::DebugInfo::initializeStreamer(filename, sformat);
      },
      py::arg("filename"),
      py::arg("format") = "json");

  m.def("closePoplarDebugInfo", [&]() { poplar::DebugInfo::closeStreamer(); });

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
