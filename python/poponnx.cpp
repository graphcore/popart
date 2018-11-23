#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <poponnx/builder.hpp>
#include <poponnx/device.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/error.hpp>
#include <poponnx/numerics.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/loss.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace py = pybind11;
using namespace willow;

std::map<std::string, DataType> initNpTypeMap() {
  std::map<std::string, DataType> M;
  // see tensorinfo.hpp for the complete list of
  // DataTypes (defined originally in ONNX)
  M["float16"] = TP::FLOAT16;
  M["float32"] = TP::FLOAT;
  M["int32"]   = TP::INT32;
  M["int64"]   = TP::INT64;
  M["bool"]    = TP::BOOL;
  return M;
}

DataType getDataTypeFromNpType(std::string npType) {
  const static std::map<std::string, DataType> M = initNpTypeMap();
  auto found                                     = M.find(npType);
  if (found == M.end()) {
    throw error("No numpy type " + npType + " registered in map to DataType");
  }
  return found->second;
}

TensorInfo getTensorInfo(py::array npArr) {
  auto dtype      = npArr.dtype();
  auto typeString = py::str(dtype);
  auto tRank      = npArr.ndim();
  std::vector<int64_t> shape;
  for (int i = 0; i < tRank; ++i) {
    shape.push_back(npArr.shape(i));
  }
  return TensorInfo(getDataTypeFromNpType(typeString), shape);
}

// The follow code attempts to convert the python dictionary
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
class PyStepIO : public StepIO {
public:
  PyStepIO(std::map<TensorId, py::array> inputs_,
           std::map<TensorId, py::array> outputs_)
      : inputs(inputs_), outputs(outputs_) {}

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, py::array> &M,
        std::string mapName) const {
    auto found = M.find(id);
    if (found == M.end()) {
      throw error("No tensor " + id + " provided in PyStepIO's " + mapName);
    }
    py::array npArr = found->second;
    T stepData;
    stepData.data = npArr.request().ptr;
    stepData.info = getTensorInfo(npArr);
    return stepData;
  }

  ConstVoidData in(TensorId id) const final {
    return get<ConstVoidData>(id, inputs, "inputs");
  }

  MutableVoidData out(TensorId id) const final {
    return get<MutableVoidData>(id, outputs, "outputs");
  }

private:
  std::map<TensorId, py::array> inputs;
  std::map<TensorId, py::array> outputs;
};

PYBIND11_MODULE(poponnx_core, m) {
  m.doc() = "binding for C++ poponnx library";

  m.def("getTensorInfo", &getTensorInfo);

  m.def("getSupportedOperations",
        &getSupportedOperations,
        py::arg("includeInternal"));

  py::class_<StepIO> stepio(m, "StepIO");

  py::enum_<AnchorReturnType>(m, "AnchorReturnType")
      .value("FINAL", AnchorReturnType::FINAL)
      .value("SUM", AnchorReturnType::SUM)
      .value("ALL", AnchorReturnType::ALL);

  py::class_<PyStepIO>(m, "PyStepIO", stepio)
      .def(py::init<std::map<TensorId, py::array>,
                    std::map<TensorId, py::array>>(),
           py::arg("inputs"),
           py::arg("outputs"));

  py::class_<DataFlow>(m, "DataFlow")
      .def(
          py::init<int, int, const std::vector<TensorId> &, AnchorReturnType>(),
          py::arg("batchesPerStep"),
          py::arg("batchSize"),
          py::arg("anchorTensors"),
          py::arg("anchorReturnType"))
      .def("nAnchors", &DataFlow::nAnchors)
      .def("batchSize", &DataFlow::batchSize)
      .def("batchesPerStep", &DataFlow::batchesPerStep)
      .def("anchors", &DataFlow::anchors, pybind11::return_value_policy::copy)
      .def("art", &DataFlow::art);

  py::class_<TensorInfo>(m, "TensorInfo")
      .def(py::init<std::string, const std::vector<int64_t> &>(),
           py::arg("dataType"),
           py::arg("shape"))
      .def("data_type_lcase", &TensorInfo::data_type_lcase)
      .def("shape", &TensorInfo::shape);

  py::class_<numerics::NumericsReport>(m, "NumericsReport")
      .def(py::init<std::string, std::string, std::string, std::string>(),
           py::arg("A0"),
           py::arg("A1"),
           py::arg("B0"),
           py::arg("B1"))
      .def("report", &numerics::NumericsReport::report)
      .def("fullReport", &numerics::NumericsReport::fullReport)
      .def("getRelativeErrors", &numerics::NumericsReport::getRelativeErrors);

  py::class_<EarlyInfo>(m, "EarlyInfo")
      .def(py::init<>())
      .def("add", &EarlyInfo::add)
      .def("get", &EarlyInfo::get)
      .def("has", &EarlyInfo::has);

  py::class_<Loss> loss(m, "Loss");
  loss.def("input", &Loss::input);
  loss.def("output", &Loss::output);

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId>(),
           py::arg("probabilities"),
           py::arg("labels"),
           py::arg("output"))
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId);

  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float>(),
           py::arg("input"),
           py::arg("output"),
           py::arg("lambda"))
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda);

  py::class_<Optimizer> optimizer(m, "Optimizer");

  py::class_<BaseSGD> basesgd(m, "BaseSGD", optimizer);
  basesgd.def("learnRate", &BaseSGD::learnRate);

  py::class_<SGD>(m, "SGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"));

  py::class_<ConstSGD>(m, "ConstSGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"));

  py::class_<SessionOptions>(m, "SessionOptionsCore")
      .def(py::init<>())
      .def_readwrite("exportDot", &SessionOptions::exportDot)
      .def_readwrite("engineOptions", &SessionOptions::engineOptions)
      .def_readwrite("convolutionOptions", &SessionOptions::convolutionOptions)
      .def_readwrite("reportOptions", &SessionOptions::reportOptions)
      .def_readwrite("logging", &SessionOptions::loggingOptions);

  py::class_<Session>(m, "SessionCore")
      .def(py::init(&Session::createFromOnnxModel),
           py::arg("model"),
           py::arg("earlyInfo").none(),
           py::arg("dataFlow").none(),
           py::arg("losses"),
           py::arg("optimizer").none(),
           py::arg("cTens"),
           py::arg("logdir"),
           py::arg("userOptions"),
           py::arg("patternNames"))
      .def("updateOptimizer", &Session::updateOptimizer)
      .def("setDevice", &Session::setDevice)
      .def("prepareDevice", &Session::prepareDevice)
      .def("weightsFromHost", &Session::weightsFromHost)
      .def("optimizerFromHost", &Session::optimizerFromHost)
      .def("train", &Session::train)
      .def("evaluate", &Session::evaluate)
      .def("infer", &Session::infer)
      .def("modelToHost", &Session::modelToHost)
      .def("getInfo", &Session::getInfo)
      .def("getSummaryReport", &Session::getSummaryReport)
      .def("getGraphReport", &Session::getGraphReport)
      .def("getExecutionReport", &Session::getExecutionReport)
      .def("resetHostWeights", &Session::resetHostWeights);

  py::class_<Builder>(m, "BuilderCore")
      .def(py::init<>())
      .def("addInputTensor", &Builder::addInputTensor, py::arg("tensorInfo"))
      .def("addInitializedInputTensor",
           [](Builder &builder, py::array array) {
             ConstVoidData initData;
             initData.data = array.request().ptr;
             initData.info = getTensorInfo(array);
             builder.addInitializedInputTensor(initData);
           },
           py::arg("initVal"))
      .def("addOutputTensor", &Builder::addOutputTensor, py::arg("outputName"))
      .def("abs", &Builder::abs, py::arg("args"))
      .def("acos", &Builder::acos, py::arg("args"))
      .def("acosh", &Builder::acosh, py::arg("args"))
      .def("add", &Builder::add, py::arg("args"))
      .def("logical_and", &Builder::logical_and, py::arg("args"))
      .def("asin", &Builder::asin, py::arg("args"))
      .def("asinh", &Builder::asinh, py::arg("args"))
      .def("atan", &Builder::atan, py::arg("args"))
      .def("atanh", &Builder::atanh, py::arg("args"))
      .def("ceil", &Builder::ceil, py::arg("args"))
      .def("cos", &Builder::cos, py::arg("args"))
      .def("cosh", &Builder::cosh, py::arg("args"))
      .def("div", &Builder::div, py::arg("args"))
      .def("elu", &Builder::elu, py::arg("args"))
      .def("equal", &Builder::equal, py::arg("args"))
      .def("exp", &Builder::exp, py::arg("args"))
      .def("floor", &Builder::floor, py::arg("args"))
      .def("greater", &Builder::greater, py::arg("args"))
      .def("identity", &Builder::identity, py::arg("args"))
      .def("less", &Builder::less, py::arg("args"))
      .def("log", &Builder::log, py::arg("args"))
      .def("max", &Builder::max, py::arg("args"))
      .def("mean", &Builder::mean, py::arg("args"))
      .def("min", &Builder::min, py::arg("args"))
      .def("mul", &Builder::mul, py::arg("args"))
      .def("neg", &Builder::neg, py::arg("args"))
      .def("logical_not", &Builder::logical_not, py::arg("args"))
      .def("logical_or", &Builder::logical_or, py::arg("args"))
      .def("pow", &Builder::pow, py::arg("args"))
      .def("reciprocal", &Builder::reciprocal, py::arg("args"))
      .def("relu", &Builder::relu, py::arg("args"))
      .def("sigmoid", &Builder::sigmoid, py::arg("args"))
      .def("sin", &Builder::sin, py::arg("args"))
      .def("sinh", &Builder::sinh, py::arg("args"))
      .def("softsign", &Builder::softsign, py::arg("args"))
      .def("sqrt", &Builder::sqrt, py::arg("args"))
      .def("sub", &Builder::sub, py::arg("args"))
      .def("sum", &Builder::sum, py::arg("args"))
      .def("tan", &Builder::tan, py::arg("args"))
      .def("tanh", &Builder::tanh, py::arg("args"))
      .def("logical_xor", &Builder::logical_xor, py::arg("args"))
      .def("convolution",
           &Builder::convolution,
           py::arg("args"),
           py::arg("strides"),
           py::arg("padding"),
           py::arg("dilation"),
           py::arg("groups"))
      .def("averagepool",
           &Builder::averagepool,
           py::arg("args"),
           py::arg("kernel_shape"),
           py::arg("strides"),
           py::arg("padding"))
      .def("maxpool",
           &Builder::maxpool,
           py::arg("args"),
           py::arg("kernel_shape"),
           py::arg("strides"),
           py::arg("padding"))
      .def("gemm",
           &Builder::gemm,
           py::arg("args"),
           py::arg("alpha"),
           py::arg("beta"),
           py::arg("transA"),
           py::arg("transB"))
      .def("matmul", &Builder::matmul, py::arg("args"))
      .def("getModelProto",
           [](const Builder &builder) {
             return py::bytes(builder.getModelProto());
           })
      .def("getInputTensorIds", &Builder::getInputTensorIds)
      .def("getOutputTensorIds", &Builder::getOutputTensorIds)
      .def("getTensorShape", &Builder::getTensorShape, py::arg("id"));

  // PyBinding to a singlton
  py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>>(
      m, "DeviceManager")
      .def(py::init([]() {
        return std::unique_ptr<DeviceManager, py::nodelete>(
            &DeviceManager::getDeviceManager());
      }))
      .def("acquireAvaliableDevice",
           static_cast<std::unique_ptr<DeviceInfo> (DeviceManager::*)()>(
               &DeviceManager::acquireAvaliableDevice))
      .def(
          "acquireAvaliableDevice",
          static_cast<std::unique_ptr<DeviceInfo> (DeviceManager::*)(int, int)>(
              &DeviceManager::acquireAvaliableDevice),
          py::arg("numIpus"),
          py::arg("tilesPerIpu"))
      .def(
          "acquireDeviceById", &DeviceManager::acquireDeviceById, py::arg("id"))
      .def("createCpuDevice", &DeviceManager::createCpuDevice)
      .def("createIpuModelDevice",
           [](DeviceManager &dm, py::dict e) {
             std::map<std::string, std::string> options = getDictionary(e);
             return dm.createIpuModelDevice(options);
           })
      .def("createSimDevice",
           [](DeviceManager &dm, py::dict e) {
             std::map<std::string, std::string> options = getDictionary(e);
             return dm.createSimDevice(options);
           })
      .def("enumerateDevices", &DeviceManager::enumerateDevices);

  // Do enum need to be uppercase?
  py::enum_<DeviceType>(m, "DeviceType")
      .value("IpuModel", DeviceType::IpuModel)
      .value("Cpu", DeviceType::Cpu)
      .value("Ipu", DeviceType::Ipu)
      .value("Sim", DeviceType::Sim);

  py::class_<DeviceInfo>(m, "DeviceInfo")
      .def("attach", &DeviceInfo::attach)
      .def("detach", &DeviceInfo::detach)
      .def_property_readonly("type", &DeviceInfo::getType)
      .def_property_readonly("version", &DeviceInfo::getVersion)
      .def_property_readonly("id", &DeviceInfo::getId)
      .def_property_readonly("numIpus", &DeviceInfo::getNumIpus)
      .def_property_readonly("tilesPerIpu", &DeviceInfo::getTilesPerIpu)
      .def_property_readonly("numWorkerContexts",
                             &DeviceInfo::getNumWorkerContexts)
      .def("__repr__", [](const DeviceInfo &di) {
        std::stringstream ss;
        ss << di;
        return ss.str();
      });

  py::register_exception<willow::error>(m, "poponnx_exception");
  py::register_exception<poplar::poplar_error>(m, "poplar_exception");
  py::register_exception<poputil::poplibs_error>(m, "poplibs_exception");
}
