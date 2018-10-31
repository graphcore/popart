#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <willow/error.hpp>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/nll.hpp>
#include <willow/optimizer.hpp>
#include <willow/stepio.hpp>
#include <willow/willownet.hpp>

// note to developers: be very careful
// about exposing functions which return pointers
// or references, python ref counter might behave
// unexpectedly. More info:
// https://pybind11.readthedocs.io/en/stable/advanced/functions.html

namespace py = pybind11;
using namespace willow;

std::map<std::string, DataType> initNpTypeMap() {
  std::map<std::string, DataType> M;
  // see tensorinfo.hpp for the complete list of
  // DataTypes (defined originally in ONNX)
  M["float16"] = TP::FLOAT16;
  M["float32"] = TP::FLOAT;
  M["float64"] = TP::DOUBLE;
  M["int16"]   = TP::INT16;
  M["int32"]   = TP::INT32;
  M["int64"]   = TP::INT64;
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
  auto dtype = npArr.dtype();
  // This seems to be the correct way to get
  // the string format from a dtype, I kind of
  // just stumbled upon it
  auto typeString = py::str(dtype);
  auto tRank      = npArr.ndim();
  std::vector<int64_t> shape;
  for (int i = 0; i < tRank; ++i) {
    shape.push_back(npArr.shape(i));
  }
  return TensorInfo(getDataTypeFromNpType(typeString), shape);
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

  virtual StepInData in(TensorId id) const override final {
    return get<StepInData>(id, inputs, "inputs");
  }

  virtual StepOutData out(TensorId id) const override final {
    return get<StepOutData>(id, outputs, "outputs");
  }

private:
  std::map<TensorId, py::array> inputs;
  std::map<TensorId, py::array> outputs;
};

PYBIND11_MODULE(pywillow, m) {
  m.doc() = "binding for C++ willow library";

  m.def("getTensorInfo", &getTensorInfo);

  py::class_<StepIO> stepio(m, "StepIO");

  py::enum_<AnchorReturnType>(m, "AnchorReturnType")
      .value("FINAL", AnchorReturnType::FINAL)
      .value("SUM", AnchorReturnType::SUM)
      .value("ALL", AnchorReturnType::ALL);


  py::class_<PyStepIO>(m, "PyStepIO", stepio)
      .def(py::init<std::map<TensorId, py::array>,
                    std::map<TensorId, py::array>>());

  py::class_<DataFlow>(m, "DataFlow")
      .def(
          py::init<int, int, const std::vector<TensorId> &, AnchorReturnType>(),
          py::arg("Batches processed between returning anchors"),
          py::arg("Batch size"),
          py::arg("Anchor tensors (tensors to return)"),
          py::arg("Anchor return type"))
      .def("nAnchors", &DataFlow::nAnchors);

  py::class_<TensorInfo>(m, "TensorInfo")
      .def(py::init<std::string, const std::vector<int64_t> &>());


  py::class_<EarlyInfo>(m, "EarlyInfo")
      .def(py::init<>())
      .def("add", &EarlyInfo::add)
      .def("get", &EarlyInfo::get)
      .def("has", &EarlyInfo::has);

  py::class_<Loss> loss(m, "Loss");
  loss.def("input", &Loss::input);

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId>())
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId);

  // TODO : document all the functions like this one
  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float>(),
           py::arg("The ID of the input tensor"),
           py::arg("The ID of the output tensor"),
           py::arg("lambda"))
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda);

  py::class_<Optimizer> optimizer(m, "Optimizer");

  py::class_<BaseSGD> basesgd(m, "BaseSGD", optimizer);
  // Note that we do not define a constructor, as it is a virtual class
  basesgd.def("learnRate", &BaseSGD::learnRate);

  // The Optimizer classes which are non-virtual:
  py::class_<SGD>(m, "SGD", basesgd).def(py::init<float>());
  py::class_<ConstSGD>(m, "ConstSGD", basesgd).def(py::init<float>());

  py::class_<WillowNet>(m, "WillowNet")
      .def(py::init<std::string,
                    const EarlyInfo &,
                    const DataFlow &,
                    const std::vector<Loss *> &,
                    const Optimizer *,
                    const std::vector<TensorId> &,
                    std::string,
                    const std::vector<std::string> &>())
      .def("updateOptimizer", &WillowNet::updateOptimizer)
      .def("setDevice", &WillowNet::setDevice)
      .def("prepareDevice", &WillowNet::prepareDevice)
      .def("weightsFromHost", &WillowNet::weightsFromHost)
      .def("optimizerFromHost", &WillowNet::optimizerFromHost)
      .def("step", &WillowNet::step)
      .def("modelToHost", &WillowNet::modelToHost)
      .def("getInfo", &WillowNet::getInfo);


  // This does not seem to work :/
  // Thoroughly read
  // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
  // and looked through pybind11 code, still can't get it to bight
  auto ex5 = py::register_exception<willow::error>(m, "WillowException");
}
