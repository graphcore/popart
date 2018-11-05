#include <willow/error.hpp>
#include <willow/filereader.hpp>
#include <willow/numerics.hpp>
#include <willow/onnxutil.hpp>

#include <cmath>

namespace willow {
namespace numerics {

template <typename T> class NumericsTracker {

private:
  // sums of squares of weight differences
  T ss_dA{0};
  T ss_dB{0};
  T ss_dAB{0};
  int64_t nSamples;

public:
  void insert(T v_A0, T v_A1, T v_B0, T v_B1) {
    ++nSamples;
    T dA = v_A1 - v_A0;
    T dB = v_B1 - v_B0;
    ss_dA += dA * dA;
    ss_dB += dB * dB;
    ss_dAB += (dA - dB) * (dA - dB);
  }

  std::string str() {
    T relerr = (ss_dAB) / (std::sqrt(ss_dA * ss_dB) + 1e-8f);
    std::stringstream ss;
    ss.precision(8);
    ss << "|dA - dB|^2 / (|dA||dB| + 1e-8)  = " << relerr;
    return ss.str();
  }
};

NumericsReport::NumericsReport(std::string A0, // A starts
                               std::string A1, // A ends
                               std::string B0, // B starts
                               std::string B1  // B ends
) {

  //   auto varTensors =
  //   willowNet.getIr()->tensors.getIds(TensorType::Variable);

  std::vector<std::string> fns{A0, A1, B0, B1};
  for (auto fn : fns) {
    io::confirmRegularFile(fn);
  }

  std::map<std::string, onnx::ModelProto> models;
  for (auto fn : fns) {
    models[fn] = io::getModel(fn);
  }

  const onnx::ModelProto &mA0 = models[A0];

  for (auto fn : fns) {
    if (models[fn].graph().initializer_size() !=
        mA0.graph().initializer_size()) {
      throw error("GraphProtos have different number of initializers");
    }
  }

  for (int wIndex = 0; wIndex < mA0.graph().initializer_size(); ++wIndex) {

    auto getTensor = [wIndex,
                      &models](std::string fn) -> const onnx::TensorProto & {
      return models[fn].graph().initializer(wIndex);
    };

    // confirm Tensor names are the same
    // across Models, at index wIndex.
    for (auto fn : fns) {
      if (getTensor(A0).name() != getTensor(fn).name()) {
        throw error("Tensor names do not correspond between TensorProtos");
      }
    }

    std::map<std::string, ConstVoidData> cv_datas;
    for (auto fn : fns) {
      cv_datas[fn] = onnxutil::getConstData(getTensor(fn));
    }

    // confirm the TensorInfos (shape and type) are the
    // same across Models, at index wIndex.
    for (auto fn : fns) {
      if (cv_datas[A0].info != cv_datas[fn].info) {
        throw error("TensorProto infos differ");
      }
    }

    if (cv_datas[A0].info.dataType() == TP::FLOAT) {
      NumericsTracker<float> tracker;
      for (unsigned i = 0; i < cv_datas[A0].info.nelms(); ++i) {
        tracker.insert(static_cast<const float *>(cv_datas[A0].data)[i],
                       static_cast<const float *>(cv_datas[A1].data)[i],
                       static_cast<const float *>(cv_datas[B0].data)[i],
                       static_cast<const float *>(cv_datas[B1].data)[i]);
      }

      reports[getTensor(A0).name()] = tracker.str();
    }

    else {
      throw error("Create report for type " + cv_datas[A0].info.data_type());
    }
  }
}

std::string NumericsReport::fullReport() const {
  std::stringstream ss;
  for (const auto &id_report : reports) {
    ss << '\n'
       << id_report.first << " : \n"
       << "     " << id_report.second;
  }
  return ss.str();
}

std::string NumericsReport::report(TensorId id) const {
  auto found = reports.find(id);
  if (found == reports.end()) {
    throw error("No report available for " + id);
  }
  return found->second;
}
} // namespace numerics
} // namespace willow
