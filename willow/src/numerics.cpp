#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/numerics.hpp>
#include <poponnx/onnxutil.hpp>

#include <cmath>

namespace poponnx {
namespace numerics {

NumericsReport::NumericsReport(std::string AStarts, // A starts
                               std::string AEnds,   // A ends
                               std::string BStarts, // B starts
                               std::string BEnds    // B ends
) {

  std::vector<std::string> fns{AStarts, AEnds, BStarts, BEnds};
  for (auto fn : fns) {
    io::confirmRegularFile(fn);
  }

  std::map<std::string, onnx::ModelProto> models;
  for (auto fn : fns) {
    models[fn] = io::getModelFromFile(fn);
  }

  const onnx::ModelProto &mAStarts = models[AStarts];

  for (auto fn : fns) {
    if (models[fn].graph().initializer_size() !=
        mAStarts.graph().initializer_size()) {
      throw error("GraphProtos have different number of initializers");
    }
  }

  for (int wIndex = 0; wIndex < mAStarts.graph().initializer_size(); ++wIndex) {

    auto getTensor = [wIndex,
                      &models](std::string fn) -> const onnx::TensorProto & {
      return models[fn].graph().initializer(wIndex);
    };

    // confirm Tensor names are the same
    // across Models, at index wIndex.
    for (auto fn : fns) {
      if (getTensor(AStarts).name() != getTensor(fn).name()) {
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
      if (cv_datas[AStarts].info != cv_datas[fn].info) {
        throw error("TensorProto infos differ");
      }
    }

    if (cv_datas[AStarts].info.dataType() == TP::FLOAT) {
      NumericsTracker<float> tracker;
      for (unsigned i = 0; i < cv_datas[AStarts].info.nelms(); ++i) {
        tracker.insert(static_cast<const float *>(cv_datas[AStarts].data)[i],
                       static_cast<const float *>(cv_datas[AEnds].data)[i],
                       static_cast<const float *>(cv_datas[BStarts].data)[i],
                       static_cast<const float *>(cv_datas[BEnds].data)[i]);
      }
      relerrs[getTensor(AStarts).name()] = tracker.getRelativeError();
      reports[getTensor(AStarts).name()] = tracker.str();
    }

    else {
      throw error("Create report for type " +
                  cv_datas[AStarts].info.data_type());
    }
  }
}

std::string NumericsReport::fullReport() const {
  std::stringstream ss;
  for (const auto &id_report : reports) {
    ss << '\n' << id_report.first << " : \n" << id_report.second;
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

std::map<TensorId, float> NumericsReport::getRelativeErrors() {
  return relerrs;
}

} // namespace numerics
} // namespace poponnx
