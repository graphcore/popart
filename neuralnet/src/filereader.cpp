#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <boost/filesystem.hpp>
#pragma clang diagnostic pop // stop ignoring warnings
#include <fstream>
#include <neuralnet/error.hpp>
#include <neuralnet/filereader.hpp>
#include <neuralnet/names.hpp>
#include <sstream>
#include <vector>

namespace neuralnet {
namespace io {

std::string getCanonicalDirName(const std::string &dirName0) {
  namespace bf = boost::filesystem;
  if (!bf::is_directory(dirName0)) {
    std::stringstream ss;
    ss << "Directory does not exist: " << dirName0;
    throw neuralnet::error(ss.str());
  }
  bf::path p(dirName0);
  return bf::canonical(dirName0).string();
}

std::string getCanonicalFilename(const std::string &fn) {
  namespace bf = boost::filesystem;
  bf::path p(fn);
  return bf::canonical(fn).string();
}

std::string appendDirFn(const std::string &dir, const std::string &fn) {
  boost::filesystem::path p(dir);
  auto fullPath = p / fn;
  return fullPath.string();
}

void confirmRegularFile(std::string filename) {
  if (!boost::filesystem::is_regular_file(filename)) {
    std::stringstream ss;
    ss << filename << " is not a regular file, cannot load";
    throw error(ss.str());
  }
}

OnnxTensors getInputTensors(const onnx::GraphProto &g, std::string dir) {
  auto fns = getMatchFns(dir, "input");
  std::vector<std::string> names;
  for (auto &x : g.input()) {
    names.push_back(x.name());
  }
  return getAndMatchTensors(fns, names);
}

OnnxTensors getOutputTensors(const onnx::GraphProto &g, std::string dir) {
  auto fns = getMatchFns(dir, "output");
  std::vector<std::string> names;
  for (auto &x : g.output()) {
    names.push_back(x.name());
  }
  return getAndMatchTensors(fns, names);
}

onnx::ModelProto getModel(std::string filename) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  // As suggested at developers.google.com/protocol-buffers/docs/cpptutorial
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  confirmRegularFile(filename);
  std::fstream input(filename, std::ios::in | std::ios::binary);

  if (!input.is_open()) {
    std::stringstream ss;
    ss << "failed to open file " << filename;
    throw error(ss.str());
  }

  onnx::ModelProto modelProto;

  if (!modelProto.ParseFromIstream(&input)) {
    std::stringstream ss;
    ss << "Failed to parse ModelProto from \n" << filename;
    throw error(ss.str());
  }

  if (modelProto.graph().node_size() == 0) {
    std::stringstream ss;
    ss << "In loading ModelProto from " << filename << '\n';
    ss << "model with zero nodes (=weird). Pedantic throw";
    throw error(ss.str());
  }
  return modelProto;
}

onnx::TensorProto getTensor(std::string filename) {

  confirmRegularFile(filename);
  std::fstream fs(filename, std::ios::in | std::ios::binary);

  if (!fs.is_open()) {
    std::stringstream ss;
    ss << "failed to open file " << filename;
    throw error(ss.str());
  }

  onnx::TensorProto tensor;
  if (!tensor.ParseFromIstream(&fs)) {
    std::stringstream ss;
    ss << "Failed to parse TensorProto from " << filename;
    throw error(ss.str());
  }

  return tensor;
}

OnnxTensors getAndMatchTensors(const std::vector<std::string> &fns,
                               const std::vector<std::string> &names) {
  namespace bf = boost::filesystem;

  OnnxTensors tensors;
  for (const auto &fn : fns) {
    auto tensor = getTensor(fn);
    // Using the specific naming convention in onnx examples repo
    bf::path p(fn);
    auto name   = p.filename().string();
    auto dStart = name.find('_');
    auto dEnd   = name.find('.');
    auto numStr = name.substr(dStart + 1, dEnd - dStart - 1);
    auto number = std::stoul(numStr);
    if (number >= names.size()) {
      std::stringstream errmss;
      errmss << "number extracted from filename exceeds size of names. "
             << "number = " << number
             << " and size of names = " << names.size();
      throw error(errmss.str());
    }
    // At this point Tensor does not have a name (at least in the test suite).
    tensor.set_name(names[number]);
    auto tensorName = tensor.name();
    tensors.insert({tensorName, std::move(tensor)});
  }
  return tensors;
}

// return all names of full path names of files which match to_match
std::vector<std::string> getMatchFns(std::string dir, std::string to_match) {
  namespace bf = boost::filesystem;
  std::vector<std::string> matches;
  auto fns = getFns(dir);
  for (const auto &fn : fns) {
    bf::path p(fn);
    std::string filename = p.filename().string();
    if (filename.find(to_match) != std::string::npos) {
      matches.push_back(fn);
    }
  }
  return matches;
}

template <typename T>
std::vector<std::string> getInDir(std::string dir, T check) {
  // std::function<bool(const boost::filesystem::path &path)>
  std::vector<std::string> fns;
  namespace bf = boost::filesystem;
  bf::path p(dir);
  if (!is_directory(p)) {
    std::stringstream ss;
    ss << p << " in not a directory, bailing from getInDir";
    throw error(ss.str());
  } else {
    bf::directory_iterator eod;
    for (bf::directory_iterator dir_itr(p); dir_itr != eod; ++dir_itr) {
      auto bf_path = dir_itr->path();
      if (check(bf_path)) {
        auto fn = bf_path.string();
        fns.push_back(fn);
      }
    }
  }
  return fns;
}

std::vector<std::string> getDirns(std::string dir) {
  auto is_dir = [](const boost::filesystem::path &path) {
    return boost::filesystem::is_directory(path);
  };
  return getInDir(dir, is_dir);
}
// return all full path names for regular files in dir
std::vector<std::string> getFns(std::string dir) {
  auto is_reg = [](const boost::filesystem::path &path) {
    return boost::filesystem::is_regular_file(path);
  };
  return getInDir(dir, is_reg);
}
} // namespace io
} // namespace neuralnet
