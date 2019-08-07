#define BOOST_ERROR_CODE_HEADER_ONLY
// this ifdef suppresses the unused macro warning for
// BOOST_ERROR_CODE_HEADER_ONLY
#ifdef BOOST_ERROR_CODE_HEADER_ONLY
#endif

#include <boost/filesystem.hpp>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/names.hpp>

namespace popart {
namespace io {

// 2GB total_bytes_limit for reading protobuf coded input streams
// see google.protobuf.io.coded_stream#CodedInputStream
static const int protobufByteLimit = std::numeric_limits<int>::max();

std::string getCanonicalDirName(const std::string &dirName0) {
  namespace bf = boost::filesystem;
  if (!bf::is_directory(dirName0)) {
    throw popart::error("Directory does not exisit: {}", dirName0);
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

bool isRegularFile(const std::string &filename) {
  boost::system::error_code ec;
  auto isRegularFile = boost::filesystem::is_regular_file(filename, ec);
  // If the file system API reports an error then we assume that this is not a
  // regular file.
  // See
  // https://www.boost.org/doc/libs/1_45_0/libs/filesystem/v3/doc/reference.html#status
  return ec ? false : isRegularFile;
}

void confirmRegularFile(const std::string &filename) {
  if (!boost::filesystem::is_regular_file(filename)) {
    throw error("{} is not a regular file, cannot load", filename);
  }
}

OnnxTensors getInputTensors(const onnx::GraphProto &g, const std::string &dir) {
  auto fns = getMatchFns(dir, "input");
  std::vector<std::string> names;
  for (auto &x : g.input()) {
    names.push_back(x.name());
  }
  return getAndMatchTensors(fns, names);
}

OnnxTensors getOutputTensors(const onnx::GraphProto &g,
                             const std::string &dir) {
  auto fns = getMatchFns(dir, "output");
  std::vector<std::string> names;
  for (auto &x : g.output()) {
    names.push_back(x.name());
  }
  return getAndMatchTensors(fns, names);
}

template <typename T>
static bool getProtobufFromStream(std::istream &istream, T &proto) {
  google::protobuf::io::IstreamInputStream inputStream(&istream);
  google::protobuf::io::CodedInputStream codedInputStream(&inputStream);

  codedInputStream.SetTotalBytesLimit(protobufByteLimit, -1);
  return proto.ParseFromCodedStream(&codedInputStream);
}

static bool getModelFromStream(std::istream &istream,
                               onnx::ModelProto &modelProto) {
  return getProtobufFromStream(istream, modelProto);
}

static void logModelInfo(onnx::ModelProto &modelProto) {
  logging::info("Onnx Model Info ir_version:{}, producer:{}.{}, domain:\"{}\", "
                "model_version:{} num_opsets:{}",
                modelProto.ir_version(),
                modelProto.producer_name(),
                modelProto.producer_version(),
                modelProto.domain(),
                modelProto.model_version(),
                modelProto.opset_import_size());

  for (auto opset : modelProto.opset_import()) {
    logging::info("Onnx Model OpSet domain:\"{}\" version:{}",
                  opset.domain(),
                  opset.version());
  }

  if (modelProto.has_graph()) {
    logging::info(
        "Onnx Graph Info name:\"{}\" num_nodes:{} num_initializers:{} "
        "num_inputs:{} num_outputs:{} num_value_infos:{}",
        modelProto.graph().name(),
        modelProto.graph().node_size(),
        modelProto.graph().initializer_size(),
        modelProto.graph().input_size(),
        modelProto.graph().output_size(),
        modelProto.graph().value_info_size());
  }
}

onnx::ModelProto getModelFromFile(const std::string &filename) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  // As suggested at developers.google.com/protocol-buffers/docs/cpptutorial
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  confirmRegularFile(filename);
  std::fstream input(filename, std::ios::in | std::ios::binary);

  if (!input.is_open()) {
    std::stringstream ss;
    ss << "Failed to open file " << filename;
    throw error(ss.str());
  }

  onnx::ModelProto modelProto;

  if (!getModelFromStream(input, modelProto)) {
    std::stringstream ss;
    ss << "Failed to parse ModelProto from file " << filename;
    throw error(ss.str());
  }

  logModelInfo(modelProto);

  return modelProto;
}

onnx::ModelProto getModelFromString(const std::string &stringProto) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  // As suggested at developers.google.com/protocol-buffers/docs/cpptutorial
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::stringstream input(stringProto);

  onnx::ModelProto modelProto;

  if (!getModelFromStream(input, modelProto)) {
    throw error("Failed to parse ModelProto from string");
  }

  logModelInfo(modelProto);

  return modelProto;
}

void writeModel(const onnx::ModelProto &model, const std::string &filename) {

  std::ofstream ofs;
  ofs.open(filename, std::ofstream::out | std::ofstream::binary);
  if (!ofs.is_open()) {
    throw error("Failed to open file {}", filename);
  }

  // Standard Message Methods have this functionality for serializing
  // https://developers.google.com/protocol-buffers/docs/cpptutorial
  if (!model.SerializeToOstream(&ofs)) {
    throw error("Failed to serialize ModelProto to {}", filename);
  }
}

onnx::TensorProto getTensor(const std::string &filename) {

  confirmRegularFile(filename);
  std::fstream fs(filename, std::ios::in | std::ios::binary);

  if (!fs.is_open()) {
    std::stringstream ss;
    ss << "failed to open file " << filename;
    throw error(ss.str());
  }

  onnx::TensorProto tensor;
  if (!getProtobufFromStream(fs, tensor)) {
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
      throw error("number extracted from filename exceeds size of names. "
                  "number = {} and size of names = {}",
                  number,
                  names.size());
    }
    // At this point Tensor does not have a name (at least in the test suite).
    tensor.set_name(names[number]);
    auto tensorName = tensor.name();
    tensors.insert({tensorName, std::move(tensor)});
  }
  return tensors;
}

// return all names of full path names of files which match to_match
std::vector<std::string> getMatchFns(const std::string &dir,
                                     const std::string &to_match) {
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
std::vector<std::string> getInDir(const std::string &dir, T check) {
  // std::function<bool(const boost::filesystem::path &path)>
  std::vector<std::string> fns;
  namespace bf = boost::filesystem;
  bf::path p(dir);
  if (!is_directory(p)) {
    throw error("{} in not a directory, bailing from getInDir", p);
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

std::vector<std::string> getDirns(const std::string &dir) {
  auto is_dir = [](const boost::filesystem::path &path) {
    return boost::filesystem::is_directory(path);
  };
  return getInDir(dir, is_dir);
}
// return all full path names for regular files in dir
std::vector<std::string> getFns(const std::string &dir) {
  auto is_reg = [](const boost::filesystem::path &path) {
    return boost::filesystem::is_regular_file(path);
  };
  return getInDir(dir, is_reg);
}
} // namespace io
} // namespace popart
