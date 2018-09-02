#ifndef GUARD_NEURALNET_FILEREADER_HPP
#define GUARD_NEURALNET_FILEREADER_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings
#include <neuralnet/names.hpp>
#include <sstream>

namespace neuralnet {
namespace io {

// get the canonical directory name
// see boost.org/doc/libs/1_33_1/libs/filesystem/doc/path.htm#Canonical
std::string getCanonicalDirName(const std::string &dirName0);

// construct a full path name by joining dir with fn (much like python
// path.join)
std::string appendDirFn(const std::string &dir, const std::string &fn);

/// load a NodeProto from a file
/// see https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto
/// and C++ generated header onnx.pb.h for class def.
onnx::ModelProto getModel(std::string filename);

// getNode function from previous versions has been removed

// load TensorProto
onnx::TensorProto getTensor(std::string filename);

// fns are of the form somethingorother_dd.suffix
// for each fn in fns:
//     1) extract dd
//     2) load the tensor from file with name fn
//     3) name it as names[dd] (assuming dd is valid)
// these Tensors are returned in an OnnxTensors.
// This function is useful for the test suites provided by ONNX
OnnxTensors getAndMatchTensors(const std::vector<std::string> &fns,
                               const std::vector<std::string> &names);

// return all full path names of files which match toMatch in directory dir
std::vector<std::string> getMatchFns(std::string dir, std::string toMatch);

// load all tensors in directory dir with "input" in their name.
// The Graph g is needed, it us used to name the tensors correctly.
// it matches the input_dd to g.input(dd).
OnnxTensors getInputTensors(const onnx::GraphProto &g, std::string dir);

// load all tensors in directory dir with "output" in their name.
OnnxTensors getOutputTensors(const onnx::GraphProto &g, std::string dir);

// utility.
void confirmRegularFile(std::string filename);

// return all full path names for regular files in dir
std::vector<std::string> getFns(std::string dir);

// return the fullpath names of all
// subdirectories of directory dir
std::vector<std::string> getDirns(std::string dir);

} // namespace io
} // namespace neuralnet
#endif
