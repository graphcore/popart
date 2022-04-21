// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FILEREADER_HPP
#define GUARD_NEURALNET_FILEREADER_HPP

#include <onnx/onnx_pb.h>
#include <sstream>
#include <popart/names.hpp>

namespace popart {
namespace io {

// Raise an exception if the directory does not exist
void assertDirectoryExists(const std::string &path);

// Raise an exception if the directory is not writable
void assertDirectoryWritable(const std::string &path);

// get the canonical directory name
// see boost.org/doc/libs/1_33_1/libs/filesystem/doc/path.htm#Canonical
std::string getCanonicalDirName(const std::string &dirName0);

std::string getCanonicalFilename(const std::string &dirName0);

// construct a full path name by joining dir with fn (much like python
// path.join)
std::string appendDirFn(const std::string &dir, const std::string &fn);

/// load a ModelProto from a file
ONNX_NAMESPACE::ModelProto getModelFromFile(const std::string &filename);

/// load a ModelProto from a string
ONNX_NAMESPACE::ModelProto getModelFromString(const std::string &modelProto);

// serialize a ModelProto to a binary protobuf file
void writeModel(const ONNX_NAMESPACE::ModelProto &model,
                const std::string &filename);

// getNode function from previous versions has been removed

// load TensorProto
ONNX_NAMESPACE::TensorProto getTensor(const std::string &filename);

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
std::vector<std::string> getMatchFns(const std::string &dir,
                                     const std::string &toMatch);

// load all tensors in directory dir with "input" in their name.
// The Graph g is needed, it us used to name the tensors correctly.
// it matches the input_dd to g.input(dd).
OnnxTensors getInputTensors(const ONNX_NAMESPACE::GraphProto &g,
                            const std::string &dir);

// load all tensors in directory dir with "output" in their name.
OnnxTensors getOutputTensors(const ONNX_NAMESPACE::GraphProto &g,
                             const std::string &dir);

// Check if a filename is a regular file
bool isRegularFile(const std::string &filename);

// utility.
void confirmRegularFile(const std::string &filename);

// return all full path names for regular files in dir
std::vector<std::string> getFns(const std::string &dir);

// return the fullpath names of all
// subdirectories of directory dir
std::vector<std::string> getDirns(const std::string &dir);

} // namespace io
} // namespace popart
#endif
