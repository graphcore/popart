// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_PBWRAP_GRAPH_HPP
#define GUARD_PBWRAP_GRAPH_HPP

#include <map>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// These classes pass calls to .size() and .at() to
// input_size() and input(int) and output_size() and output()
// so that standard containers can be used in Ir::connectInputs (as T)

class InputVecWrapper {
public:
  InputVecWrapper(const std::vector<TensorId> &inputs_);
  int input_size() const;
  const TensorId &input(int inIndex) const;

private:
  const std::vector<TensorId> &inputs;
};

class OutputVecWrapper {
public:
  OutputVecWrapper(const std::vector<TensorId> &outputs_);
  int output_size() const;
  const TensorId &output(int inIndex) const;

private:
  const std::vector<TensorId> &outputs;
};

class IOMapWrapper {
public:
  IOMapWrapper(const std::map<int, TensorId> &M_);

  const TensorId &idAt(int inIndex) const;
  int get_size() const;

private:
  const std::map<int, TensorId> &M;
  int max_index{0};
  std::string nullString{""};
};

class InputMapWrapper : public IOMapWrapper {
public:
  InputMapWrapper(const std::map<int, TensorId> &M_);
  const TensorId &input(int inIndex) const;
  int input_size() const;
};

class OutputMapWrapper : public IOMapWrapper {
public:
  OutputMapWrapper(const std::map<int, TensorId> &M_);
  const TensorId &output(int inIndex) const;
  int output_size() const;
};

} // namespace popart

#endif
