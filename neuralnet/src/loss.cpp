#include <neuralnet/error.hpp>
#include <neuralnet/loss.hpp>
#include <sstream>

namespace neuralnet {

std::map<std::string, eLoss> initLossMap() {
  return {{"NLL", eLoss::NLL}, {"L1", eLoss::L1}};
}

const std::map<std::string, eLoss> &lossMap() {
  static std::map<std::string, eLoss> m = initLossMap();
  return m;
}

int Loss::input_size() const { return static_cast<int>(input_.size()); }
const TensorId &Loss::input(int i) const { return input_.at(i); }
int Loss::output_size() const { return 1; }
const TensorId &Loss::output(int i) const {
  if (i != 0) {
    throw error("only 1 loss output");
  }
  return output_;
}

Loss::Loss(const std::string &argstring) {
  // where argstring is inputs : output : other

  auto p0 = argstring.find(':');
  if (p0 == std::string::npos) {
    throw error("invalid loss string in Loss constructor");
  }
  auto p1 = argstring.find(':', p0 + 1);
  if (p1 == std::string::npos) {
    throw error("invalid loss string in Loss constructor");
  }
  std::string s_in   = argstring.substr(0, p0);
  std::string s_out  = argstring.substr(p0 + 1, p1 - p0 - 1);
  std::string s_args = argstring.substr(p1 + 1);

  std::string newArg;

  std::istringstream iss(s_in);
  while (iss >> newArg) {
    input_.push_back(newArg);
  }

  int outCount = 0;
  std::istringstream oss(s_out);
  while (oss >> output_) {
    ++outCount;
  }
  if (outCount != 1) {
    throw error("expected 1 output");
  }

  std::istringstream ass(s_args);
  while (ass >> newArg) {
    args_.push_back(newArg);
    std::cout << "args : " << newArg << std::endl;
  }

  std::cout << "----------------" << std::endl;
  for (auto &x : input_) {
    std::cout << '`' << x << '\'' << std::endl;
  }
  std::cout << "----------------" << std::endl;

  std::cout << '`' << output_ << '\'' << std::endl;

  std::cout << "----------------" << std::endl;

  for (auto &x : args_) {
    std::cout << '`' << x << '\'' << std::endl;
  }

  std::cout << "----------------" << std::endl;
}

const std::vector<std::string> &Loss::args() const { return args_; }

void Loss::confirmSizes(int nIn, int nArgs) const {
  if (input_size() != nIn) {
    throw error("Loss expected " + std::to_string(input_size()) + " inputs");
  }

  if (args().size() != nArgs) {
    throw error("Loss expected " + std::to_string(args().size()) + " args");
  }
}

} // namespace neuralnet
