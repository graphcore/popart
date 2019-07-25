#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/util.hpp>

namespace popart {

char *getenv(std::string env_var) {
  auto result = std::getenv(fmt::format("POPART_{}", env_var).c_str());
  if (result) {
    return result;
  } else {
    result = std::getenv(fmt::format("POPONNX_{}", env_var).c_str());
    if (result != nullptr) {
      logging::warn("You are using a deprecated environment variable '{}', "
                    "please change it to '{}'",
                    fmt::format("POPONNX_{}", env_var),
                    fmt::format("POPART_{}", env_var));
    }
    return result;
  }
}

std::ostream &operator<<(std::ostream &ss, const std::vector<std::size_t> &v) {
  appendSequence(ss, v);
  return ss;
}
} // namespace popart
