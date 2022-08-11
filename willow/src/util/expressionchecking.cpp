// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/util/expressionchecking.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "boost/utility/string_view.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"

namespace popart {
namespace internal {

FailedCheckThrower::FailedCheckThrower(const std::string prefix,
                                       const std::string suffix)
    : prefix_(prefix.c_str()), suffix_(suffix.c_str()) {}

FailedCheckThrower::FailedCheckThrower(const char *prefix, const char *suffix)
    : prefix_(prefix), suffix_(suffix) {}

FailedCheckThrower::~FailedCheckThrower() noexcept(false) {
  const auto &message = buildErrorMessage();
  throw popart::error(message);
}

std::string FailedCheckThrower::buildErrorMessage() const {
  const auto extra_message = extra_message_.str();
  std::vector<boost::string_view> message_components;

  if (!prefix_.empty()) {
    message_components.push_back(prefix_);
  }
  if (!extra_message.empty()) {
    message_components.push_back(extra_message);
  }
  if (!suffix_.empty()) {
    message_components.push_back(suffix_);
  }

  std::stringstream message;
  for (std::size_t i = 0; i < message_components.size(); i++) {
    message << message_components[i];
    if (i + 1 != message_components.size()) {
      message << " ";
    }
  }

  return message.str();
}

} // namespace internal
} // namespace popart
