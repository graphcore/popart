// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef GUARD_STREAM_MODE_CPP
#define GUARD_STREAM_MODE_CPP

#include <ostream>
#include <popart/replicatedstreammode.hpp>

std::ostream &operator<<(std::ostream &os, const ReplicatedStreamMode &tt) {
  switch (tt) {
  case ReplicatedStreamMode::Broadcast:
    os << "Broadcast";
    break;
  case ReplicatedStreamMode::Replicate:
    os << "Replicate";
    break;
  default:
    os << "Undefined";
    break;
  }

  return os;
}

#endif
