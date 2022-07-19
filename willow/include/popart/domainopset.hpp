// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_DOMAINOPSET_HPP_
#define POPART_WILLOW_INCLUDE_POPART_DOMAINOPSET_HPP_

#include <memory>

namespace popart {

class BuilderImpl;

class DomainOpSet {

protected:
  std::unique_ptr<BuilderImpl> &impl;

  virtual int getOpsetVersion() const = 0;

public:
  DomainOpSet(std::unique_ptr<BuilderImpl> &impl_) : impl(impl_) {}
  DomainOpSet(const DomainOpSet &other) = default;
  virtual ~DomainOpSet()                = default;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_DOMAINOPSET_HPP_
