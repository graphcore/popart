// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_DOMAINOPSET_HPP
#define GUARD_DOMAINOPSET_HPP

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

#endif