// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/debugcontext.hpp>
#include <popart/popx/debugcontextx.hpp>
//#include <popart/names.hpp>

namespace popart {
struct DebugContextImpl;

struct ProfileValueImpl {
  ProfileValueImpl(popart::ProfileValue::String init)
      : pv((poplar::ProfileValue::String)init) {}
  ProfileValueImpl(popart::ProfileValue::Vector init) : pv() {
    poplar::ProfileValue::Vector vect;
    for (auto i : init) {
      vect.push_back(i.impl->pv);
    }
    pv = vect;
  }
  ProfileValueImpl(popart::ProfileValue::Map init) : pv() {
    poplar::ProfileValue::Map map;
    for (auto i : init) {
      map.insert({i.first, i.second.impl->pv});
    }
    pv = map;
  }
  ProfileValueImpl(popart::ProfileValue::Number init)
      : pv((poplar::ProfileValue::Number)init) {}
  ProfileValueImpl(popart::ProfileValue::Boolean init)
      : pv((poplar::ProfileValue::Boolean)init) {}

  poplar::ProfileValue pv;
};

// Need to subclass to make setValue public
class _DebugInfo : public poplar::DebugInfo {
public:
  _DebugInfo(poplar::DebugContext &dc, const std::string &layer)
      : poplar::DebugInfo(dc, layer) {}

  using poplar::DebugInfo::setValue;
};

struct DebugInfoImpl {
  DebugInfoImpl(DebugContextImpl &impl, const std::string &layer);
  DebugId getId() const { return di.getId(); }
  std::string getPathName() const { return di.getPathName(); }
  bool setValue(std::string name, ProfileValueImpl &impl) {
    return di.setValue(name, impl.pv);
  }
  _DebugInfo di;
};

struct DebugNameAndIdImpl {
  DebugNameAndIdImpl(const std::string &name,
                     DebugId id,
                     const std::string &parentPath)
      : dnai(name, id, parentPath) {}

  DebugNameAndIdImpl(DebugId id) : dnai(id) {}

  DebugNameAndIdImpl(const char *name) : dnai(name) {}

  DebugNameAndIdImpl(DebugInfoImpl &impl, const std::string &name)
      : dnai(impl.di, name) {}

  DebugNameAndIdImpl(DebugNameAndIdImpl &impl, const std::string &name)
      : dnai(impl.dnai, name) {}

  poplar::DebugNameAndId dnai;
};

struct DebugContextImpl {
  DebugContextImpl(const std::string &name, SourceLocation &_loc)
      : loc(std::make_shared<SourceLocation>(_loc)),
        dc(name,
           poplar::SourceLocation(loc->getFunctionName().c_str(),
                                  loc->getFileName().c_str(),
                                  loc->getLineNumber())) {}

  DebugContextImpl(DebugInfoImpl &impl,
                   const std::string &name,
                   SourceLocation &_loc)
      : loc(std::make_shared<SourceLocation>(_loc)),
        dc(impl.di,
           name,
           poplar::SourceLocation(loc->getFunctionName().c_str(),
                                  loc->getFileName().c_str(),
                                  loc->getLineNumber())) {}

  DebugContextImpl(DebugNameAndIdImpl &impl,
                   const std::string &name,
                   SourceLocation &_loc)
      : loc(std::make_shared<SourceLocation>(_loc)),
        dc(impl.dnai,
           name,
           poplar::SourceLocation(loc->getFunctionName().c_str(),
                                  loc->getFileName().c_str(),
                                  loc->getLineNumber())) {}

  std::string getPathName() { return dc.getPathName(); }

  // Need to be a shared ptr as the char* is passed to the poplar
  // DebugContext and we need to make sure it lives long enough to be
  // written out when when the DebugContext is copied.
  std::shared_ptr<SourceLocation> loc;
  poplar::DebugContext dc;
};

DebugInfoImpl::DebugInfoImpl(DebugContextImpl &impl, const std::string &layer)
    : di(impl.dc, layer) {}
} // namespace popart

using namespace popart;

//-----------------------------------------------------------------------------
// ProfileValue wrapper

ProfileValue::ProfileValue(String init)
    : impl(std::make_unique<ProfileValueImpl>(init)) {}
ProfileValue::ProfileValue(Vector init)
    : impl(std::make_unique<ProfileValueImpl>(init)) {}
ProfileValue::ProfileValue(Map init)
    : impl(std::make_unique<ProfileValueImpl>(init)) {}
ProfileValue::ProfileValue(Number init)
    : impl(std::make_unique<ProfileValueImpl>(init)) {}
ProfileValue::ProfileValue(Boolean init)
    : impl(std::make_unique<ProfileValueImpl>(init)) {}
ProfileValue::~ProfileValue() = default;

ProfileValue::ProfileValue(const ProfileValue &other)
    : impl(std::make_unique<ProfileValueImpl>(*other.impl)) {}
ProfileValue::ProfileValue(ProfileValue &&other) noexcept
    : impl(std::move(other.impl)) {}

ProfileValue &ProfileValue::operator=(const ProfileValue &other) {
  impl = std::make_unique<ProfileValueImpl>(*other.impl);
  return *this;
}

ProfileValue &ProfileValue::operator=(ProfileValue &&other) noexcept {
  impl = std::move(other.impl);
  return *this;
}

ProfileValue &ProfileValue::operator=(Boolean init) {
  impl = std::make_unique<ProfileValueImpl>(init);
  return *this;
}
ProfileValue &ProfileValue::operator=(Number init) {
  impl = std::make_unique<ProfileValueImpl>(init);
  return *this;
}
ProfileValue &ProfileValue::operator=(String init) {
  impl = std::make_unique<ProfileValueImpl>(init);
  return *this;
}
ProfileValue &ProfileValue::operator=(Vector init) {
  impl = std::make_unique<ProfileValueImpl>(init);
  return *this;
}
ProfileValue &ProfileValue::operator=(Map init) {
  impl = std::make_unique<ProfileValueImpl>(init);
  return *this;
}

//-----------------------------------------------------------------------------
// DebugInfo wrapper

DebugInfo::DebugInfo(const DebugContext &dc, const std::string &layer)
    : impl(std::make_unique<DebugInfoImpl>(*dc.impl, layer)) {}

DebugInfo::~DebugInfo() = default;

DebugId DebugInfo::getId() const { return impl->getId(); }
std::string DebugInfo::getPathName() const { return impl->getPathName(); }

bool DebugInfo::setValue(std::string name, ProfileValue value) {
  return impl->setValue(name, *value.impl);
}

void DebugInfo::initializeStreamer(const std::string &fileName,
                                   const SerializationFormat &format) {

  poplar::DebugSerializationFormat poplarFormat =
      poplar::DebugSerializationFormat::CBOR;
  switch (format) {
  case SerializationFormat::CBOR:
    poplarFormat = poplar::DebugSerializationFormat::CBOR;
    break;
  case SerializationFormat::JSON:
    poplarFormat = poplar::DebugSerializationFormat::JSON;
    break;
  }

  poplar::DebugInfo::initializeStreamer(fileName, poplarFormat);
}

void DebugInfo::closeStreamer() { poplar::DebugInfo::closeStreamer(); }

//-----------------------------------------------------------------------------
// DebugNameAndId wrapper

DebugNameAndId::DebugNameAndId(std::string name,
                               DebugId debugId,
                               std::string parentPath)
    : impl(std::make_unique<DebugNameAndIdImpl>(name, debugId, parentPath)) {}
DebugNameAndId::DebugNameAndId(const char *name)
    : impl(std::make_unique<DebugNameAndIdImpl>(name)) {}
DebugNameAndId::DebugNameAndId(DebugId debugId)
    : impl(std::make_unique<DebugNameAndIdImpl>(debugId)) {}
DebugNameAndId::DebugNameAndId(const DebugInfo &debugInfo, std::string name)
    : impl(std::make_unique<DebugNameAndIdImpl>(*debugInfo.impl, name)) {}
DebugNameAndId::DebugNameAndId(const DebugNameAndId &debugNameAndId,
                               std::string name)
    : impl(std::make_unique<DebugNameAndIdImpl>(*debugNameAndId.impl, name)) {}

DebugNameAndId &DebugNameAndId::operator=(const DebugNameAndId &other) {
  impl->dnai = other.impl->dnai;
  return *this;
}
DebugNameAndId::~DebugNameAndId() {}
std::string DebugNameAndId::getPathName() const {
  return impl->dnai.getPathName();
}

//-----------------------------------------------------------------------------
// DebugContext wrapper

DebugContext::DebugContext(SourceLocation loc)
    : impl(std::make_unique<DebugContextImpl>("", loc)) {}
DebugContext::DebugContext(const char *name, SourceLocation loc)
    : impl(std::make_unique<DebugContextImpl>(name, loc)) {}
DebugContext::DebugContext(std::string name, SourceLocation loc)
    : impl(std::make_unique<DebugContextImpl>(name, loc)) {}
DebugContext::DebugContext(const DebugInfo &debugInfo,
                           std::string name,
                           SourceLocation loc)
    : impl(std::make_unique<DebugContextImpl>(*debugInfo.impl, name, loc)) {}
DebugContext::DebugContext(const DebugNameAndId &debugNameAndId,
                           std::string name,
                           SourceLocation loc)
    : impl(
          std::make_unique<DebugContextImpl>(*debugNameAndId.impl, name, loc)) {
}
std::string DebugContext::getPathName() const { return impl->getPathName(); }

DebugContext::DebugContext(DebugContext &&) = default;
DebugContext::~DebugContext()               = default;
DebugContext::DebugContext(const DebugContext &dc)
    : DebugContext(dc.getPathName(), *(dc.impl->loc)) {}
