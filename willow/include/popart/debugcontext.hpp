#ifndef GUARD_NEURALNET_DEBUGCONTEXT_HPP
#define GUARD_NEURALNET_DEBUGCONTEXT_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace popart {

#if defined(__clang__)
#define SUPPORTS_LOCATION_BUILTINS                                             \
  (__has_builtin(__builtin_FUNCTION) && __has_builtin(__builtin_FILE) &&       \
   __has_builtin(__builtin_LINE))
#elif __GNUC__ >= 7
#define SUPPORTS_LOCATION_BUILTINS 1
#else
#define SUPPORTS_LOCATION_BUILTINS 0
#endif

// TODO: not needed with C++20
// Need to redefine SourceLocation so as to hold the filename & function
// as an std::string. When used from python the strings are dynamic and
// are deleted before they are written out by poplar.
class SourceLocation {
  std::string functionName;
  std::string fileName;
  unsigned lineNumber;
  bool valid{false};

public:
  SourceLocation() = default;
  SourceLocation(const std::string &functionName,
                 const std::string &fileName,
                 unsigned lineNumber)
      : functionName{functionName}, fileName{fileName},
        lineNumber{lineNumber}, valid{true} {}

  const std::string &getFunctionName() const { return functionName; }
  const std::string &getFileName() const { return fileName; }
  unsigned getLineNumber() const { return lineNumber; }
  bool isValid() const { return valid; }
#if SUPPORTS_LOCATION_BUILTINS
  // __builtin_FUNCTION etc get the information about the caller, where as
  // __func__ and __FILE__ etc get the name of the callee.
  // Can move to https://en.cppreference.com/w/cpp/utility/source_location when
  // using C++20
  static SourceLocation Current(const char *functionName = __builtin_FUNCTION(),
                                const char *fileName     = __builtin_FILE(),
                                unsigned lineNumber      = __builtin_LINE()) {
    return {functionName, fileName, lineNumber};
  }
#else
  static SourceLocation Current() { return {}; }
#endif
};

class DebugContext;
class DebugInfo;

struct ProfileValueImpl;

class ProfileValue {

  friend DebugInfo;
  friend ProfileValueImpl;
  std::unique_ptr<ProfileValueImpl> impl;

public:
  using Boolean = bool;
  using Number  = double;
  using String  = std::string;
  using Vector  = std::vector<ProfileValue>;
  using Map     = std::map<std::string, ProfileValue>;

  // Default value is 0.0.
  ProfileValue() : ProfileValue(0.0) {}

  ProfileValue(String init);
  ProfileValue(Vector init);
  ProfileValue(Map init);
  ProfileValue(Number init);
  explicit ProfileValue(Boolean init);

  // Disambiguate cast from integral type to Number
  template <
      class T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  ProfileValue(T init) : ProfileValue(ProfileValue::Number(init)) {}

  // Disambiguate conversion from const char* to bool, which is preferred over
  // std::string
  ProfileValue(const char *init) : ProfileValue(ProfileValue::String(init)) {}

  ~ProfileValue();

  ProfileValue(const ProfileValue &other);
  ProfileValue(ProfileValue &&other) noexcept;

  ProfileValue &operator=(const ProfileValue &other);
  ProfileValue &operator=(ProfileValue &&other) noexcept;
  ProfileValue &operator=(Boolean init);
  ProfileValue &operator=(Number init);
  ProfileValue &operator=(String init);
  ProfileValue &operator=(Vector init);
  ProfileValue &operator=(Map init);

  // Disambiguate cast from any non-boolean integral type to Number
  template <
      class T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  ProfileValue &operator=(T init) {
    return operator=(ProfileValue::Number(init));
  }
};

using DebugId = std::uint64_t;

struct DebugInfoImpl;
class DebugNameAndId;

class DebugInfo {
  friend DebugNameAndId;
  friend DebugContext;

  std::unique_ptr<DebugInfoImpl> impl;

public:
  DebugInfo(const DebugContext &debugContext, const std::string &layer);

  // Need to delete these as you can not copy DebugInfo's as per the base class
  // DebugInfo is written when this class is deleted, so we don't want
  // duplicates.
  DebugInfo &operator=(const DebugInfo &) = delete;
  DebugInfo(const DebugInfo &)            = delete;
  virtual ~DebugInfo();

  DebugId getId() const;
  std::string getPathName() const;
  bool setValue(std::string name, ProfileValue value);

  enum class SerializationFormat {
    JSON, ///< Serialise in JSON format
    CBOR, ///< Serialise in CBOR format
  };

  static void initializeStreamer(
      const std::string &fileName,
      const SerializationFormat &format = SerializationFormat::CBOR);
  static void closeStreamer();
};

struct DebugNameAndIdImpl;

class DebugNameAndId {

  friend DebugContext;

  std::unique_ptr<DebugNameAndIdImpl> impl;

public:
  DebugNameAndId(std::string name       = "",
                 DebugId debugId        = {},
                 std::string parentPath = "");
  DebugNameAndId(const char *name);
  DebugNameAndId(DebugId debugId);
  DebugNameAndId(const DebugInfo &debugInfo, std::string name = "");
  DebugNameAndId(const DebugNameAndId &DebugNameAndId, std::string name = "");
  DebugNameAndId &operator=(const DebugNameAndId &other);
  ~DebugNameAndId();
  std::string getPathName() const;
};

struct DebugContextImpl;

// Need to wrap poplar debugContext as we want to hold the popart
// SourceLocation.
class DebugContext {

  friend DebugInfo;
  std::unique_ptr<DebugContextImpl> impl;

public:
  DebugContext(SourceLocation loc = SourceLocation::Current());
  DebugContext(const char *name,
               SourceLocation loc = SourceLocation::Current());
  DebugContext(std::string name,
               SourceLocation loc = SourceLocation::Current());
  DebugContext(const DebugInfo &debugInfo,
               std::string name   = "",
               SourceLocation loc = SourceLocation::Current());
  DebugContext(const DebugNameAndId &debugNameAndId,
               std::string name   = "",
               SourceLocation loc = SourceLocation::Current());
  DebugContext(DebugContext &&);

  ~DebugContext();
  std::string getPathName() const;

private:
  DebugContext(const DebugContext &other) = default;
};

} // namespace popart

#endif