#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <string>

namespace willow {

// This struct contains all the user-overridable options for
// configuring the Ir class. They can be any type.
struct Options {
  bool exportDot;
};
enum eOptions { e_exportDot };

// User-options are supplied to the class constructor as a
// comma-separated list of key-value pairs. No whitespace is
// allowed after each comma. The options are parsed by this
// class, and used to override the default options for
// configuring the Ir class
class OptionFlags {
public:
  OptionFlags();
  OptionFlags(std::string);
  ~OptionFlags();

  // Initialize with default options:
  Options options = {
      0, // exportDot
  };

  std::string userOptions;
  std::map<std::string, std::string> userOptionsMap;

  // Converts comma separated key-value pair list into
  // string map of options
  void mapUserOptions();

  // Convert from user option to enum for switch-case statements
  eOptions getEOption(std::string const &option);

  // Update default options from userOptions input. Checks
  // for invalid user-option
  void updateOptions();
};

} // namespace willow

#endif
