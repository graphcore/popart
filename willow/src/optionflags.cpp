#include <boost/lexical_cast.hpp>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

#include <poponnx/optionflags.hpp>

namespace willow {

eOptions OptionFlags::getEOption(std::string const &option) {
  if (option == "exportDot")
    return e_exportDot;
  // if (option == "anotherOpt") return e_anotherOpt;
}

void OptionFlags::updateOptions() {
  for (auto const &opt : userOptionsMap) {
    switch (getEOption(opt.first)) {
    case e_exportDot:
      options.exportDot = boost::lexical_cast<bool>(opt.second);
      break;
    default:
      throw std::invalid_argument("Invalid user option-value pair");
    }
  }
}

void OptionFlags::mapUserOptions() {
  std::string key, val, token;
  std::istringstream iss(userOptions);
  while (std::getline(iss, token, ',')) {
    size_t pos          = token.find('=');
    key                 = token.substr(0, pos);
    val                 = token.substr(pos + 1);
    userOptionsMap[key] = val;
  }
}

OptionFlags::OptionFlags() {}
OptionFlags::OptionFlags(std::string userOptions_) : userOptions(userOptions_) {

  mapUserOptions();
  updateOptions();
}

OptionFlags::~OptionFlags() = default;

} // namespace willow
