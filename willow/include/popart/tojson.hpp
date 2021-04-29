// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/logging.hpp>

void writeJSON(std::size_t value, std::ostream &ss);

template <typename KeyType, typename ValueType>
void writeJSON(const std::map<KeyType, ValueType> &m, std::ostream &ss);

template <typename ValueType>
void writeJSON(const std::vector<ValueType> &m, std::ostream &ss);

void writeJSON(std::size_t value, std::ostream &ss) { ss << value; }

template <typename ValueType>
void writeJSON(const std::pair<ValueType, ValueType> &p, std::ostream &ss) {
  writeJSON(std::vector<ValueType>{p.first, p.second}, ss);
}

template <typename Container, typename Writer>
void writeContainerToJson(const Container &c,
                          char start,
                          char end,
                          std::ostream &ss,
                          const Writer &w) {
  ss << start;

  int comma_counter = 0;

  for (auto &item : c) {
    w(item);

    if (comma_counter < c.size() - 1) {
      ss << ",";
      comma_counter++;
    }
  }

  ss << end;
}

template <typename KeyType, typename ValueType>
void writeJSON(const std::map<KeyType, ValueType> &m, std::ostream &ss) {
  writeContainerToJson(m, '{', '}', ss, [&ss](auto &i) {
    auto &key   = i.first;
    auto &value = i.second;

    ss << popart::logging::format("\"{}\":", key);
    writeJSON(value, ss);
  });
}

template <typename ValueType>
void writeJSON(const std::vector<ValueType> &v, std::ostream &ss) {
  writeContainerToJson(v, '[', ']', ss, [&ss](auto &i) { writeJSON(i, ss); });
}
