// nano-infer tests: minimal .npy reader.
//
// Reference activations come out of PyTorch as .npy. Pulling in a library to
// read them would add a build dependency for the sake of a 20-byte header, so
// this parses the format directly.
//
// Format, from the NumPy spec:
//   bytes 0-5    magic "\x93NUMPY"
//   byte  6      major version
//   byte  7      minor version
//   v1: 2-byte little-endian header length, v2 and v3: 4-byte
//   header       a Python dict literal, space-padded, newline terminated
//   data         raw elements
//
// Only C-contiguous little-endian float32 and float64 are accepted. Anything
// else throws rather than silently misreading, because a shape or dtype
// surprise in a numerical test is worse than a hard failure.
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanoinfer::test {

struct NpyArray {
  std::vector<float> data;
  std::vector<std::size_t> shape;

  std::size_t size() const { return data.size(); }

  std::size_t rows() const {
    // Everything but the last axis, flattened. Reference tensors arrive as
    // [batch, tokens, features] and the kernels see [tokens, features], so the
    // leading axes collapse.
    std::size_t n = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) n *= shape[i];
    return shape.empty() ? 0 : n;
  }

  std::size_t cols() const { return shape.empty() ? 0 : shape.back(); }

  std::string shape_string() const {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) out += ", ";
      out += std::to_string(shape[i]);
    }
    return out + "]";
  }
};

namespace detail {

[[noreturn]] inline void npy_fail(const std::string& path,
                                  const std::string& reason) {
  throw std::runtime_error("npy '" + path + "': " + reason);
}

// Pulls the value that follows a key in the header dict. The header is a fixed
// literal produced by NumPy, not arbitrary Python, so locating the key and
// reading to the delimiter is enough. A real parser would be more code for no
// extra safety here.
inline std::string header_field(const std::string& header, const std::string& key,
                                char open_delim, char close_delim) {
  const std::string quoted = "'" + key + "'";
  const std::size_t key_pos = header.find(quoted);
  if (key_pos == std::string::npos) return {};

  // Search past the key itself. Starting at key_pos would match the closing
  // quote of the key when the delimiter is also a quote.
  const std::size_t value_pos = key_pos + quoted.size();
  const std::size_t start = header.find(open_delim, value_pos);
  if (start == std::string::npos) return {};

  const std::size_t end = header.find(close_delim, start + 1);
  if (end == std::string::npos) return {};

  return header.substr(start + 1, end - start - 1);
}

}  // namespace detail

inline NpyArray load_npy(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    detail::npy_fail(path, "cannot open, run tools/dump_reference.py first");
  }

  char magic[6] = {};
  file.read(magic, 6);
  if (!file || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    detail::npy_fail(path, "bad magic, not a .npy file");
  }

  std::uint8_t major = 0;
  std::uint8_t minor = 0;
  file.read(reinterpret_cast<char*>(&major), 1);
  file.read(reinterpret_cast<char*>(&minor), 1);

  std::size_t header_len = 0;
  if (major == 1) {
    std::uint16_t len16 = 0;
    file.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else if (major == 2 || major == 3) {
    std::uint32_t len32 = 0;
    file.read(reinterpret_cast<char*>(&len32), 4);
    header_len = len32;
  } else {
    detail::npy_fail(path, "unsupported version " + std::to_string(major));
  }

  std::string header(header_len, '\0');
  file.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!file) detail::npy_fail(path, "truncated header");

  const std::string descr = detail::header_field(header, "descr", '\'', '\'');
  if (descr != "<f4" && descr != "<f8" && descr != "|f4") {
    detail::npy_fail(path, "dtype '" + descr +
                               "' is unsupported, save as float32 or float64");
  }

  if (header.find("'fortran_order': False") == std::string::npos) {
    detail::npy_fail(path, "Fortran order is unsupported, save C-contiguous");
  }

  NpyArray array;
  const std::string shape_text = detail::header_field(header, "shape", '(', ')');
  std::string number;
  for (char c : shape_text + ",") {
    if (c >= '0' && c <= '9') {
      number += c;
    } else if (!number.empty()) {
      array.shape.push_back(std::stoull(number));
      number.clear();
    }
  }

  std::size_t count = 1;
  for (std::size_t dim : array.shape) count *= dim;
  if (array.shape.empty()) count = 1;  // zero-rank scalar

  array.data.resize(count);
  if (descr == "<f8") {
    // Reference dumps stay in float32, but a double file is worth reading
    // rather than rejecting, since it costs four lines.
    std::vector<double> wide(count);
    file.read(reinterpret_cast<char*>(wide.data()),
              static_cast<std::streamsize>(count * sizeof(double)));
    for (std::size_t i = 0; i < count; ++i) {
      array.data[i] = static_cast<float>(wide[i]);
    }
  } else {
    file.read(reinterpret_cast<char*>(array.data.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
  }

  if (!file) detail::npy_fail(path, "truncated data, expected " +
                                        std::to_string(count) + " elements");

  return array;
}

}  // namespace nanoinfer::test
