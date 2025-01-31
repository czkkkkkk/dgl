/*!
 *  Copyright (c) 2022 by Contributors
 */
#include "./bin_stream.h"

#include <dmlc/logging.h>
#include <string>
#include <vector>

namespace dgl {
namespace dsf {

BinStream::BinStream() : front_(0) {}

BinStream::BinStream(size_t sz) : front_(0) { buffer_.resize(sz); }

BinStream::BinStream(const char *src, size_t sz) : front_(0) {
  push_back_bytes(src, sz);
}

BinStream::BinStream(const std::vector<char> &v) : front_(0), buffer_(v) {}

BinStream::BinStream(std::vector<char> &&v)
    : front_(0), buffer_(std::move(v)) {}

BinStream::BinStream(const BinStream &stream) {
  front_ = stream.front_;
  buffer_ = stream.buffer_;
}

BinStream::BinStream(BinStream &&stream)
    : front_(stream.front_), buffer_(std::move(stream.buffer_)) {
  stream.front_ = 0;
}

BinStream &BinStream::operator=(BinStream &&stream) {
  front_ = stream.front_;
  buffer_ = std::move(stream.buffer_);
  stream.front_ = 0;
  return *this;
}

size_t BinStream::hash() {
  size_t ret = 0;
  for (auto &i : buffer_) ret += i;
  return ret;
}

void BinStream::clear() {
  buffer_.clear();
  front_ = 0;
}

void BinStream::purge() {
  std::vector<char> tmp;
  buffer_.swap(tmp);
  front_ = 0;
}

void BinStream::resize(size_t size) {
  buffer_.resize(size);
  front_ = 0;
}

void BinStream::seek(size_t pos) { front_ = pos; }

void BinStream::push_back_bytes(const char *src, size_t sz) {
  buffer_.insert(buffer_.end(), (const char *)src, (const char *)src + sz);
}

void *BinStream::pop_front_bytes(size_t sz) {
  CHECK_LE(front_, buffer_.size());
  CHECK_LE(front_ + sz, buffer_.size());
  void *ret = &buffer_[front_];
  front_ += sz;
  return ret;
}

void BinStream::append(const BinStream &stream) {
  push_back_bytes(stream.get_remained_buffer(), stream.size());
}

BinStream &operator<<(BinStream &stream, const BinStream &bin) {
  stream << bin.size();
  stream.push_back_bytes(bin.get_remained_buffer(), bin.size());
  return stream;
}

BinStream &operator>>(BinStream &stream, BinStream &bin) {
  size_t len;
  stream >> len;
  bin.resize(len);
  for (char *i = bin.get_buffer(); len--; i++) stream >> *i;
  return stream;
}

BinStream &operator<<(BinStream &stream, const std::string &x) {
  stream << x.size();
  stream.push_back_bytes(x.data(), x.length());
  return stream;
}

BinStream &operator>>(BinStream &stream, std::string &x) {
  size_t len;
  stream >> len;
  std::string s(reinterpret_cast<char *>(stream.pop_front_bytes(len)), len);
  x.swap(s);
  return stream;
}

BinStream &operator<<(BinStream &stream, const std::vector<bool> &v) {
  size_t len = v.size();
  stream << len;
  for (int i = 0; i < v.size(); ++i) stream << static_cast<bool>(v[i]);
  return stream;
}

BinStream &operator>>(BinStream &stream, std::vector<bool> &v) {
  size_t len;
  stream >> len;
  v.clear();
  v.resize(len);
  bool bool_tmp;
  for (int i = 0; i < v.size(); ++i) {
    stream >> bool_tmp;
    v[i] = bool_tmp;
  }
  return stream;
}

BinStream::~BinStream() {}

}  // namespace dsf
}  // namespace dgl
