// nano-infer: CUDA resource wrappers and error handling.
//
// Everything here is move-only RAII. No raw cudaMalloc / cudaFree pairs should
// appear anywhere else in the codebase.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace nanoinfer {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const std::string& what)
      : std::runtime_error(what), code_(code) {}
  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

inline void cuda_check(cudaError_t err, const char* file, int line,
                       const char* expr) {
  if (err != cudaSuccess) {
    throw CudaError(err, std::string(file) + ":" + std::to_string(line) + ": " +
                             expr + " -> " + cudaGetErrorString(err));
  }
}

// Wrap every runtime API call.
#define CUDA_CHECK(expr) \
  ::nanoinfer::cuda_check((expr), __FILE__, __LINE__, #expr)

// Call immediately after a kernel launch. Catches launch-configuration errors
// without forcing a device synchronise.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// ---------------------------------------------------------------------------
// Alignment helper
// ---------------------------------------------------------------------------

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    release();
    if (count == 0) return;
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
    ptr_ = static_cast<T*>(raw);
    count_ = count;
  }

  // Destructors must not throw. A failed free during teardown is swallowed
  // deliberately, since the process is already unwinding.
  void release() noexcept {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* get() noexcept { return ptr_; }
  const T* get() const noexcept { return ptr_; }
  std::size_t count() const noexcept { return count_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }
  bool empty() const noexcept { return count_ == 0; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Pinned host memory
// ---------------------------------------------------------------------------
//
// Page-locked staging buffers let cudaMemcpyAsync run on the copy engine and
// overlap with compute. Copying from pageable memory blocks the calling thread
// and roughly halves achievable PCIe bandwidth.

template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;

  explicit PinnedBuffer(std::size_t count) { allocate(count); }

  ~PinnedBuffer() { release(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    release();
    if (count == 0) return;
    void* raw = nullptr;
    CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault));
    ptr_ = static_cast<T*>(raw);
    count_ = count;
  }

  void release() noexcept {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t count() const noexcept { return count_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Streams and events
// ---------------------------------------------------------------------------

class CudaStream {
 public:
  CudaStream() {
    // Non-blocking streams never implicitly synchronise with the legacy default
    // stream, which keeps copy and compute overlap intact.
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~CudaStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_ != nullptr) cudaStreamDestroy(stream_);
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const noexcept { return stream_; }
  void sync() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  explicit CudaEvent(unsigned flags = cudaEventDisableTiming) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
  }

  ~CudaEvent() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  cudaEvent_t get() const noexcept { return event_; }
  void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(event_, stream)); }

  // Synchronising on an event that was never recorded returns immediately, so
  // the first pass of a double-buffered loop needs no special case.
  void sync() const { CUDA_CHECK(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_ = nullptr;
};

// ---------------------------------------------------------------------------
// Bump arena over a single device allocation
// ---------------------------------------------------------------------------
//
// Model weights, KV cache and activation scratch each live in one contiguous
// allocation. One cudaMalloc instead of several hundred removes allocator
// overhead at start-up and keeps related tensors close in the address space.

class DeviceArena {
 public:
  DeviceArena() = default;

  explicit DeviceArena(std::size_t bytes) : buffer_(bytes) {}

  void reset(std::size_t bytes) {
    buffer_.allocate(bytes);
    used_ = 0;
  }

  void rewind() noexcept { used_ = 0; }

  std::byte* alloc(std::size_t bytes, std::size_t alignment = 256) {
    const std::size_t start = align_up(used_, alignment);
    if (start + bytes > buffer_.count()) {
      throw std::runtime_error("DeviceArena: out of capacity, requested " +
                               std::to_string(bytes) + " bytes with " +
                               std::to_string(buffer_.count() - start) +
                               " remaining");
    }
    used_ = start + bytes;
    return buffer_.get() + start;
  }

  std::byte* base() noexcept { return buffer_.get(); }
  std::size_t capacity() const noexcept { return buffer_.count(); }
  std::size_t used() const noexcept { return used_; }

 private:
  DeviceBuffer<std::byte> buffer_;
  std::size_t used_ = 0;
};

}  // namespace nanoinfer
