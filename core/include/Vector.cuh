#pragma once

#include <cstddef>
#include <utility>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cuda_runtime.h>

inline void check_cuda(cudaError_t res) {
    if (res != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(res));
    }
}

template <typename AtomT>
class Data {
private:
    std::size_t size_{0};
    AtomT* data_{nullptr};

public:
    explicit Data(std::size_t size) : size_(size) {
        if (size_ > 0) check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
    }

    ~Data() {
        if (data_) cudaFree(data_);
    }

    Data(const Data& other) : Data(other.size_) {
        if (size_ > 0) check_cuda(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
    }

    Data(Data&& other) noexcept : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr)) {}

    Data& operator=(Data other) noexcept {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
        return *this;
    }

    AtomT* data() noexcept { return data_; }
    const AtomT* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ && host_ptr) check_cuda(cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost));
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ && host_ptr) check_cuda(cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice));
    }
};

template <typename AtomT>
class VectorView {
private:
    AtomT* data_{nullptr};
    std::size_t size_{0};

public:
    __host__ __device__ VectorView() {}
    __host__ __device__ VectorView(AtomT* data, std::size_t size) : data_(data), size_(size) {}

    template <typename OtherT>
    requires std::is_convertible_v<OtherT*, AtomT*>
    __host__ __device__ VectorView(const VectorView<OtherT>& o) : data_(o.data()), size_(o.size()) {}

    __host__ __device__ std::size_t size() const noexcept { return size_; }
    __host__ __device__ AtomT* data() noexcept { return data_; }
    __host__ __device__ const AtomT* data() const noexcept { return data_; }

    __host__ __device__ AtomT& operator[](std::size_t n) noexcept { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const noexcept { return data_[n]; }
    __host__ __device__ AtomT& operator()(std::size_t i) noexcept { return data_[i]; }
    __host__ __device__ const AtomT& operator()(std::size_t i) const noexcept { return data_[i]; }
};

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<const AtomT> a, VectorView<const AtomT> b, VectorView<AtomT> c) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < a.size()) c[idx] = a[idx] + b[idx];
}

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)), view_(data_->data(), size) {}

    std::size_t size() const noexcept { return view_.size(); }
    AtomT* data() noexcept { return view_.data(); }
    const AtomT* data() const noexcept { return view_.data(); }

    VectorView<AtomT> view() noexcept { return view_; }
    VectorView<const AtomT> view() const noexcept { return VectorView<const AtomT>(view_.data(), view_.size()); }

    void copy_to_host(AtomT* host_ptr) const { data_->copy_to_host(host_ptr); }
    void copy_from_host(const AtomT* host_ptr) { data_->copy_from_host(host_ptr); }
};

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& a, const Vector<AtomT>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector dimensions mismatch");
    const std::size_t n = a.size();
    Vector<AtomT> res(n);
    if (n == 0) return res;

    constexpr unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);

    kernel_vecadd<AtomT><<<grid, block>>>(a.view(), b.view(), res.view());
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    return res;
}