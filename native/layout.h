#pragma once

#include <sstream>
#include <memory>
#include <vector>
#include <string>

#include "macro.h"

namespace witin {

#define WITIN_MAX_DIM 6

template <typename scalar_t, size_t MaxDim = WITIN_MAX_DIM>
struct SimpleVector {
public:
    INLINE_HOST_DEVICE SimpleVector() {};

    INLINE_HOST_DEVICE SimpleVector(std::initializer_list<scalar_t> q) {
        WITIN_ASSERT(q.size() <= MaxDim);
        mSize = 0;
        for (scalar_t s : q) {
            mArray[mSize++] = s;
        }
        mSize = q.size();
    }

    INLINE_HOST_DEVICE SimpleVector(size_t size, scalar_t val) {
        WITIN_ASSERT(size <= MaxDim);
        for (size_t i = 0; i < size; ++i) {
            mArray[i] = val;
        }
        mSize = size;
    }

    SimpleVector(const std::vector<scalar_t>& arr) {
        WITIN_ASSERT(arr.size() <= MaxDim);
        for (size_t i = 0; i < arr.size(); ++i) {
            mArray[i] = arr[i];
        }
        mSize = arr.size();
    }

    INLINE_HOST_DEVICE SimpleVector(const SimpleVector<scalar_t, MaxDim>& arr) {
        WITIN_ASSERT(arr.size() <= MaxDim);
        for (size_t i = 0; i < arr.size(); ++i) {
            mArray[i] = arr[i];
        }
        mSize = arr.size();
    }

    INLINE_HOST_DEVICE scalar_t& operator[](int idx) {
#ifdef WITIN_DEBUG
        WITIN_ASSERT(idx >= 0 && idx < mSize);
#endif
        return mArray[idx];
    }

    INLINE_HOST_DEVICE const scalar_t& operator[](int idx) const {
#ifdef WITIN_DEBUG
        WITIN_ASSERT(idx >= 0 && idx < mSize);
#endif
        return mArray[idx];
    }
  
    INLINE_HOST_DEVICE void push_back(scalar_t s) {
#ifdef WITIN_DEBUG
        WITIN_ASSERT(mSize < MaxDim);
#endif
        mArray[mSize] = s;
        mSize++;
    }

    INLINE_HOST_DEVICE void pop_back() {
#ifdef WITIN_DEBUG
        WITIN_ASSERT(mSize > 0);
#endif
        mSize--;
    }

    INLINE_HOST_DEVICE size_t size() const { return mSize; }
    INLINE_HOST_DEVICE const scalar_t* data() const { return mArray; }
    INLINE_HOST_DEVICE size_t empty() const { return mSize == 0; }

    class iterator {
    public:
        typedef iterator  self_type;
        typedef scalar_t  value_type;
        typedef scalar_t &reference;
        typedef scalar_t *pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef std::ptrdiff_t difference_type;

        INLINE_HOST_DEVICE iterator(pointer ptr) : ptr_(ptr) {}
        // iterator ++
        INLINE_HOST_DEVICE self_type operator++(int junk) {
            self_type i = *this;
            ptr_++;
            return i;
        }
        // ++ iterator
        INLINE_HOST_DEVICE self_type operator++() {
            ptr_++;
            return *this;
        }
        INLINE_HOST_DEVICE reference operator*() { return *ptr_; }
        INLINE_HOST_DEVICE pointer operator->() { return ptr_; }
        INLINE_HOST_DEVICE bool operator==(const self_type& rhs) {
            return ptr_ == rhs.ptr_;
        }
        INLINE_HOST_DEVICE bool operator!=(const self_type& rhs) {
            return ptr_ != rhs.ptr_;
        }

    private:
        pointer ptr_;
    };

    class const_iterator {
    public:
        typedef const_iterator self_type;
        typedef scalar_t value_type;
        typedef const scalar_t &reference;
        typedef const scalar_t *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::forward_iterator_tag iterator_category;

        INLINE_HOST_DEVICE const_iterator(pointer ptr) : ptr_(ptr) {}
        // const_iterator ++
        INLINE_HOST_DEVICE self_type operator++(int junk) {
            self_type i = *this;
            ptr_++;
            return i;
        }
        // ++ const_iterator
        INLINE_HOST_DEVICE self_type operator++() {
            ptr_++;
            return *this;
        }
        INLINE_HOST_DEVICE reference operator*() { return *ptr_; }
        INLINE_HOST_DEVICE pointer operator->() { return ptr_; }
        INLINE_HOST_DEVICE bool operator==(const self_type& rhs) {
            return ptr_ == rhs.ptr_;
        }
        INLINE_HOST_DEVICE bool operator!=(const self_type& rhs) {
            return ptr_ != rhs.ptr_;
        }

    private:
        pointer ptr_;
    };

    INLINE_HOST_DEVICE iterator begin() { return iterator(mArray); }
    INLINE_HOST_DEVICE iterator end() { return iterator(mArray + mSize); }
    INLINE_HOST_DEVICE const_iterator begin() const { return const_iterator(mArray); }
    INLINE_HOST_DEVICE const_iterator end() const { return const_iterator(mArray + mSize); }

protected:
    scalar_t mArray[MaxDim];
    size_t mSize = 0;
};

template <typename scalar_t, size_t MaxDim>
bool operator==(const SimpleVector<scalar_t, MaxDim> &lfs, const SimpleVector<scalar_t, MaxDim> &rfs) {
    if (lfs.size() != rfs.size()) return false;

    for (size_t i = 0; i < lfs.size(); ++i) {
        if (lfs[i] != rfs[i]) return false;
    }
    return true;
}

template <typename scalar_t, size_t MaxDim>
bool operator!=(const SimpleVector<scalar_t, MaxDim> &lfs, const SimpleVector<scalar_t, MaxDim> &rfs) {
    return !(lfs == rfs);
}

template <size_t MaxDim = WITIN_MAX_DIM>
struct ShapeBase : public SimpleVector<int, MaxDim> {
    INLINE_HOST_DEVICE ShapeBase() : SimpleVector<int, MaxDim>() {};
    INLINE_HOST_DEVICE ShapeBase(std::initializer_list<int> shape) : SimpleVector<int, MaxDim>(shape) {}

    INLINE_HOST_DEVICE ShapeBase(const ShapeBase<MaxDim>& shape) : SimpleVector<int, MaxDim>(shape) {}
    ShapeBase<MaxDim>& operator=(const ShapeBase<MaxDim>& shape) = default;

    INLINE_HOST_DEVICE ShapeBase(size_t size, int val = 0) : SimpleVector<int, MaxDim>(size, val) {}
    ShapeBase(const std::vector<int>& arr) : SimpleVector<int, MaxDim>(arr) {}

    INLINE_HOST_DEVICE void resize(size_t size, int val = 0) {
        WITIN_ASSERT(size <= MaxDim);

        for (size_t i = this->mSize; i < size; ++i) {
            this->mArray[i] = val;
        }
        this->mSize = size; 
    }
};

using SmallVector = ShapeBase<WITIN_MAX_DIM>;

inline std::string to_string(const SmallVector& v) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) out << ", ";
        out << v[i];
    }
    out << "]";
    return out.str();
}

struct TensorLayout {
    TensorLayout(
        void* _data,
        const SmallVector& _shape,
        const SmallVector& _stride,
        const bool _is_output = false
    ) : 
        data(_data),
        shape(std::move(_shape)),
        stride(std::move(_stride)),
        is_output(_is_output)
    {}

    TensorLayout(const TensorLayout& other)
        : data(other.data),
          shape(other.shape),
          stride(other.stride),
          is_output(other.is_output)
    {}

    TensorLayout(TensorLayout&& other) noexcept
        : data(other.data),
          shape(std::move(other.shape)),
          stride(std::move(other.stride)),
          is_output(other.is_output)
    {
        // 防止悬空指针
        other.data = nullptr;
    }

    // TODO: add more functions to calculate offset.
    INLINE_HOST_DEVICE int operator()(int idx) const {
        int offset = 0;

        #pragma unroll
        for (int dim = 0; dim < shape.size(); ++dim) { 
            int64_t div = idx / shape[dim];
            int64_t mod = idx % shape[dim];
            idx = div;

            offset += mod * stride[dim];
        }

        return offset;
    }

    void* data;
    SmallVector shape;
    SmallVector stride;

    bool is_output;

private:
    TensorLayout() = delete;
};

inline std::string to_string(const TensorLayout& layout) {
    std::ostringstream os;
    os << "(";
    os << "shape=" << to_string(layout.shape) << ", ";
    os << "stride=" << to_string(layout.stride) << ", ";
    os << "type=" << (layout.is_output ? "output" : "input");
    os << ")";
    return os.str();
}

using TensorLayoutPtr = std::shared_ptr<TensorLayout>;

template <typename... Args>
TensorLayoutPtr SetTensorLayout(Args&&... args) {
    return std::make_shared<TensorLayout>(std::forward<Args>(args)...);
}

}; // end of namespace witin