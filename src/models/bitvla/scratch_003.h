#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace robotcpp::bitvla {
namespace {

struct SmeScratchStorage003 {
    std::unique_ptr<int8_t[]> weights;
    std::unique_ptr<int8_t[]> activations;
    std::unique_ptr<int32_t[]> products;
    size_t weight_capacity = 0;
    size_t activation_capacity = 0;
    size_t product_capacity = 0;
    bool active = false;
};

class SmeScratchLease003 {
public:
    SmeScratchLease003(bool reuse, size_t weights, size_t activations, size_t products) : storage_(&private_) {
        if (reuse && weights <= 3u * 1024u * 1024u && activations <= 512u * 1024u && products <= 65536u) {
            static thread_local SmeScratchStorage003 retained;
            if (!retained.active) storage_ = &retained;
        }
        reserve(storage_->weights, storage_->weight_capacity, weights);
        reserve(storage_->activations, storage_->activation_capacity, activations);
        reserve(storage_->products, storage_->product_capacity, products);
        storage_->active = true;
    }

    ~SmeScratchLease003() { storage_->active = false; }

    SmeScratchLease003(const SmeScratchLease003 &) = delete;
    SmeScratchLease003 & operator=(const SmeScratchLease003 &) = delete;

    const std::unique_ptr<int8_t[]> & weights() const { return storage_->weights; }
    const std::unique_ptr<int8_t[]> & activations() const { return storage_->activations; }
    const std::unique_ptr<int32_t[]> & products() const { return storage_->products; }

private:
    template<class Value>
    static void reserve(std::unique_ptr<Value[]> & storage, size_t & capacity, size_t required) {
        if (required > capacity) {
            storage.reset(new Value[required]);
            capacity = required;
        }
    }

    SmeScratchStorage003 private_;
    SmeScratchStorage003 * storage_;
};

}
}
