#include "ops.h"
#include "i2s.h"

#include "ggml-backend.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/binary-ops.h"

#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#include <memory>
#include <new>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace robotcpp::bitvla {
namespace {
// Copy exact precomputed F32 lookup bytes. No normalization arithmetic here.
void vision_rgb_u8_f32_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * rgb = static_cast<const uint8_t *>(dst->src[0]->data);
    const auto * lut = static_cast<const float *>(dst->src[1]->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t plane = dst->ne[0] * dst->ne[1];
    for (int64_t p = ith; p < plane; p += nth) {
        output[p] = lut[rgb[3*p]];
        output[plane+p] = lut[256+rgb[3*p+1]];
        output[2*plane+p] = lut[512+rgb[3*p+2]];
    }
}


void softmax_f32_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    // Call the existing CPU kernel itself: preserve its vector exp, FP64 sum,
    // scaling, nonfinite handling and thread/row partition exactly.
    ggml_tensor original = *dst;
    original.op = GGML_OP_SOFT_MAX;
    original.src[1] = original.src[2] = nullptr;
    const float options[] = {1.0f, 0.0f};
    std::memcpy(original.op_params, options, sizeof(options));
    std::vector<float> workspace((dst->ne[0] + CACHE_LINE_SIZE_F32) * nth);
    ggml_compute_params params{};
    params.ith = ith;
    params.nth = nth;
    params.wsize = workspace.size() * sizeof(float);
    params.wdata = workspace.data();
    ggml_compute_forward_soft_max(&params, &original);
    auto * output = static_cast<float *>(dst->data);
    for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2)
            for (int64_t i1 = ith; i1 < dst->ne[1]; i1 += nth) {
                auto * row = output + ((i3 * dst->ne[2] + i2) * dst->ne[1] + i1) * dst->ne[0];
                for (int64_t col = 0; col < dst->ne[0]; ++col) row[col] = row[col];
            }
}


void softmax_score_f32_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * source = dst->src[0];
    const float scale = *static_cast<const float *>(userdata);
    const int64_t width = source->ne[0];
    std::vector<float> values(width);
#if defined(__APPLE__) && defined(__aarch64__)
    // CPU156: keep the same zeroed worker-local scratch, allocated with calloc.
    const size_t workspace_count = width + CACHE_LINE_SIZE_F32;
    std::unique_ptr<float, decltype(&std::free)> workspace(
        static_cast<float *>(std::calloc(workspace_count, sizeof(float))), &std::free);
    if (workspace_count != 0 && !workspace) throw std::bad_alloc();
    float * const workspace_data = workspace.get();
    const size_t workspace_bytes = workspace_count * sizeof(float);
#else
    std::vector<float> workspace(width + CACHE_LINE_SIZE_F32);
    float * const workspace_data = workspace.data();
    const size_t workspace_bytes = workspace.size() * sizeof(float);
#endif
    for (int64_t row = ith; row < ggml_nrows(source); row += nth) {
        const auto * input = static_cast<const float *>(source->data) + row * width;
        auto * output = static_cast<float *>(dst->data) + row * width;
        for (int64_t col = 0; col < width; ++col) values[col] = input[col] * scale;
        ggml_tensor transformed = *source, original = *dst;
        transformed.data = values.data(); original.data = output;
        transformed.ne[1] = transformed.ne[2] = transformed.ne[3] = 1;
        original.ne[1] = original.ne[2] = original.ne[3] = 1;
        for(int axis=1;axis<4;++axis) transformed.nb[axis]=original.nb[axis]=width*sizeof(float);
        original.op = GGML_OP_SOFT_MAX; original.src[0] = &transformed;
        original.src[1] = original.src[2] = nullptr;
        const float options[] = {1.0f, 0.0f}; std::memcpy(original.op_params, options, sizeof(options));
        ggml_compute_params params{}; params.ith=0; params.nth=1;
        params.wsize=workspace_bytes; params.wdata=workspace_data;
        ggml_compute_forward_soft_max(&params,&original);
    }
}

void rms_norm_f32_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * source = dst->src[0];
    const auto * input = static_cast<const char *>(source->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t width = source->ne[0];
    const float eps = *static_cast<const float *>(userdata);
    for (int64_t row = ith; row < ggml_nrows(dst); row += nth) {
        const int64_t i1 = row % source->ne[1];
        const int64_t i2 = (row / source->ne[1]) % source->ne[2];
        const int64_t i3 = row / (source->ne[1] * source->ne[2]);
        const auto * x = reinterpret_cast<const float *>(input + i1 * source->nb[1] + i2 * source->nb[2] + i3 * source->nb[3]);
        // Original CPU RMS: F32 square, left-to-right FP64 sum, then F32 scale.
        // Its vector-scale kernel gives a NaN scale priority over the input NaN.
        double sum = 0.0;
        for (int64_t col = 0; col < width; ++col) sum += double(x[col] * x[col]);
        const float mean = sum / width;
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int64_t col = 0; col < width; ++col)
            output[row * width + col] = std::isnan(scale) ? scale : x[col] * scale;
    }
}


void rms_affine_f32_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * source=dst->src[0];
    const auto * input=static_cast<const char *>(source->data);
    const auto * weight=static_cast<const float *>(dst->src[1]->data);
    auto * output=static_cast<float *>(dst->data);
    const int64_t width=source->ne[0]; const float eps=*static_cast<const float *>(userdata);
    const int64_t rows = ggml_nrows(dst);
    int64_t row = ith;
#if defined(__APPLE__) && defined(__aarch64__)
    const auto in_address = reinterpret_cast<uintptr_t>(source->data);
    const auto out_address = reinterpret_cast<uintptr_t>(dst->data);
    const auto weight_address = reinterpret_cast<uintptr_t>(weight);
    const size_t output_bytes = ggml_nbytes(dst);
    const auto disjoint = [&](uintptr_t address, size_t bytes) {
        return address <= out_address ? out_address - address >= bytes : address - out_address >= output_bytes;
    };
    if (ggml_is_contiguous(source) && disjoint(in_address, ggml_nbytes(source)) &&
        disjoint(weight_address, size_t(width) * sizeof(float))) {
        for (; row < rows && rows - row > nth; row += int64_t(nth) * 2) {
            const auto * x0 = reinterpret_cast<const float *>(input) + row * width;
            const auto * x1 = x0 + int64_t(nth) * width;
            auto sums = vdupq_n_f64(0.0);
            for (int64_t col = 0; col < width; ++col) {
                auto pair = vdup_n_f32(x0[col]);
                pair = vset_lane_f32(x1[col], pair, 1);
                sums = vaddq_f64(sums, vcvt_f64_f32(vmul_f32(pair, pair)));
            }
            double totals[2];
            vst1q_f64(totals, sums);
            if (!std::isfinite(totals[0]) || !std::isfinite(totals[1])) break;
            const float mean0 = totals[0] / width, mean1 = totals[1] / width;
            const float scale0 = 1.0f / std::sqrt(mean0 + eps), scale1 = 1.0f / std::sqrt(mean1 + eps);
            if (!std::isfinite(scale0) || !std::isfinite(scale1)) break;
            const auto store_row = [&](const float * x, float * out, float scale) {
                const auto scales = vdupq_n_f32(scale);
                int64_t col = 0;
                for (; col + 4 <= width; col += 4) {
                    const auto normalized = vmulq_f32(vld1q_f32(x + col), scales);
                    vst1q_f32(out + col, vmulq_f32(normalized, vld1q_f32(weight + col)));
                }
                for (; col < width; ++col) out[col] = (x[col] * scale) * weight[col];
            };
            store_row(x0, output + row * width, scale0);
            store_row(x1, output + (row + nth) * width, scale1);
        }
    }
#endif
    for (; row < rows; row += nth) {
        const int64_t i1=row%source->ne[1],i2=(row/source->ne[1])%source->ne[2],i3=row/(source->ne[1]*source->ne[2]);
        const auto * x=reinterpret_cast<const float *>(input+i1*source->nb[1]+i2*source->nb[2]+i3*source->nb[3]);
        double sum=0.0;
        for(int64_t col=0;col<width;++col) sum+=double(x[col]*x[col]);
        const float mean=sum/width,scale=1.0f/std::sqrt(mean+eps);
        for(int64_t col=0;col<width;++col) {
            const float normalized=std::isnan(scale)?scale:x[col]*scale;
            // Original ggml CPU MUL chooses its left NaN, as observed in035.
            const float product=std::isnan(normalized)?normalized:normalized*weight[col];
            output[row*width+col]=product;
        }
    }
}


// Preserve the original CPU RMS-affine callback and original encoder itself.
void rms_affine_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void * userdata) {
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    const auto * input=dst->src[0];const int64_t width=input->ne[0];
    std::vector<float> values(2*width);
    ggml_tensor row_input=*input,normalized=*input,encoded=*dst;
    normalized.data=values.data();normalized.src[0]=&row_input;normalized.src[1]=dst->src[1];encoded.src[0]=&normalized;
    for(int64_t row=int64_t(ith)*2;row<input->ne[1];row+=int64_t(nth)*2) {
        const int64_t count=std::min<int64_t>(2,input->ne[1]-row);
        for(int dim=1;dim<4;++dim) {
            row_input.ne[dim]=normalized.ne[dim]=encoded.ne[dim]=(dim==1?count:1);
            row_input.nb[dim]=normalized.nb[dim]=size_t(width)*sizeof(float)*(dim==1?1:count);
            encoded.nb[dim]=size_t(width+8)*(dim==1?1:count);
        }
        row_input.data=static_cast<char *>(input->data)+row*input->nb[1];
        encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];
        rms_affine_f32_cpu(&normalized,0,1,userdata);
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
#else
    const auto * input=dst->src[0];const int64_t width=input->ne[0];std::vector<float> values(width);
    ggml_tensor row_input=*input,normalized=*input,encoded=*dst;
    for(int dim=1;dim<4;++dim){row_input.ne[dim]=normalized.ne[dim]=encoded.ne[dim]=1;row_input.nb[dim]=normalized.nb[dim]=width*sizeof(float);encoded.nb[dim]=(width+8);}
    normalized.data=values.data();normalized.src[0]=&row_input;normalized.src[1]=dst->src[1];encoded.src[0]=&normalized;
    for(int64_t row=ith;row<input->ne[1];row+=nth) {
        row_input.data=static_cast<char *>(input->data)+row*input->nb[1];encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];
        rms_affine_f32_cpu(&normalized,0,1,userdata);i2s_encode_cpu(&encoded,0,1,nullptr);
    }
#endif
}

// Exact original CPU kernels, including their reduction and NaN operand order.
void layer_norm_affine_f32_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * input=dst->src[0];const auto * weight=dst->src[1];const auto * bias=dst->src[2];
    const int64_t width=input->ne[0];const float eps=*static_cast<const float *>(userdata);
    std::vector<float> normalized(width),product(width);
    ggml_tensor row_input=*input;
    for(int dim=1;dim<4;++dim){row_input.ne[dim]=1;row_input.nb[dim]=width*sizeof(float);}
    ggml_tensor row_norm=row_input;row_norm.op=GGML_OP_NORM;row_norm.src[0]=&row_input;row_norm.data=normalized.data();
    std::memcpy(row_norm.op_params,&eps,sizeof(eps));
    ggml_tensor row_mul=row_input;row_mul.op=GGML_OP_MUL;row_mul.src[0]=&row_norm;row_mul.src[1]=const_cast<ggml_tensor *>(weight);row_mul.data=product.data();
    ggml_tensor row_add=row_input;row_add.op=GGML_OP_ADD;row_add.src[0]=&row_mul;row_add.src[1]=const_cast<ggml_tensor *>(bias);
    ggml_compute_params params{};params.nth=1;
    for(int64_t row=ith;row<ggml_nrows(input);row+=nth) {
        const int64_t i1=row%input->ne[1],i2=(row/input->ne[1])%input->ne[2],i3=row/(input->ne[1]*input->ne[2]);
        row_input.data=static_cast<char *>(input->data)+i1*input->nb[1]+i2*input->nb[2]+i3*input->nb[3];
        auto * output=static_cast<float *>(dst->data)+row*width;row_add.data=output;
        ggml_compute_forward_norm(&params,&row_norm);ggml_compute_forward_mul(&params,&row_mul);ggml_compute_forward_add(&params,&row_add);
    }
}

void layer_norm_affine_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void * userdata) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0];std::vector<float> values(width);
    ggml_tensor row_input=*input,normalized=*input,encoded=*dst;
    for(int dim=1;dim<4;++dim){row_input.ne[dim]=normalized.ne[dim]=encoded.ne[dim]=1;row_input.nb[dim]=normalized.nb[dim]=width*sizeof(float);encoded.nb[dim]=width+8;}
    normalized.data=values.data();normalized.src[0]=&row_input;normalized.src[1]=dst->src[1];normalized.src[2]=dst->src[2];encoded.src[0]=&normalized;
    for(int64_t row=ith;row<input->ne[1];row+=nth) {
        row_input.data=static_cast<char *>(input->data)+row*input->nb[1];encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];
        layer_norm_affine_f32_cpu(&normalized,0,1,userdata);i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

template <bool value_layout>
void bias_attention_layout_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * input = dst->src[0];
    const auto * bias = dst->src[1];
    const int64_t channels = input->ne[0], tokens = input->ne[1], heads = dst->ne[2];
    const int64_t width = channels / heads;
    auto * output = static_cast<float *>(dst->data);
    std::vector<float> added(channels);
    ggml_tensor row_input = *input, row_output = *input;
    for (int dim = 1; dim < 4; ++dim) {
        row_input.ne[dim] = row_output.ne[dim] = 1;
        row_input.nb[dim] = row_output.nb[dim] = channels * sizeof(float);
    }
    row_output.op = GGML_OP_ADD;
    row_output.src[0] = &row_input;
    row_output.src[1] = const_cast<ggml_tensor *>(bias);
    row_output.data = added.data();
    ggml_compute_params params{};
    params.nth = 1;
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // Group neighboring V writes within each worker's token block.
    constexpr int64_t token_block = 16;
    for (int64_t first=int64_t(ith)*token_block; first<tokens; first+=int64_t(nth)*token_block) {
        const int64_t last=std::min(tokens,first+token_block);
        for (int64_t token=first; token<last; ++token) {
            row_input.data=static_cast<char *>(input->data)+token*input->nb[1];
            ggml_compute_forward_add(&params,&row_output);
            for (int64_t head=0; head<heads; ++head) {
                const auto * source=added.data()+head*width;
                if constexpr (!value_layout) {
                    std::memcpy(output+(head*tokens+token)*width,source,size_t(width)*sizeof(float));
                } else {
                    auto * target=output+head*width*tokens+token;
                    for (int64_t col=0; col<width; ++col)target[col*tokens]=source[col];
                }
            }
        }
    }
    return;
#endif
    for (int64_t token = ith; token < tokens; token += nth) {
        row_input.data = static_cast<char *>(input->data) + token * input->nb[1];
        // Invoke the same pinned CPU ADD kernel, including its NaN operand order.
        ggml_compute_forward_add(&params, &row_output);
        for (int64_t channel = 0; channel < channels; ++channel) {
            const int64_t head = channel / width, col = channel % width;
            const int64_t destination = value_layout
                ? (head * width + col) * tokens + token : (head * tokens + token) * width + col;
            output[destination] = added[channel];
        }
    }
}

void contiguous_layout_f32_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * source = dst->src[0];
    const auto * input = static_cast<const char *>(source->data);
    auto * output = static_cast<float *>(dst->data);
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    if (nth == 1 && ith == 0 && source->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        source->data && dst->data && source->ne[0] == 128 && source->ne[1] == 20 &&
        source->ne[2] == 1 && source->ne[3] == 1 && source->nb[0] == sizeof(float) &&
        ggml_are_same_shape(source, dst) && ggml_is_contiguous(dst)) {
        constexpr size_t row_bytes = 128 * sizeof(float), output_bytes = 20 * row_bytes;
        if (source->nb[1] >= row_bytes && source->nb[1] <= (std::numeric_limits<size_t>::max() - row_bytes) / 19) {
            const size_t input_span = 19 * source->nb[1] + row_bytes;
            const auto input_address = reinterpret_cast<uintptr_t>(source->data);
            const auto output_address = reinterpret_cast<uintptr_t>(dst->data);
            const bool disjoint = input_address <= output_address
                ? output_address - input_address >= input_span : input_address - output_address >= output_bytes;
            if (disjoint) {
                for (size_t head = 0; head < 20; ++head)
                    std::memcpy(output + head * 128, input + head * source->nb[1], row_bytes);
                return;
            }
        }
    }
#endif
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // C007: 72 is divisible by eight; keep exactly the original flat%8 owner.
    if(nth==8 && ith>=0 && ith<8 &&
       source->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 && source->data && dst->data &&
       source->ne[0]==72 && source->ne[1]==16 && source->ne[2]==256 && source->ne[3]==1 &&
       source->nb[0]==sizeof(float) && ggml_are_same_shape(source,dst) && ggml_is_contiguous(dst)) {
        constexpr size_t row_bytes=72*sizeof(float),output_bytes=row_bytes*16*256;
        const size_t maximum=std::numeric_limits<size_t>::max();
        if(source->nb[1]>=row_bytes && source->nb[2]>=row_bytes &&
           source->nb[1]<=(maximum-row_bytes)/15) {
            const size_t head_span=source->nb[1]*15+row_bytes;
            if(source->nb[2]<=(maximum-head_span)/255) {
                const size_t input_span=head_span+source->nb[2]*255;
                const uintptr_t a=reinterpret_cast<uintptr_t>(source->data),b=reinterpret_cast<uintptr_t>(dst->data);
                const bool bounded=a<=std::numeric_limits<uintptr_t>::max()-input_span && b<=std::numeric_limits<uintptr_t>::max()-output_bytes;
                const bool disjoint=b<=a ? a-b>=output_bytes : b-a>=input_span;
                if(bounded && disjoint) {
                    for(size_t token=0;token<256;++token)for(size_t head=0;head<16;++head) {
                        const auto * row=input+token*source->nb[2]+head*source->nb[1];
                        auto * target=output+(token*16+head)*72;
                        for(int col=ith;col<72;col+=8) {
                            uint32_t bits;std::memcpy(&bits,row+size_t(col)*sizeof(float),sizeof(bits));
                            std::memcpy(target+col,&bits,sizeof(bits));
                        }
                    }
                    return;
                }
            }
        }
    }
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    if (ith >= 0 && nth > 0 && source->ne[0] > 0 && source->ne[1] > 0
        && source->ne[2] > 0 && source->ne[3] > 0) {
        if (ith >= ggml_nelements(dst)) return;
        uint64_t coordinates[4], increments[4];
        uint64_t position = uint64_t(ith), step = uint64_t(nth);
        for (int dim = 0; dim < 4; ++dim) {
            const auto extent = uint64_t(source->ne[dim]);
            coordinates[dim] = position % extent;
            position /= extent;
            increments[dim] = step % extent;
            step /= extent;
        }
        for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
            const size_t offset = coordinates[0] * source->nb[0]
                + coordinates[1] * source->nb[1]
                + coordinates[2] * source->nb[2]
                + coordinates[3] * source->nb[3];
            // CPU106 as retained in165: preserve all F32 bits without BF16 rounding.
            uint32_t bits;std::memcpy(&bits,input+offset,sizeof(bits));
            std::memcpy(output+i,&bits,sizeof(bits));
            // Add the original cyclic step in mixed radix, without division.
            // Each digit plus increment plus carry fits uint64_t for int64 extents.
            uint64_t carry = 0;
            for (int dim = 0; dim < 4; ++dim) {
                const auto extent = uint64_t(source->ne[dim]);
                const auto digit = coordinates[dim] + increments[dim] + carry;
                carry = digit >= extent;
                coordinates[dim] = carry ? digit - extent : digit;
            }
        }
        return;
    }
#endif
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
        int64_t position = i;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            offset += (position % source->ne[dim]) * source->nb[dim];
            position /= source->ne[dim];
        }
        output[i] = *reinterpret_cast<const float *>(input + offset);
    }
}

void contiguous_layout_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void *) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0]*input->ne[1];
    std::vector<float> temporary(width);
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+(row%input->ne[2])*input->nb[2]+(row/input->ne[2])*input->nb[3];view.ne[2]=view.ne[3]=1;
        ggml_tensor contiguous=view;contiguous.data=temporary.data();contiguous.src[0]=&view;contiguous.nb[0]=4;contiguous.nb[1]=4*contiguous.ne[0];contiguous.nb[2]=contiguous.nb[3]=4*width;
        contiguous_layout_f32_cpu(&contiguous,0,1,nullptr);
        ggml_tensor flat=contiguous;flat.ne[0]=width;flat.ne[1]=flat.ne[2]=flat.ne[3]=1;flat.nb[0]=4;flat.nb[1]=flat.nb[2]=flat.nb[3]=4*width;
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&flat;
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

void permuted_rms_affine_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void * userdata) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0]*input->ne[1];
#if defined(BITVLA_USE_CPU_SME) && defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // CUDA154 composition: gather two rows with the original layout copy,
    // then use the accepted paired RMS and unchanged encoder.
    std::vector<float> packed_values(2*width),normalized_values(2*width);
    for(int64_t row=int64_t(ith)*2;row<dst->ne[1];row+=int64_t(nth)*2) {
        const int64_t count=std::min<int64_t>(2,dst->ne[1]-row);
        for(int64_t j=0;j<count;++j) {
            ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+(row+j)*input->nb[2];view.ne[2]=view.ne[3]=1;
            ggml_tensor contiguous=view;contiguous.data=packed_values.data()+j*width;contiguous.src[0]=&view;
            contiguous.nb[0]=4;contiguous.nb[1]=4*contiguous.ne[0];contiguous.nb[2]=contiguous.nb[3]=4*width;
            contiguous_layout_f32_cpu(&contiguous,0,1,nullptr);
        }
        ggml_tensor flat=*input;flat.data=packed_values.data();flat.ne[0]=width;flat.ne[1]=count;flat.ne[2]=flat.ne[3]=1;
        flat.nb[0]=4;flat.nb[1]=4*width;flat.nb[2]=flat.nb[3]=4*width*count;
        ggml_tensor normalized=flat;normalized.data=normalized_values.data();normalized.src[0]=&flat;normalized.src[1]=dst->src[1];normalized.src[2]=nullptr;
        rms_affine_f32_cpu(&normalized,0,1,userdata);
        ggml_tensor encoded=*dst;encoded.ne[1]=count;encoded.ne[2]=encoded.ne[3]=1;encoded.nb[2]=encoded.nb[3]=encoded.nb[1]*count;
        encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&normalized;encoded.src[1]=nullptr;
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
#else
    std::vector<float> temporary(width);
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+row*input->nb[2];view.ne[2]=view.ne[3]=1;
        ggml_tensor contiguous=view;contiguous.data=temporary.data();contiguous.src[0]=&view;contiguous.nb[0]=4;contiguous.nb[1]=4*contiguous.ne[0];contiguous.nb[2]=contiguous.nb[3]=4*width;
        contiguous_layout_f32_cpu(&contiguous,0,1,nullptr);
        ggml_tensor flat=contiguous;flat.ne[0]=width;flat.ne[1]=flat.ne[2]=flat.ne[3]=1;flat.nb[0]=4;flat.nb[1]=flat.nb[2]=flat.nb[3]=4*width;
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&flat;encoded.src[1]=dst->src[1];
        rms_affine_i2s_encode_cpu(&encoded,0,1,userdata);
    }
#endif
}

void f32_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void *) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0];
    std::vector<float> temporary(width);
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+row*input->nb[1];view.ne[1]=view.ne[2]=view.ne[3]=1;
        ggml_tensor contiguous=view;contiguous.data=temporary.data();contiguous.src[0]=&view;
        contiguous_layout_f32_cpu(&contiguous,0,1,nullptr);
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&contiguous;
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

void gelu_f32_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void *) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0];
    std::vector<float> activated_values(width);
    ggml_compute_params params{};params.nth=1;
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+row*input->nb[1];view.ne[1]=view.ne[2]=view.ne[3]=1;
        ggml_tensor activated=view;activated.op=GGML_OP_UNARY;activated.src[0]=&view;activated.src[1]=activated.src[2]=nullptr;activated.data=activated_values.data();
        const int32_t unary=GGML_UNARY_OP_GELU;std::memcpy(activated.op_params,&unary,sizeof(unary));
        ggml_compute_forward_unary(&params,&activated);
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&activated;
        f32_i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

void bias_gelu_f32_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void *) {
    const auto * input=dst->src[0];const auto * bias=dst->src[1];const int64_t width=input->ne[0];
    std::vector<float> added_values(width);ggml_compute_params params{};params.nth=1;
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor view=*input;view.data=static_cast<char *>(input->data)+row*input->nb[1];view.ne[1]=view.ne[2]=view.ne[3]=1;
        ggml_tensor added=view;added.op=GGML_OP_ADD;added.src[0]=&view;added.src[1]=const_cast<ggml_tensor *>(bias);added.src[2]=nullptr;added.data=added_values.data();
        ggml_compute_forward_add(&params,&added);
        for(auto & value:added_values)value=value;
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&added;encoded.src[1]=nullptr;
        gelu_f32_i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

template <bool packed_attention>
void rope_f32_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * input = static_cast<const float *>(dst->src[0]->data);
    const auto * cosine = static_cast<const float *>(dst->src[1]->data);
    const auto * sine = static_cast<const float *>(dst->src[2]->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t width = dst->src[0]->ne[0], half = width / 2, heads = dst->src[0]->ne[1], tokens = dst->src[0]->ne[2];
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    if constexpr (packed_attention) {
        const auto output_address = reinterpret_cast<uintptr_t>(output);
        const size_t output_bytes = ggml_nbytes(dst);
        const auto disjoint = [&](const ggml_tensor * tensor) {
            const auto address = reinterpret_cast<uintptr_t>(tensor->data);
            return address <= output_address ? output_address - address >= ggml_nbytes(tensor)
                                             : address - output_address >= output_bytes;
        };
        if (width == 128 && (heads == 5 || heads == 20) && tokens > 0 && nth > 0 &&
            dst->src[0]->ne[3] == 1 && dst->ne[0] == width && dst->ne[1] == tokens &&
            dst->ne[2] == heads && dst->ne[3] == 1 && ggml_is_contiguous(dst) &&
            ggml_is_contiguous(dst->src[0]) && ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst->src[2]) &&
            disjoint(dst->src[0]) && disjoint(dst->src[1]) && disjoint(dst->src[2])) {
            for (int64_t row = ith; row < heads * tokens; row += nth) {
                const int64_t token = row / heads, head = row % heads;
                const auto * values = input + row * 128;
                const auto * cosines = cosine + token * 128;
                const auto * sines = sine + token * 128;
                auto * result = output + (head * tokens + token) * 128;
                for (int64_t column = 0; column < 128; ++column) {
                    const float rotated = column < 64 ? -values[column + 64] : values[column - 64];
                    const float left = std::isnan(values[column]) ? values[column] : values[column] * cosines[column];
                    const float right = std::isnan(rotated) ? rotated : rotated * sines[column];
                    result[column] = std::isnan(left) ? left : left + right;
                }
            }
            return;
        }
    }
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    if constexpr (packed_attention) {
        if (width == 128 && heads == 5) {
            // CPU151 only: specialize K indexes; keep the current F32 operations.
            const int64_t width = 128, half = width / 2, heads = 5;
            for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
                const int64_t col = i % width, token = i / (width * heads);
                const float rotated = col < half ? -input[i + half] : input[i - half];
                // Preserve the original CPU ggml binary kernels' first-NaN operand.
                const float left = std::isnan(input[i]) ? input[i] : input[i] * cosine[token * width + col];
                const float right = std::isnan(rotated) ? rotated : rotated * sine[token * width + col];
                const int64_t head = (i / width) % heads;
                const int64_t destination = packed_attention ? (head * tokens + token) * width + col : i;
                output[destination] = std::isnan(left) ? left : left + right;
            }
            return;
        }
    }
#endif
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
        const int64_t col = i % width, token = i / (width * heads);
        const float rotated = col < half ? -input[i + half] : input[i - half];
        // Preserve the original CPU ggml binary kernels' first-NaN operand.
        const float left = std::isnan(input[i]) ? input[i] : input[i] * cosine[token * width + col];
        const float right = std::isnan(rotated) ? rotated : rotated * sine[token * width + col];
        const int64_t head = (i / width) % heads;
        const int64_t destination = packed_attention ? (head * tokens + token) * width + col : i;
        output[destination] = std::isnan(left) ? left : left + right;
    }
}

void gated_relu_square_f32_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * gate = static_cast<const float *>(dst->src[0]->data);
    const auto * up = static_cast<const float *>(dst->src[1]->data);
    auto * output = static_cast<float *>(dst->data);
#if defined(__APPLE__) && defined(__aarch64__)
    const int64_t count = ggml_nelements(dst);
    const int64_t chunk = count / nth, remainder = count % nth;
    const int64_t first = chunk * ith + remainder * ith / nth;
    const int64_t last = chunk * (ith + 1) + remainder * (ith + 1) / nth;
    const auto zero = vdupq_n_f32(0.0f);
    int64_t i = first;
    for (; i + 4 <= last; i += 4) {
        const auto positive = vmaxnmq_f32(vld1q_f32(gate + i), zero);
        const auto squared = vmulq_f32(positive, positive);
        vst1q_f32(output + i, vmulq_f32(vld1q_f32(up + i), squared));
    }
    for (; i < last; ++i) {
        const float value = gate[i] > 0.0f ? gate[i] : 0.0f;
        const float squared = value * value;
        output[i] = squared * up[i];
    }
#else
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
        const float value = gate[i] > 0.0f ? gate[i] : 0.0f;
        const float squared = value * value;
        output[i] = squared * up[i];
    }

#endif
}

// Compose the existing gate and RMS callbacks; preserve each row's arithmetic.
void gated_rms_affine_f32_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const int64_t width=dst->ne[0];
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // Two adjacent rows keep the accepted paired-row RMS SIMD path available.
    // Scratch is private to this callback invocation, not cached across graphs.
    std::vector<float> scratch(2 * width);
    const int64_t rows = ggml_nrows(dst);
    for (int64_t row = int64_t(ith) * 2; row < rows; row += int64_t(nth) * 2) {
        const int64_t count = std::min<int64_t>(2, rows - row);
        ggml_tensor gate=*dst->src[0], up=*dst->src[1], activated=gate, output=*dst;
        for (int dim=1; dim<4; ++dim) {
            gate.ne[dim]=up.ne[dim]=activated.ne[dim]=output.ne[dim]=(dim==1 ? count : 1);
            gate.nb[dim]=up.nb[dim]=activated.nb[dim]=output.nb[dim]=
                size_t(width) * sizeof(float) * (dim==1 ? 1 : count);
        }
        gate.data=static_cast<float *>(dst->src[0]->data)+row*width;
        up.data=static_cast<float *>(dst->src[1]->data)+row*width;
        activated.data=scratch.data(); activated.src[0]=&gate; activated.src[1]=&up;
        gated_relu_square_f32_cpu(&activated,0,1,nullptr);
        output.data=static_cast<float *>(dst->data)+row*width;
        output.src[0]=&activated; output.src[1]=dst->src[2];
        rms_affine_f32_cpu(&output,0,1,userdata);
    }
#else
    std::vector<float> scratch(width);
    for(int64_t row=ith;row<ggml_nrows(dst);row+=nth) {
        ggml_tensor gate=*dst->src[0], up=*dst->src[1], activated=gate, output=gate;
        for(int dim=1;dim<4;++dim) gate.ne[dim]=up.ne[dim]=activated.ne[dim]=output.ne[dim]=1;
        gate.data=static_cast<float *>(dst->src[0]->data)+row*width;
        up.data=static_cast<float *>(dst->src[1]->data)+row*width;
        activated.data=scratch.data(); activated.src[0]=&gate;activated.src[1]=&up;
        gated_relu_square_f32_cpu(&activated,0,1,nullptr);
        output.data=static_cast<float *>(dst->data)+row*width;
        output.src[0]=&activated;output.src[1]=dst->src[2];
        rms_affine_f32_cpu(&output,0,1,userdata);
    }
#endif
}

void gated_rms_affine_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void * userdata) {
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    const auto * input=dst->src[0];
    const int64_t width=input->ne[0];
    std::vector<float> activated_values(2 * width), normalized_values(2 * width);
    ggml_tensor gate=*input, up=*dst->src[1], activated=gate, normalized=gate, encoded=*dst;
    activated.data=activated_values.data(); activated.src[0]=&gate; activated.src[1]=&up;
    normalized.data=normalized_values.data(); normalized.src[0]=&activated; normalized.src[1]=dst->src[2];
    encoded.src[0]=&normalized;
    for (int64_t row=int64_t(ith)*2; row<input->ne[1]; row+=int64_t(nth)*2) {
        const int64_t count=std::min<int64_t>(2,input->ne[1]-row);
        for (int dim=1; dim<4; ++dim) {
            gate.ne[dim]=up.ne[dim]=activated.ne[dim]=normalized.ne[dim]=encoded.ne[dim]=(dim==1 ? count : 1);
            gate.nb[dim]=up.nb[dim]=activated.nb[dim]=normalized.nb[dim]=
                size_t(width)*sizeof(float)*(dim==1 ? 1 : count);
            encoded.nb[dim]=size_t(width+8)*(dim==1 ? 1 : count);
        }
        gate.data=static_cast<char *>(input->data)+row*input->nb[1];
        up.data=static_cast<char *>(dst->src[1]->data)+row*dst->src[1]->nb[1];
        encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];
        // Reuse the accepted arithmetic and encoder; only their intermediate
        // storage and scheduling are fused. No persistent activation cache.
        gated_relu_square_f32_cpu(&activated,0,1,nullptr);
        rms_affine_f32_cpu(&normalized,0,1,userdata);
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
#else
    const auto * input=dst->src[0];const int64_t width=input->ne[0];std::vector<float> values(width);
    ggml_tensor gate=*input,up=*dst->src[1],normalized=*input,encoded=*dst;
    for(int dim=1;dim<4;++dim){gate.ne[dim]=up.ne[dim]=normalized.ne[dim]=encoded.ne[dim]=1;gate.nb[dim]=up.nb[dim]=normalized.nb[dim]=width*sizeof(float);encoded.nb[dim]=width+8;}
    normalized.data=values.data();normalized.src[0]=&gate;normalized.src[1]=&up;normalized.src[2]=dst->src[2];encoded.src[0]=&normalized;
    for(int64_t row=ith;row<input->ne[1];row+=nth) {
        gate.data=static_cast<char *>(input->data)+row*input->nb[1];up.data=static_cast<char *>(dst->src[1]->data)+row*dst->src[1]->nb[1];encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];
        gated_rms_affine_f32_cpu(&normalized,0,1,userdata);i2s_encode_cpu(&encoded,0,1,nullptr);
    }
#endif
}

void score_scale_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const float scale = *static_cast<const float *>(userdata);
    const auto * input = static_cast<const float *>(dst->src[0]->data);
    auto * output = static_cast<float *>(dst->data);
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth)
        output[i] = input[i] * scale;
}

#ifdef BITVLA_USE_CUDA
void rms_norm_cpu_order_fallback(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * input = static_cast<const float *>(dst->src[0]->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t width = dst->ne[0];
    const float eps = *static_cast<const float *>(userdata);
    for (int64_t row = ith; row < ggml_nrows(dst); row += nth) {
        const auto * x = input + row * width;
        double sum = 0.0;
        for (int64_t col = 0; col < width; ++col) sum += double(x[col] * x[col]);
        const float mean = sum / width;
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int64_t col = 0; col < width; ++col) output[row * width + col] = x[col] * scale;
    }
}
#endif

void cpu_order_rms_i2s_encode_cpu(ggml_tensor * dst,int ith,int nth,void * userdata) {
    const auto * input=dst->src[0];const int64_t width=input->ne[0];std::vector<float> normal_values(width),product_values(width);
    ggml_compute_params params{};params.nth=1;
    for(int64_t row=ith;row<dst->ne[1];row+=nth) {
        ggml_tensor row_input=*input;row_input.data=static_cast<char *>(input->data)+row*input->nb[1];row_input.ne[1]=row_input.ne[2]=row_input.ne[3]=1;
        ggml_tensor normalized=row_input;normalized.src[0]=&row_input;normalized.data=normal_values.data();normalized.op=GGML_OP_RMS_NORM;std::memcpy(normalized.op_params,userdata,sizeof(float));
#ifdef BITVLA_USE_CUDA
        rms_norm_cpu_order_fallback(&normalized,0,1,userdata);
#else
        ggml_compute_forward_rms_norm(&params,&normalized);
#endif
        ggml_tensor product=row_input;product.op=GGML_OP_MUL;product.src[0]=&normalized;product.src[1]=dst->src[1];product.data=product_values.data();
        ggml_compute_forward_mul(&params,&product);
        ggml_tensor encoded=*dst;encoded.ne[1]=encoded.ne[2]=encoded.ne[3]=1;encoded.data=static_cast<char *>(dst->data)+row*dst->nb[1];encoded.src[0]=&product;
        i2s_encode_cpu(&encoded,0,1,nullptr);
    }
}

void rope_cache_cpu(ggml_tensor * cosine, ggml_tensor * sine) {
    auto * c = static_cast<float *>(cosine->data);
    auto * s = static_cast<float *>(sine->data);
    const int width = static_cast<int>(cosine->ne[0]);
    for (int64_t i = 0; i < ggml_nelements(cosine); ++i) {
        const int dim = i % (width / 2), position = i / width;
        const float angle = position * (1.0f / std::pow(500000.0f, 2.0f * dim / width));
        c[i] = std::cos(angle);
        s[i] = std::sin(angle);
    }
}
} // namespace

ggml_tensor * score_scale(ggml_context * ctx, ggml_tensor * input, float scale) {
    auto * factor = static_cast<float *>(ggml_new_buffer(ctx, sizeof(float)));
    *factor = scale;
    ggml_tensor * sources[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        sources, 1, score_scale_cpu, GGML_N_TASKS_MAX, factor);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, score_scale_cuda);
#endif
    return result;
}

ggml_tensor * vision_rgb_u8_f32(ggml_context * ctx, ggml_tensor * rgb, ggml_tensor * lookup, int size) {
    GGML_ASSERT(size > 0 && rgb->type == GGML_TYPE_I8 && lookup->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(rgb) && ggml_is_contiguous(lookup));
    GGML_ASSERT(ggml_nelements(rgb) == int64_t(size)*size*3 && ggml_nelements(lookup) == 768);
    ggml_tensor * sources[] = {rgb, lookup};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, size, size, 3, 1,
                                  sources, 2, vision_rgb_u8_f32_cpu, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, vision_rgb_u8_f32_cuda);
#endif
    return result;
}

ggml_tensor * gated_relu_square_f32(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up) {
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_is_contiguous(gate) && ggml_is_contiguous(up));
    ggml_tensor * sources[] = {gate, up};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        gate->ne[0], gate->ne[1], gate->ne[2], gate->ne[3],
        sources, 2, gated_relu_square_f32_cpu, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, gated_relu_square_f32_cuda);
#endif
    return result;
}

ggml_tensor * permuted_rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps) {
    // Only the original contiguous GEMM output with its permuted head/row view.
    if(input->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 || !ggml_is_contiguous(weight) ||
       input->ne[0]<=0 || input->ne[0]>2560 || 2560%input->ne[0]!=0 || input->ne[1]!=2560/input->ne[0] ||
       input->ne[2]<=0 || input->ne[2]>INT32_MAX/2568 || input->ne[3]!=1 ||
       input->nb[0]!=4 || input->nb[2]!=uint64_t(input->ne[0])*4 || input->nb[1]!=uint64_t(input->ne[0])*uint64_t(input->ne[2])*4 ||
       weight->ne[0]!=2560 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)return nullptr;
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;ggml_tensor * sources[]={input,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,2568,input->ne[2],1,1,sources,2,permuted_rms_affine_i2s_encode_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,permuted_rms_affine_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * cpu_order_rms_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps) {
    if(input->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 || !ggml_is_contiguous(input) || !ggml_is_contiguous(weight) ||
       input->ne[0]!=2560 || input->ne[1]<=0 || input->ne[1]>INT32_MAX/2568 || input->ne[2]!=1 || input->ne[3]!=1 ||
       weight->ne[0]!=2560 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)return nullptr;
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;ggml_tensor * sources[]={input,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,2568,input->ne[1],1,1,sources,2,cpu_order_rms_i2s_encode_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,cpu_order_rms_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,float eps) {
    // Optional producer: a null result leaves the caller on its original graph.
    if(input->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 || !ggml_is_contiguous(input) || !ggml_is_contiguous(weight) ||
       input->ne[0]!=2560 || input->ne[1]<=0 || input->ne[1]>INT32_MAX/2568 || input->ne[2]!=1 || input->ne[3]!=1 ||
       weight->ne[0]!=2560 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)return nullptr;
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;ggml_tensor * sources[]={input,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,2568,input->ne[1],1,1,sources,2,rms_affine_i2s_encode_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,rms_affine_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * gated_rms_affine_i2s_encode(ggml_context * ctx,ggml_tensor * gate,ggml_tensor * up,ggml_tensor * weight,float eps) {
    if(gate->type!=GGML_TYPE_F32 || up->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 ||
       !ggml_are_same_shape(gate,up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || !ggml_is_contiguous(weight) ||
       gate->ne[0]!=6912 || gate->ne[1]<=0 || gate->ne[1]>INT32_MAX/6920 || gate->ne[2]!=1 || gate->ne[3]!=1 ||
       weight->ne[0]!=6912 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)return nullptr;
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;ggml_tensor * sources[]={gate,up,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,6920,gate->ne[1],1,1,sources,3,gated_rms_affine_i2s_encode_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,gated_rms_affine_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * gated_rms_affine_f32(ggml_context * ctx, ggml_tensor * gate,
        ggml_tensor * up, ggml_tensor * weight, float eps) {
    // The original two-operation graph remains the general fallback.
    if(gate->type!=GGML_TYPE_F32 || up->type!=GGML_TYPE_F32 ||
       !ggml_are_same_shape(gate,up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) ||
       gate->ne[0]!=6912 || weight->type!=GGML_TYPE_F32 || !ggml_is_contiguous(weight) ||
       weight->ne[0]!=6912 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1)
        return rms_norm(ctx,gated_relu_square_f32(ctx,gate,up),weight,eps);
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;
    ggml_tensor * sources[]={gate,up,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,gate->ne[0],gate->ne[1],gate->ne[2],gate->ne[3],
        sources,3,gated_rms_affine_f32_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,gated_rms_affine_f32_cuda);
#endif
    return result;
}

ggml_tensor * contiguous_layout_f32(ggml_context * ctx, ggml_tensor * input) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    ggml_tensor * sources[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        sources, 1, contiguous_layout_f32_cpu, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, contiguous_layout_f32_cuda);
#endif
    return result;
}

ggml_tensor * contiguous_layout_i2s_encode(ggml_context * ctx,ggml_tensor * input) {
    GGML_ASSERT(input->type==GGML_TYPE_F32 && input->ne[0]>0 && input->ne[1]>0 && input->ne[2]>0 && input->ne[3]>0 && input->ne[0]<=2147483647 && input->ne[1]<=2147483647 && input->ne[2]<=2147483647 && input->ne[3]<=2147483647);
    const int64_t width=input->ne[0]*input->ne[1],rows=input->ne[2]*input->ne[3];
    const bool fused=input->ne[0]<=1152 && input->ne[1]<=1152 && width==1152 && input->ne[3]==1 && rows<=2147483647/1160 &&
        input->nb[0]==4 && input->nb[2]==size_t(input->ne[0])*4 && input->nb[1]==input->nb[2]*size_t(input->ne[2]);
    if(!fused) {
        auto * contiguous=contiguous_layout_f32(ctx,input);
        return i2s_encode(ctx,ggml_reshape_2d(ctx,contiguous,width,rows),(width+127)/128*128);
    }
    ggml_tensor * sources[]={input};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,1160,rows,1,1,sources,1,contiguous_layout_i2s_encode_cpu,GGML_N_TASKS_MAX,nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,contiguous_layout_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * bias_gelu_f32_i2s_encode(ggml_context * ctx,ggml_tensor * added) {
    if(!added || added->type!=GGML_TYPE_F32 || added->op!=GGML_OP_ADD ||
       (added->flags&GGML_TENSOR_FLAG_OUTPUT) || !added->src[0] || !added->src[1])return nullptr;
    auto * input=added->src[0];auto * bias=added->src[1];
    if(input->type!=GGML_TYPE_F32 || bias->type!=GGML_TYPE_F32 ||
       !ggml_is_contiguous(input) || !ggml_is_contiguous(bias) || !ggml_is_contiguous(added) ||
       !ggml_are_same_shape(input,added) || input->ne[0]!=4304 || input->ne[1]<=0 ||
       input->ne[1]>2147483647/4360 || input->ne[2]!=1 || input->ne[3]!=1 ||
       bias->ne[0]!=4304 || bias->ne[1]!=1 || bias->ne[2]!=1 || bias->ne[3]!=1)return nullptr;
    ggml_tensor * sources[]={input,bias};auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,4360,input->ne[1],1,1,sources,2,bias_gelu_f32_i2s_encode_cpu,GGML_N_TASKS_MAX,nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,bias_gelu_f32_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * gelu_f32_i2s_encode(ggml_context * ctx,ggml_tensor * input) {
    if(!input || input->type!=GGML_TYPE_F32 || input->ne[0]!=4304 || input->ne[1]<=0 || input->ne[1]>2147483647/4360 || input->ne[2]!=1 || input->ne[3]!=1 || !ggml_is_contiguous(input))return nullptr;
    ggml_tensor * sources[]={input};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,4360,input->ne[1],1,1,sources,1,gelu_f32_i2s_encode_cpu,GGML_N_TASKS_MAX,nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,gelu_f32_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * f32_i2s_encode(ggml_context * ctx,ggml_tensor * input) {
    // Optional F32-to-I2S encoder; unsupported layouts retain the original graph.
    if(input->type!=GGML_TYPE_F32 || !ggml_is_contiguous(input) || input->ne[0]!=4304 ||
       input->ne[1]<=0 || input->ne[1]>2147483647/4360 || input->ne[2]!=1 || input->ne[3]!=1)return nullptr;
    ggml_tensor * sources[]={input};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,4360,input->ne[1],1,1,sources,1,f32_i2s_encode_cpu,GGML_N_TASKS_MAX,nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,f32_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * activation_quant(ggml_context * ctx, ggml_tensor * tensor) {
    const int64_t width = tensor->ne[0], rows = tensor->ne[1];
    auto * absolute = ggml_abs(ctx, tensor);
    auto * indices = ggml_reshape_2d(ctx, ggml_argmax(ctx, absolute), 1, rows);
    auto * maximum = ggml_get_rows(ctx, ggml_reshape_3d(ctx, absolute, 1, width, rows), indices);
    maximum = ggml_clamp(ctx, ggml_reshape_2d(ctx, maximum, 1, rows), 1e-5f, std::numeric_limits<float>::max());
    // PyTorch's `127 / tensor` evaluates reciprocal first, then multiplies by 127.
    auto * scale = ggml_div(ctx, ggml_scale_bias(ctx, maximum, 0.0f, 1.0f), maximum);
    scale = ggml_scale(ctx, scale, 127.0f);
    auto * scaled = ggml_mul(ctx, tensor, scale);
    // In [-127, 127], these two F32 writes round to nearest integer, ties to even.
    auto * rounded = ggml_scale_bias(ctx, ggml_scale_bias(ctx, scaled, 1.0f, 12582912.0f), 1.0f, -12582912.0f);
    return ggml_div(ctx, ggml_clamp(ctx, rounded, -128.0f, 127.0f), scale);
}

ggml_tensor * linear_input(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * tensor) {
    return weight->type == GGML_TYPE_I8 || weight->type == GGML_TYPE_I2_S
        ? tensor : activation_quant(ctx, tensor);
}

ggml_tensor * shared_linear_input(ggml_context * ctx, ggml_tensor * input,
                                   std::initializer_list<ggml_tensor *> weights) {
    if (weights.size() < 2 || input->type != GGML_TYPE_F32 || !ggml_is_contiguous(input) ||
        input->ne[2] != 1 || input->ne[3] != 1) return nullptr;
    for (const auto * weight : weights) {
        if (!weight || weight->type != GGML_TYPE_I2_S || weight->ne[0] != input->ne[0]) return nullptr;
    }
    return i2s_encode(ctx, input, (input->ne[0] + 127) / 128 * 128);
}

void initialize_rope_cache(ggml_tensor * cosine, ggml_tensor * sine) {
    if (ggml_backend_buffer_is_host(cosine->buffer)) {
        rope_cache_cpu(cosine, sine);
    } else {
#ifdef BITVLA_USE_CUDA
        if (!rope_cache_cuda(cosine, sine)) GGML_ABORT("BitVLA CUDA RoPE cache failed");
#else
        GGML_ABORT("BitVLA RoPE cache requires CPU or CUDA");
#endif
    }
}

static ggml_tensor * make_rope(ggml_context * ctx, ggml_tensor * tensor, ggml_tensor * cosine, ggml_tensor * sine, bool packed_attention) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && cosine->type == GGML_TYPE_F32 && sine->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(tensor) && ggml_is_contiguous(cosine) && ggml_is_contiguous(sine));
    GGML_ASSERT(tensor->ne[0] % 2 == 0 && tensor->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(cosine, sine) && cosine->ne[0] == tensor->ne[0]);
    GGML_ASSERT(cosine->ne[1] == 1 && cosine->ne[2] == tensor->ne[2] && cosine->ne[3] == 1);
    ggml_tensor * sources[] = {tensor, cosine, sine};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        tensor->ne[0], packed_attention ? tensor->ne[2] : tensor->ne[1],
        packed_attention ? tensor->ne[1] : tensor->ne[2], 1,
        sources, 3, packed_attention ? rope_f32_cpu<true> : rope_f32_cpu<false>, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, packed_attention ? rope_attention_layout_cuda : rope_f32_cuda);
#endif
    return result;
}

ggml_tensor * rope(ggml_context * ctx, ggml_tensor * input, ggml_tensor * cosine, ggml_tensor * sine) {
    return make_rope(ctx, input, cosine, sine, false);
}

ggml_tensor * rope_attention_layout(ggml_context * ctx, ggml_tensor * input, ggml_tensor * cosine, ggml_tensor * sine) {
    return make_rope(ctx, input, cosine, sine, true);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input, ggml_tensor * bias) {
    if (weight->type == GGML_TYPE_I8 || weight->type == GGML_TYPE_I2_S) {
        // With no bias, use the optimized I2S route and store the F32 result directly.
        if (!bias) return i2s_linear(ctx, weight, input, true);
        auto * result = i2s_linear(ctx, weight, input);
        if (bias) result = ggml_add(ctx, result, bias);
        return result;
    }
    auto * result = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    // CUDA's fused vector matmul + bias discards the matmul's F32 precision.
    if (bias && input->ne[1] == 1) ggml_set_output(result);
    if (bias) result = ggml_add(ctx, result, bias);
    return result;
}

ggml_tensor * linear_shared(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                             ggml_tensor * encoded, ggml_tensor * bias) {
    if (!encoded) return linear(ctx, weight, input, bias);
    GGML_ASSERT(weight->type == GGML_TYPE_I2_S);
    if (!bias) return i2s_linear_encoded(ctx, weight, input, encoded, true);
    auto * result = i2s_linear_encoded(ctx, weight, input, encoded);
    return ggml_add(ctx, result, bias);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight,
                          ggml_tensor * bias, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, input, eps), weight), bias);
}


ggml_tensor * layer_norm_affine_i2s_encode(ggml_context * ctx,ggml_tensor * input,ggml_tensor * weight,ggml_tensor * bias,float eps) {
    if(input->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 || bias->type!=GGML_TYPE_F32 ||
       !ggml_is_contiguous(input) || !ggml_is_contiguous(weight) || !ggml_is_contiguous(bias) ||
       input->ne[0]!=1152 || input->ne[1]<=0 || input->ne[1]>INT32_MAX/1160 || input->ne[2]!=1 || input->ne[3]!=1 || eps<0.0f ||
       weight->ne[0]!=1152 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1 ||
       bias->ne[0]!=1152 || bias->ne[1]!=1 || bias->ne[2]!=1 || bias->ne[3]!=1)return nullptr;
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;ggml_tensor * sources[]={input,weight,bias};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_I8,1160,input->ne[1],1,1,sources,3,layer_norm_affine_i2s_encode_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,layer_norm_affine_i2s_encode_cuda);
#endif
    return result;
}

ggml_tensor * layer_norm_affine_f32(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias, float eps) {
#ifndef BITVLA_USE_CUDA
    return layer_norm(ctx,input,weight,bias,eps);
#else
    if(input->type!=GGML_TYPE_F32 || weight->type!=GGML_TYPE_F32 || bias->type!=GGML_TYPE_F32 ||
       !ggml_is_contiguous(input) || !ggml_is_contiguous(weight) || !ggml_is_contiguous(bias) ||
       input->ne[0]<=0 || input->ne[0]>std::numeric_limits<int>::max() ||
       weight->ne[0]!=input->ne[0] || bias->ne[0]!=input->ne[0] ||
       weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1 ||
       bias->ne[1]!=1 || bias->ne[2]!=1 || bias->ne[3]!=1 || eps<0.0f)
        return layer_norm(ctx,input,weight,bias,eps);
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;
    ggml_tensor * sources[]={input,weight,bias};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1],input->ne[2],input->ne[3],sources,3,layer_norm_affine_f32_cpu,GGML_N_TASKS_MAX,epsilon);
    ggml_custom_set_cuda(result,layer_norm_affine_f32_cuda);return result;
#endif
}

ggml_tensor * softmax_f32(ggml_context * ctx, ggml_tensor * input) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    // Preserve the original general path outside the shared-memory plain softmax.
    if (!ggml_is_contiguous(input) || input->ne[0] > 4096) return ggml_soft_max(ctx, input);
    ggml_tensor * sources[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        sources, 1, softmax_f32_cpu, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, softmax_f32_cuda);
#endif
    return result;
}


ggml_tensor * softmax_score_f32(ggml_context * ctx, ggml_tensor * input, float scale) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    if (!ggml_is_contiguous(input) || input->ne[0] > 4096)
        return softmax_f32(ctx, score_scale(ctx, input, scale));
    auto * factor=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float))); *factor=scale;
    ggml_tensor * sources[] = {input};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1],input->ne[2],input->ne[3],sources,1,softmax_score_f32_cpu,GGML_N_TASKS_MAX,factor);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,softmax_score_f32_cuda);
#endif
    return result;
}

ggml_tensor * rms_norm_f32(ggml_context * ctx, ggml_tensor * input, float eps) {
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->nb[0] == sizeof(float) && eps >= 0.0f);
    auto * epsilon = static_cast<float *>(ggml_new_buffer(ctx, sizeof(float)));
    *epsilon = eps;
    ggml_tensor * sources[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        sources, 1, rms_norm_f32_cpu, GGML_N_TASKS_MAX, epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, rms_norm_f32_cuda);
#endif
    return result;
}


ggml_tensor * rms_affine_f32(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, float eps) {
    GGML_ASSERT(input->type==GGML_TYPE_F32 && input->nb[0]==sizeof(float) && eps>=0.0f);
    GGML_ASSERT(weight->type==GGML_TYPE_F32 && ggml_is_contiguous(weight) && weight->ne[0]==input->ne[0] && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1);
    auto * epsilon=static_cast<float *>(ggml_new_buffer(ctx,sizeof(float)));*epsilon=eps;
    ggml_tensor * sources[]={input,weight};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,input->ne[0],input->ne[1],input->ne[2],input->ne[3],sources,2,rms_affine_f32_cpu,GGML_N_TASKS_MAX,epsilon);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result,rms_affine_f32_cuda);
#endif
    return result;
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, float eps) {
    if (weight->type==GGML_TYPE_F32 && ggml_is_contiguous(weight) && weight->ne[0]==input->ne[0] && weight->ne[1]==1 && weight->ne[2]==1 && weight->ne[3]==1)
        return rms_affine_f32(ctx,input,weight,eps);
    return ggml_mul(ctx, rms_norm_f32(ctx, input, eps), weight);
}

ggml_tensor * rms_norm_cpu_order(ggml_context * ctx, ggml_tensor * input, float eps) {
#ifdef BITVLA_USE_CUDA
    auto * epsilon = static_cast<float *>(ggml_new_buffer(ctx, sizeof(float)));
    *epsilon = eps;
    ggml_tensor * sources[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        sources, 1, rms_norm_cpu_order_fallback, GGML_N_TASKS_MAX, epsilon);
    ggml_custom_set_cuda(result, rms_norm_cpu_order_cuda);
    return result;
#else
    return ggml_rms_norm(ctx, input, eps);
#endif
}

void bias_attention_layout_cpu_existing(ggml_tensor * dst,int ith,int nth,bool value_layout) {
    if(value_layout)bias_attention_layout_cpu<true>(dst,ith,nth,nullptr);
    else bias_attention_layout_cpu<false>(dst,ith,nth,nullptr);
}

ggml_tensor * bias_attention_layout(ggml_context * ctx, ggml_tensor * input,
                                    ggml_tensor * bias, int heads, bool value_layout) {
    GGML_ASSERT(heads > 0 && input->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(bias));
    GGML_ASSERT(input->ne[2] == 1 && input->ne[3] == 1 && input->ne[0] % heads == 0);
    GGML_ASSERT(bias->ne[0] == input->ne[0] && bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1);
    const int64_t width = input->ne[0] / heads, tokens = input->ne[1];
    ggml_tensor * sources[] = {input, bias};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        value_layout ? tokens : width, value_layout ? width : tokens, heads, 1,
        sources, 2, value_layout ? bias_attention_layout_cpu<true> : bias_attention_layout_cpu<false>, GGML_N_TASKS_MAX, nullptr);
#ifdef BITVLA_USE_CUDA
    ggml_custom_set_cuda(result, value_layout ? bias_value_layout_cuda : bias_query_key_layout_cuda);
#endif
    return result;
}

static ggml_tensor * try_bias_attention_layout(ggml_context * ctx, ggml_tensor * restored, int heads, bool value_layout) {
#if defined(__APPLE__) && defined(__aarch64__) && !defined(BITVLA_USE_CUDA) && !defined(GGML_USE_METAL)
    // The accepted CPU path has F32 ADD directly, with no BF16 boundary.
    if (restored->op==GGML_OP_ADD && restored->type==GGML_TYPE_F32 &&
        !(restored->flags&GGML_TENSOR_FLAG_OUTPUT) && restored->src[0] && restored->src[1]) {
        auto * input=restored->src[0];auto * bias=restored->src[1];
        if (heads>0 && input->type==GGML_TYPE_F32 && bias->type==GGML_TYPE_F32 &&
            ggml_is_contiguous(input) && ggml_is_contiguous(bias) && ggml_are_same_shape(input,restored) &&
            input->ne[0]>0 && input->ne[1]>0 && input->ne[2]==1 && input->ne[3]==1 && input->ne[0]%heads==0 &&
            bias->ne[0]==input->ne[0] && bias->ne[1]==1 && bias->ne[2]==1 && bias->ne[3]==1)
            return bias_attention_layout(ctx,input,bias,heads,value_layout);
    }
#endif
    // Match only the existing plain per-channel bias ADD followed by its one BF16 boundary.
    if (restored->op != GGML_OP_CPY || restored->type != GGML_TYPE_F32 ||
        (restored->flags & GGML_TENSOR_FLAG_OUTPUT) || !restored->src[0]) return nullptr;
    auto * stored = restored->src[0];
    if (stored->op != GGML_OP_CPY || stored->type != GGML_TYPE_BF16 ||
        (stored->flags & GGML_TENSOR_FLAG_OUTPUT) || !stored->src[0]) return nullptr;
    auto * add = stored->src[0];
    if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 ||
        (add->flags & GGML_TENSOR_FLAG_OUTPUT) || !add->src[0] || !add->src[1]) return nullptr;
    auto * input = add->src[0];
    auto * bias = add->src[1];
    if (heads <= 0 || input->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(input) || !ggml_is_contiguous(bias) || !ggml_are_same_shape(input, restored) ||
        input->ne[2] != 1 || input->ne[3] != 1 || input->ne[0] % heads != 0 ||
        bias->ne[0] != input->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) return nullptr;
    return bias_attention_layout(ctx, input, bias, heads, value_layout);
}

static ggml_tensor * attention_unrounded_permuted(ggml_context * ctx, ggml_tensor * query, ggml_tensor * key,
                         ggml_tensor * value, int heads, int kv_heads, bool eager) {
    GGML_ASSERT(heads > 0 && kv_heads > 0 && value->ne[0] % kv_heads == 0);
    const int64_t tokens = value->ne[1];
    const int64_t query_tokens = query->ne[1];
    const int64_t dim = value->ne[0] / kv_heads;
    auto prepare_query_key = [&](ggml_tensor * input, int count, int64_t rows) {
        // A packed RoPE producer already has the exact old contiguous GEMM layout.
        if (input->ne[0] == dim && input->ne[1] == rows && input->ne[2] == count &&
            input->ne[3] == 1 && ggml_is_contiguous(input)) return input;
        GGML_ASSERT(input->ne[0] == dim * count && input->ne[1] == rows && input->ne[2] == 1 && input->ne[3] == 1);
        if (auto * packed = try_bias_attention_layout(ctx, input, count, false)) return packed;
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, input, dim, count, rows), 0, 2, 1, 3));
    };
    query = prepare_query_key(query, heads, query_tokens);
    key = prepare_query_key(key, kv_heads, tokens);
    if (auto * packed = try_bias_attention_layout(ctx, value, kv_heads, true)) value = packed;
    else value = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, value, dim, kv_heads, tokens), 1, 2, 0, 3));
    // ggml broadcasts GQA heads in groups, matching repeat_interleave in BitNet.
    auto * scores = ggml_mul_mat(ctx, key, query);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    // Preserve the accepted F32 score/softmax sequence without extra intermediate stores.
    auto * probabilities = eager ? softmax_score_f32(ctx, scores, scale)
                                : ggml_soft_max(ctx, ggml_scale(ctx, scores, scale));
    if (query_tokens != tokens) ggml_set_name(probabilities, "bitvla.last.partial_prob");
    auto * output = ggml_mul_mat(ctx, value, probabilities);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (query_tokens != tokens) ggml_set_name(output, "bitvla.last.partial_value");
    return ggml_permute(ctx,output,0,2,1,3);
}
ggml_tensor * attention(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,int kv_heads,bool eager) {
    auto * output=attention_unrounded_permuted(ctx,query,key,value,heads,kv_heads,eager);
    output=contiguous_layout_f32(ctx,output);
    return ggml_reshape_2d(ctx,output,value->ne[0]/kv_heads*heads,query->ne[1]);
}
ggml_tensor * attention_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,int kv_heads,bool eager) {
    auto * output=attention_unrounded_permuted(ctx,query,key,value,heads,kv_heads,eager);
    return contiguous_layout_i2s_encode(ctx,output);
}

ggml_tensor * attention_packed_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,bool eager) {
    GGML_ASSERT(query && key && value && heads>0);
    GGML_ASSERT(query->type==GGML_TYPE_F32 && key->type==GGML_TYPE_F32 && value->type==GGML_TYPE_F32);
    GGML_ASSERT(query->ne[0]>0 && query->ne[1]>0 && query->ne[2]==heads && query->ne[3]==1);
    GGML_ASSERT(key->ne[0]==query->ne[0] && key->ne[1]==query->ne[1] && key->ne[2]==heads && key->ne[3]==1);
    GGML_ASSERT(value->ne[0]==key->ne[1] && value->ne[1]==key->ne[0] && value->ne[2]==heads && value->ne[3]==1);
    GGML_ASSERT(ggml_is_contiguous(query) && ggml_is_contiguous(key) && ggml_is_contiguous(value));
    const int64_t dim=query->ne[0],tokens=key->ne[1],query_tokens=query->ne[1];
    // ggml broadcasts GQA heads in groups, matching repeat_interleave in BitNet.
    auto * scores = ggml_mul_mat(ctx, key, query);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    // Preserve the accepted F32 score/softmax sequence without extra intermediate stores.
    auto * probabilities = eager ? softmax_score_f32(ctx, scores, scale)
                                : ggml_soft_max(ctx, ggml_scale(ctx, scores, scale));
    if (query_tokens != tokens) ggml_set_name(probabilities, "bitvla.last.partial_prob");
    auto * output = ggml_mul_mat(ctx, value, probabilities);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (query_tokens != tokens) ggml_set_name(output, "bitvla.last.partial_value");
    return contiguous_layout_i2s_encode(ctx,ggml_permute(ctx,output,0,2,1,3));
}

ggml_tensor * attention_packed_f32(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,bool eager) {
    GGML_ASSERT(query && key && value && heads>0);
    GGML_ASSERT(query->type==GGML_TYPE_F32 && key->type==GGML_TYPE_F32 && value->type==GGML_TYPE_F32);
    GGML_ASSERT(query->ne[0]>0 && query->ne[1]>0 && query->ne[2]==heads && query->ne[3]==1);
    GGML_ASSERT(key->ne[0]==query->ne[0] && key->ne[1]==query->ne[1] && key->ne[2]==heads && key->ne[3]==1);
    GGML_ASSERT(value->ne[0]==key->ne[1] && value->ne[1]==key->ne[0] && value->ne[2]==heads && value->ne[3]==1);
    GGML_ASSERT(ggml_is_contiguous(query) && ggml_is_contiguous(key) && ggml_is_contiguous(value));
    auto * scores=ggml_mul_mat(ctx,key,query);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
    const float scale=1.0f/std::sqrt(static_cast<float>(query->ne[0]));
    auto * probabilities=eager?softmax_score_f32(ctx,scores,scale):ggml_soft_max(ctx,ggml_scale(ctx,scores,scale));
    auto * output=ggml_mul_mat(ctx,value,probabilities);ggml_mul_mat_set_prec(output,GGML_PREC_F32);
    output=contiguous_layout_f32(ctx,ggml_permute(ctx,output,0,2,1,3));
    return ggml_reshape_2d(ctx,output,query->ne[0]*heads,query->ne[1]);
}

ggml_tensor * attention_rms_i2s_encode(ggml_context * ctx,ggml_tensor * query,ggml_tensor * key,ggml_tensor * value,int heads,int kv_heads,bool eager,ggml_tensor * weight,float eps) {
    if(heads<=0 || kv_heads<=0 || value->ne[0]<=0 || value->ne[0]%kv_heads!=0 ||
       value->ne[0]/kv_heads>2560 || heads!=2560/(value->ne[0]/kv_heads) || 2560%(value->ne[0]/kv_heads)!=0 ||
       weight->type!=GGML_TYPE_F32 || !ggml_is_contiguous(weight) || weight->ne[0]!=2560 || weight->ne[1]!=1 || weight->ne[2]!=1 || weight->ne[3]!=1 ||
       query->ne[1]<=0 || query->ne[1]>INT32_MAX/2568)return nullptr;
    auto * raw=attention_unrounded_permuted(ctx,query,key,value,heads,kv_heads,eager);
    return permuted_rms_affine_i2s_encode(ctx,raw,weight,eps);
}

} // namespace robotcpp::bitvla
