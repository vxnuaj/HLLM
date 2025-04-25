# AOT ID: ['2_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_add_embedding_eq_mean_mul_pow_rsqrt_0 = async_compile.cpp_pybinding(['float*', 'const bool*', 'const bool*', 'const bool*', 'const bool*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'bool*', 'bool*', 'bool*', 'bool*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1,
                       const bool* in_ptr2,
                       const bool* in_ptr3,
                       const int64_t* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr6)
{
    auto out_ptr5 = in_out_ptr0;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512LL)))
                {
                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x0));
                    auto tmp1 = tmp0.to<int64_t,2>();
                    auto tmp2 = static_cast<int64_t>(0);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512LL)))
                {
                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr1 + static_cast<int64_t>(x0));
                    auto tmp1 = tmp0.to<int64_t,2>();
                    auto tmp2 = static_cast<int64_t>(0);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                    tmp4.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512LL)))
                {
                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr2 + static_cast<int64_t>(x0));
                    auto tmp1 = tmp0.to<int64_t,2>();
                    auto tmp2 = static_cast<int64_t>(0);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                    tmp4.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(512LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512LL)))
                {
                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr3 + static_cast<int64_t>(x0));
                    auto tmp1 = tmp0.to<int64_t,2>();
                    auto tmp2 = static_cast<int64_t>(0);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                    tmp4.store(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = in_ptr4[static_cast<int64_t>(x0)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(130816LL + x1), static_cast<int64_t>(4));
                            auto tmp1 = 10000LL;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 10000LL), "index out of bounds: 0 <= tmp7 < 10000LL");
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 256LL*tmp5), static_cast<int64_t>(4));
                            auto tmp11 = tmp9 + tmp10;
                            auto tmp12 = tmp11 * tmp11;
                            tmp9.store(out_ptr4 + static_cast<int64_t>(x1 + 256LL*x0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(130816LL + x1), static_cast<int64_t>(4));
                        auto tmp3 = in_out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 * tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr6 + static_cast<int64_t>(x1 + 256LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_1 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 32LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 16416LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_stack_2 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                        auto tmp1 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                        auto tmp5 = in_ptr2[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp6 = c10::convert<float>(tmp5);
                        auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                        auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                        auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        out_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                        out_ptr1[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_3 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const half*', 'const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'const bool*', 'const half*', 'const half*', 'const half*', 'const half*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const half* in_ptr1,
                       const float* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       const half* in_ptr8,
                       const half* in_ptr9,
                       const half* in_ptr10,
                       const half* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr14)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr12 = in_out_ptr1;
    auto out_ptr13 = in_out_ptr2;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 131328LL*x0));
                    }
                }
            }
        }
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16384LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16384LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 16384LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr1 + static_cast<int64_t>(x1 + 16416LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(32LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr2[static_cast<int64_t>(33LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr3[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x2 + 32LL*x1 + 16384LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL)))), static_cast<int64_t>(4));
                                    auto tmp2 = tmp0 * tmp1;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr4[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                }
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr7 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                            }
                        }
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr7 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp8 = out_ptr5[static_cast<int64_t>(x0)];
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = tmp10.exp();
                                tmp11.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                                tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr6[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16384LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16384LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr8 + static_cast<int64_t>(x1 + 16384LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr7 + static_cast<int64_t>(x1 + 16416LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16384LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16384LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr9 + static_cast<int64_t>(x1 + 16384LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr8 + static_cast<int64_t>(x1 + 16416LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16384LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16384LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr10 + static_cast<int64_t>(x1 + 16384LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr9 + static_cast<int64_t>(x1 + 16416LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                    {
                        for(int64_t x3=static_cast<int64_t>(0LL); x3<static_cast<int64_t>(512LL); x3+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL) && x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(512LL)))
                                {
                                    alignas(std::max(std::size_t(4), alignof(float))) float tmp0[4*4];
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr6 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr10 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr11 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr11 + static_cast<int64_t>(x1 + 131328LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(1LL))
                        {
                            {
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + 512LL*x0)];
                                    auto tmp1 = in_ptr12[static_cast<int64_t>(256LL + x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL)))), static_cast<int64_t>(131072LL))))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    tmp_acc0 = tmp_acc0 + tmp2;
                                }
                            }
                        }
                        out_ptr12[static_cast<int64_t>(x1 + 32LL*x0)] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<int64_t>(130816LL + x1), static_cast<int64_t>(4));
                                    auto tmp3 = tmp1 + tmp2;
                                    auto tmp4 = tmp0 + tmp3;
                                    auto tmp5 = tmp4 * tmp4;
                                    tmp4.store(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0));
                                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        in_out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr13 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(in_out_ptr2 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = in_out_ptr2[static_cast<int64_t>(x0)];
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp5 = tmp3 * tmp4;
                                tmp5.store(out_ptr14 + static_cast<int64_t>(x1 + 256LL*x0));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_4 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(10912LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(10912LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_5 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = tmp0 * tmp0;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = in_out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_6 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 32LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 16416LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_stack_7 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                        auto tmp1 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                        auto tmp5 = in_ptr2[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp6 = c10::convert<float>(tmp5);
                        auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                        auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                        auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        out_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                        out_ptr1[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_8 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'const bool*', 'const half*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const half* in_ptr2,
                       const half* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const half* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr10)
{
    auto out_ptr3 = in_out_ptr0;
    auto out_ptr8 = in_out_ptr1;
    auto out_ptr9 = in_out_ptr2;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 131328LL*x0));
                    }
                }
            }
        }
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp1 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr1[static_cast<int64_t>(33LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp5 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x1 + 16384LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL)))), static_cast<int64_t>(4));
                                    auto tmp2 = tmp0 * tmp1;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr3[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                }
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                            }
                        }
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr4[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp8 = out_ptr4[static_cast<int64_t>(x0)];
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = tmp10.exp();
                                tmp11.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                                tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr5[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                    {
                        for(int64_t x3=static_cast<int64_t>(0LL); x3<static_cast<int64_t>(512LL); x3+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL) && x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(512LL)))
                                {
                                    alignas(std::max(std::size_t(4), alignof(float))) float tmp0[4*4];
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr6 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr7 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr7 + static_cast<int64_t>(x1 + 131328LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(1LL))
                        {
                            {
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + 512LL*x0)];
                                    auto tmp1 = in_ptr8[static_cast<int64_t>(256LL + x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL)))), static_cast<int64_t>(131072LL))))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    tmp_acc0 = tmp_acc0 + tmp2;
                                }
                            }
                        }
                        out_ptr8[static_cast<int64_t>(x1 + 32LL*x0)] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(in_out_ptr1 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0 * tmp0;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        in_out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(in_out_ptr2 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = in_out_ptr2[static_cast<int64_t>(x0)];
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp5 = tmp3 * tmp4;
                                tmp5.store(out_ptr10 + static_cast<int64_t>(x1 + 256LL*x0));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_9 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(10912LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(10912LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_10 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = tmp0 * tmp0;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = in_out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_11 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 32LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 16416LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_stack_12 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                        auto tmp1 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                        auto tmp5 = in_ptr2[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp6 = c10::convert<float>(tmp5);
                        auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                        auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                        auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        out_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                        out_ptr1[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_13 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'const bool*', 'const half*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const half* in_ptr2,
                       const half* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const half* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr10)
{
    auto out_ptr3 = in_out_ptr0;
    auto out_ptr8 = in_out_ptr1;
    auto out_ptr9 = in_out_ptr2;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 131328LL*x0));
                    }
                }
            }
        }
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp1 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr1[static_cast<int64_t>(33LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp5 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x1 + 16384LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL)))), static_cast<int64_t>(4));
                                    auto tmp2 = tmp0 * tmp1;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr3[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                }
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                            }
                        }
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr4[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp8 = out_ptr4[static_cast<int64_t>(x0)];
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = tmp10.exp();
                                tmp11.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                                tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr5[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                    {
                        for(int64_t x3=static_cast<int64_t>(0LL); x3<static_cast<int64_t>(512LL); x3+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL) && x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(512LL)))
                                {
                                    alignas(std::max(std::size_t(4), alignof(float))) float tmp0[4*4];
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr6 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr7 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr7 + static_cast<int64_t>(x1 + 131328LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(1LL))
                        {
                            {
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + 512LL*x0)];
                                    auto tmp1 = in_ptr8[static_cast<int64_t>(256LL + x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL)))), static_cast<int64_t>(131072LL))))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    tmp_acc0 = tmp_acc0 + tmp2;
                                }
                            }
                        }
                        out_ptr8[static_cast<int64_t>(x1 + 32LL*x0)] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(in_out_ptr1 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0 * tmp0;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        in_out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(in_out_ptr2 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = in_out_ptr2[static_cast<int64_t>(x0)];
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp5 = tmp3 * tmp4;
                                tmp5.store(out_ptr10 + static_cast<int64_t>(x1 + 256LL*x0));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_14 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(10912LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(10912LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_15 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = tmp0 * tmp0;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = in_out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_16 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(32LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 32LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 16416LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_stack_17 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                        auto tmp1 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                        auto tmp5 = in_ptr2[static_cast<int64_t>(32LL + 2LL*x1)];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp6 = c10::convert<float>(tmp5);
                        auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                        auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                        auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        out_ptr0[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                        out_ptr1[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_18 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'const bool*', 'const half*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const half* in_ptr2,
                       const half* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const half* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr10)
{
    auto out_ptr3 = in_out_ptr0;
    auto out_ptr8 = in_out_ptr1;
    auto out_ptr9 = in_out_ptr2;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 131328LL*x0));
                    }
                }
            }
        }
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr1[static_cast<int64_t>(32LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp1 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr1[static_cast<int64_t>(33LL + 2LL*x1 + 16416LL*x0)];
                            auto tmp5 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x1 + 16384LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL)))), static_cast<int64_t>(4));
                                    auto tmp2 = tmp0 * tmp1;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr3[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                }
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                            }
                        }
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr4[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr6 + static_cast<int64_t>(x1));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                auto tmp8 = out_ptr4[static_cast<int64_t>(x0)];
                                auto tmp2 = static_cast<float>(0.17677669529663687);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 * tmp3;
                                auto tmp5 = -std::numeric_limits<float>::infinity();
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = tmp10.exp();
                                tmp11.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                                tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr5[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(32LL); x2+=static_cast<int64_t>(4LL))
                    {
                        for(int64_t x3=static_cast<int64_t>(0LL); x3<static_cast<int64_t>(512LL); x3+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(32LL) && x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(512LL)))
                                {
                                    alignas(std::max(std::size_t(4), alignof(float))) float tmp0[4*4];
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr5 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr6 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(8LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<half>::loadu(in_ptr7 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::convert<float,2,half,1>(tmp0);
                            tmp1.store(out_ptr7 + static_cast<int64_t>(x1 + 131328LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(32LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(1LL))
                        {
                            {
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x2 + 512LL*x0)];
                                    auto tmp1 = in_ptr8[static_cast<int64_t>(256LL + x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(8LL))) + 131328LL*(c10::div_floor_integer(static_cast<int64_t>(x1 + 32LL*x2 + 16384LL*((static_cast<int64_t>(x0) % static_cast<int64_t>(8LL)))), static_cast<int64_t>(131072LL))))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    tmp_acc0 = tmp_acc0 + tmp2;
                                }
                            }
                        }
                        out_ptr8[static_cast<int64_t>(x1 + 32LL*x0)] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(in_out_ptr1 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0 * tmp0;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        in_out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(in_out_ptr2 + static_cast<int64_t>(x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = in_out_ptr2[static_cast<int64_t>(x0)];
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp5 = tmp3 * tmp4;
                                tmp5.store(out_ptr10 + static_cast<int64_t>(x1 + 256LL*x0));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_silu_19 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(10912LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(10912LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_20 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4096LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = tmp0 * tmp0;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1920928955078125e-07);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                        auto tmp1 = in_out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(out_ptr1 + static_cast<int64_t>(x1 + 256LL*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(256LL + x1 + 131328LL*x0), static_cast<int64_t>(4));
                            tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 131072LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(256LL + x1 + 131328LL*x0), static_cast<int64_t>(4));
                            tmp0.store(out_ptr1 + static_cast<int64_t>(x1 + 131072LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(256LL + x1 + 131328LL*x0), static_cast<int64_t>(4));
                            tmp0.store(out_ptr2 + static_cast<int64_t>(x1 + 131072LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(131072LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(131072LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(256LL + x1 + 131328LL*x0), static_cast<int64_t>(4));
                            tmp0.store(out_ptr3 + static_cast<int64_t>(x1 + 131072LL*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81 = args
    args.clear()
    assert_size_stride(primals_1, (10000, 256), (256, 1))
    assert_size_stride(primals_2, (16, 1), (1, 1))
    assert_size_stride(primals_3, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, 256), (256, 1))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (32, 256), (256, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (256, 256), (256, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (16, 512, 32), (16384, 32, 1))
    assert_size_stride(primals_12, (16, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_13, (512, 32), (32, 1))
    assert_size_stride(primals_14, (512, 32), (32, 1))
    assert_size_stride(primals_15, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (682, 256), (256, 1))
    assert_size_stride(primals_18, (682, ), (1, ))
    assert_size_stride(primals_19, (682, 256), (256, 1))
    assert_size_stride(primals_20, (682, ), (1, ))
    assert_size_stride(primals_21, (256, 682), (682, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, 256), (256, 1))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (32, 256), (256, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (256, 256), (256, 1))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (16, 512, 32), (16384, 32, 1))
    assert_size_stride(primals_31, (16, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_32, (512, 32), (32, 1))
    assert_size_stride(primals_33, (512, 32), (32, 1))
    assert_size_stride(primals_34, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (682, 256), (256, 1))
    assert_size_stride(primals_37, (682, ), (1, ))
    assert_size_stride(primals_38, (682, 256), (256, 1))
    assert_size_stride(primals_39, (682, ), (1, ))
    assert_size_stride(primals_40, (256, 682), (682, 1))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 256), (256, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (32, 256), (256, 1))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (256, 256), (256, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (16, 512, 32), (16384, 32, 1))
    assert_size_stride(primals_50, (16, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_51, (512, 32), (32, 1))
    assert_size_stride(primals_52, (512, 32), (32, 1))
    assert_size_stride(primals_53, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (682, 256), (256, 1))
    assert_size_stride(primals_56, (682, ), (1, ))
    assert_size_stride(primals_57, (682, 256), (256, 1))
    assert_size_stride(primals_58, (682, ), (1, ))
    assert_size_stride(primals_59, (256, 682), (682, 1))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, 256), (256, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (32, 256), (256, 1))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (256, 256), (256, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (16, 512, 32), (16384, 32, 1))
    assert_size_stride(primals_69, (16, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_70, (512, 32), (32, 1))
    assert_size_stride(primals_71, (512, 32), (32, 1))
    assert_size_stride(primals_72, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (682, 256), (256, 1))
    assert_size_stride(primals_75, (682, ), (1, ))
    assert_size_stride(primals_76, (682, 256), (256, 1))
    assert_size_stride(primals_77, (682, ), (1, ))
    assert_size_stride(primals_78, (256, 682), (682, 1))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (10000, ), (1, ))
    buf20 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.bool)
    buf54 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.bool)
    buf88 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.bool)
    buf122 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.bool)
    buf0 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    buf1 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf2 = reinterpret_tensor(buf1, (16, 1, 1), (1, 1, 1), 0); del buf1  # reuse
    buf3 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused_add_embedding_eq_mean_mul_pow_rsqrt_0(buf2, primals_15, primals_34, primals_53, primals_72, primals_2, primals_1, primals_3, primals_4, buf20, buf54, buf88, buf122, buf0, buf3)
    del primals_15
    del primals_34
    del primals_53
    del primals_72
    buf5 = empty_strided_cpu((16, 32), (32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [k], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, reinterpret_tensor(buf3, (16, 256), (256, 1), 0), reinterpret_tensor(primals_7, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf5)
    del primals_8
    buf9 = empty_strided_cpu((16, 513, 32), (16416, 32, 1), torch.float32)
    buf8 = reinterpret_tensor(buf9, (16, 1, 32), (16416, 32, 1), 16384)  # alias
    cpp_fused_cat_1(buf5, buf8)
    buf4 = empty_strided_cpu((16, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, reinterpret_tensor(buf3, (16, 256), (256, 1), 0), reinterpret_tensor(primals_5, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf4)
    del primals_6
    buf15 = empty_strided_cpu((16, 8, 1, 16, 2), (256, 32, 32, 2, 1), torch.float32)
    buf13 = reinterpret_tensor(buf15, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 0)  # alias
    buf14 = reinterpret_tensor(buf15, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 1)  # alias
    cpp_fused_stack_2(buf4, primals_13, primals_14, buf13, buf14)
    buf6 = buf4; del buf4  # reuse
    # Topologically Sorted Source Nodes: [v], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, reinterpret_tensor(buf3, (16, 256), (256, 1), 0), reinterpret_tensor(primals_9, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
    del primals_10
    buf12 = empty_strided_cpu((16, 513, 256), (131328, 256, 1), torch.float32)
    buf11 = reinterpret_tensor(buf12, (16, 1, 256), (131328, 256, 1), 131072)  # alias
    buf7 = reinterpret_tensor(buf9, (16, 512, 32), (16416, 32, 1), 0)  # alias
    buf18 = empty_strided_cpu((16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), torch.float32)
    buf16 = reinterpret_tensor(buf18, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf17 = reinterpret_tensor(buf18, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf19 = empty_strided_cpu((128, 1, 512), (512, 512, 1), torch.float32)
    buf21 = empty_strided_cpu((16, 8, 1, 1), (8, 1, 128, 128), torch.float32)
    buf22 = reinterpret_tensor(buf19, (16, 8, 1, 512), (4096, 512, 65536, 1), 0); del buf19  # reuse
    buf23 = empty_strided_cpu((16, 8, 1, 1), (8, 1, 128, 128), torch.float32)
    buf24 = reinterpret_tensor(buf22, (16, 8, 1, 512), (4096, 512, 512, 1), 0); del buf22  # reuse
    buf43 = empty_strided_cpu((16, 513, 32), (16416, 32, 1), torch.float32)
    buf41 = reinterpret_tensor(buf43, (16, 512, 32), (16416, 32, 1), 0)  # alias
    buf77 = empty_strided_cpu((16, 513, 32), (16416, 32, 1), torch.float32)
    buf75 = reinterpret_tensor(buf77, (16, 512, 32), (16416, 32, 1), 0)  # alias
    buf111 = empty_strided_cpu((16, 513, 32), (16416, 32, 1), torch.float32)
    buf109 = reinterpret_tensor(buf111, (16, 512, 32), (16416, 32, 1), 0)  # alias
    buf148 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    buf10 = reinterpret_tensor(buf12, (16, 512, 256), (131328, 256, 1), 0)  # alias
    buf25 = empty_strided_cpu((128, 1, 32), (32, 32, 1), torch.float32)
    buf26 = reinterpret_tensor(buf25, (16, 1, 256), (256, 256, 1), 0); del buf25  # reuse
    buf27 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf28 = reinterpret_tensor(buf27, (16, 1, 1), (1, 1, 1), 0); del buf27  # reuse
    buf29 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_3(buf24, buf26, buf28, buf6, primals_11, buf9, primals_13, primals_14, buf15, buf18, buf20, primals_30, primals_49, primals_68, primals_12, buf12, buf0, primals_3, primals_16, buf11, buf7, buf16, buf17, buf21, buf23, buf41, buf75, buf109, buf148, buf10, buf29)
    del buf16
    del buf17
    del primals_11
    del primals_12
    del primals_30
    del primals_49
    del primals_68
    buf30 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, reinterpret_tensor(buf29, (16, 256), (256, 1), 0), reinterpret_tensor(primals_17, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf30)
    del primals_18
    buf31 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, reinterpret_tensor(buf29, (16, 256), (256, 1), 0), reinterpret_tensor(primals_19, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf31)
    del primals_20
    buf32 = empty_strided_cpu((16, 1, 682), (682, 682, 1), torch.float32)
    cpp_fused_mul_silu_4(buf30, buf31, buf32)
    buf33 = buf6; del buf6  # reuse
    # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, reinterpret_tensor(buf32, (16, 682), (682, 1), 0), reinterpret_tensor(primals_21, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf33)
    del primals_22
    buf34 = reinterpret_tensor(buf33, (16, 1, 256), (256, 256, 1), 0); del buf33  # reuse
    buf35 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf36 = reinterpret_tensor(buf35, (16, 1, 1), (1, 1, 1), 0); del buf35  # reuse
    buf37 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_5(buf34, buf36, buf29, primals_23, buf37)
    buf39 = buf5; del buf5  # reuse
    # Topologically Sorted Source Nodes: [k_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, reinterpret_tensor(buf37, (16, 256), (256, 1), 0), reinterpret_tensor(primals_26, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf39)
    del primals_27
    buf42 = reinterpret_tensor(buf43, (16, 1, 32), (16416, 32, 1), 16384)  # alias
    cpp_fused_cat_6(buf39, buf42)
    buf38 = empty_strided_cpu((16, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_25, reinterpret_tensor(buf37, (16, 256), (256, 1), 0), reinterpret_tensor(primals_24, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf38)
    del primals_25
    buf49 = empty_strided_cpu((16, 8, 1, 16, 2), (256, 32, 32, 2, 1), torch.float32)
    buf47 = reinterpret_tensor(buf49, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 0)  # alias
    buf48 = reinterpret_tensor(buf49, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 1)  # alias
    cpp_fused_stack_7(buf38, primals_32, primals_33, buf47, buf48)
    buf40 = buf38; del buf38  # reuse
    # Topologically Sorted Source Nodes: [v_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_29, reinterpret_tensor(buf37, (16, 256), (256, 1), 0), reinterpret_tensor(primals_28, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf40)
    del primals_29
    buf46 = empty_strided_cpu((16, 513, 256), (131328, 256, 1), torch.float32)
    buf45 = reinterpret_tensor(buf46, (16, 1, 256), (131328, 256, 1), 131072)  # alias
    buf52 = buf18; del buf18  # reuse
    buf50 = reinterpret_tensor(buf52, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf51 = reinterpret_tensor(buf52, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf53 = empty_strided_cpu((128, 1, 512), (512, 512, 1), torch.float32)
    buf55 = buf23; del buf23  # reuse
    buf56 = reinterpret_tensor(buf53, (16, 8, 1, 512), (4096, 512, 65536, 1), 0); del buf53  # reuse
    buf57 = buf21; del buf21  # reuse
    buf58 = reinterpret_tensor(buf56, (16, 8, 1, 512), (4096, 512, 512, 1), 0); del buf56  # reuse
    buf146 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    buf44 = reinterpret_tensor(buf46, (16, 512, 256), (131328, 256, 1), 0)  # alias
    buf59 = empty_strided_cpu((128, 1, 32), (32, 32, 1), torch.float32)
    buf60 = reinterpret_tensor(buf59, (16, 1, 256), (256, 256, 1), 0); del buf59  # reuse
    buf61 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf62 = reinterpret_tensor(buf61, (16, 1, 1), (1, 1, 1), 0); del buf61  # reuse
    buf63 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_8(buf58, buf60, buf62, buf40, buf43, primals_32, primals_33, buf49, buf52, buf54, primals_31, buf46, buf34, primals_35, buf45, buf50, buf51, buf55, buf57, buf146, buf44, buf63)
    del buf50
    del buf51
    del primals_31
    buf64 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, reinterpret_tensor(buf63, (16, 256), (256, 1), 0), reinterpret_tensor(primals_36, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf64)
    del primals_37
    buf65 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, reinterpret_tensor(buf63, (16, 256), (256, 1), 0), reinterpret_tensor(primals_38, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf65)
    del primals_39
    buf66 = empty_strided_cpu((16, 1, 682), (682, 682, 1), torch.float32)
    cpp_fused_mul_silu_9(buf64, buf65, buf66)
    buf67 = buf40; del buf40  # reuse
    # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, reinterpret_tensor(buf66, (16, 682), (682, 1), 0), reinterpret_tensor(primals_40, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf67)
    del primals_41
    buf68 = reinterpret_tensor(buf67, (16, 1, 256), (256, 256, 1), 0); del buf67  # reuse
    buf69 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf70 = reinterpret_tensor(buf69, (16, 1, 1), (1, 1, 1), 0); del buf69  # reuse
    buf71 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_10(buf68, buf70, buf63, primals_42, buf71)
    buf73 = buf39; del buf39  # reuse
    # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_46, reinterpret_tensor(buf71, (16, 256), (256, 1), 0), reinterpret_tensor(primals_45, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf73)
    del primals_46
    buf76 = reinterpret_tensor(buf77, (16, 1, 32), (16416, 32, 1), 16384)  # alias
    cpp_fused_cat_11(buf73, buf76)
    buf72 = empty_strided_cpu((16, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, reinterpret_tensor(buf71, (16, 256), (256, 1), 0), reinterpret_tensor(primals_43, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf72)
    del primals_44
    buf83 = empty_strided_cpu((16, 8, 1, 16, 2), (256, 32, 32, 2, 1), torch.float32)
    buf81 = reinterpret_tensor(buf83, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 0)  # alias
    buf82 = reinterpret_tensor(buf83, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 1)  # alias
    cpp_fused_stack_12(buf72, primals_51, primals_52, buf81, buf82)
    buf74 = buf72; del buf72  # reuse
    # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, reinterpret_tensor(buf71, (16, 256), (256, 1), 0), reinterpret_tensor(primals_47, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf74)
    del primals_48
    buf80 = empty_strided_cpu((16, 513, 256), (131328, 256, 1), torch.float32)
    buf79 = reinterpret_tensor(buf80, (16, 1, 256), (131328, 256, 1), 131072)  # alias
    buf86 = buf52; del buf52  # reuse
    buf84 = reinterpret_tensor(buf86, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf85 = reinterpret_tensor(buf86, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf87 = empty_strided_cpu((128, 1, 512), (512, 512, 1), torch.float32)
    buf89 = buf57; del buf57  # reuse
    buf90 = reinterpret_tensor(buf87, (16, 8, 1, 512), (4096, 512, 65536, 1), 0); del buf87  # reuse
    buf91 = buf55; del buf55  # reuse
    buf92 = reinterpret_tensor(buf90, (16, 8, 1, 512), (4096, 512, 512, 1), 0); del buf90  # reuse
    buf144 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    buf78 = reinterpret_tensor(buf80, (16, 512, 256), (131328, 256, 1), 0)  # alias
    buf93 = empty_strided_cpu((128, 1, 32), (32, 32, 1), torch.float32)
    buf94 = reinterpret_tensor(buf93, (16, 1, 256), (256, 256, 1), 0); del buf93  # reuse
    buf95 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf96 = reinterpret_tensor(buf95, (16, 1, 1), (1, 1, 1), 0); del buf95  # reuse
    buf97 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_13(buf92, buf94, buf96, buf74, buf77, primals_51, primals_52, buf83, buf86, buf88, primals_50, buf80, buf68, primals_54, buf79, buf84, buf85, buf89, buf91, buf144, buf78, buf97)
    del buf84
    del buf85
    del primals_50
    buf98 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, reinterpret_tensor(buf97, (16, 256), (256, 1), 0), reinterpret_tensor(primals_55, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf98)
    del primals_56
    buf99 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, reinterpret_tensor(buf97, (16, 256), (256, 1), 0), reinterpret_tensor(primals_57, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf99)
    del primals_58
    buf100 = empty_strided_cpu((16, 1, 682), (682, 682, 1), torch.float32)
    cpp_fused_mul_silu_14(buf98, buf99, buf100)
    buf101 = buf74; del buf74  # reuse
    # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, reinterpret_tensor(buf100, (16, 682), (682, 1), 0), reinterpret_tensor(primals_59, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf101)
    del primals_60
    buf102 = reinterpret_tensor(buf101, (16, 1, 256), (256, 256, 1), 0); del buf101  # reuse
    buf103 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf104 = reinterpret_tensor(buf103, (16, 1, 1), (1, 1, 1), 0); del buf103  # reuse
    buf105 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_15(buf102, buf104, buf97, primals_61, buf105)
    buf107 = buf73; del buf73  # reuse
    # Topologically Sorted Source Nodes: [k_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, reinterpret_tensor(buf105, (16, 256), (256, 1), 0), reinterpret_tensor(primals_64, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf107)
    del primals_65
    buf110 = reinterpret_tensor(buf111, (16, 1, 32), (16416, 32, 1), 16384)  # alias
    cpp_fused_cat_16(buf107, buf110)
    del buf107
    buf106 = empty_strided_cpu((16, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_63, reinterpret_tensor(buf105, (16, 256), (256, 1), 0), reinterpret_tensor(primals_62, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf106)
    del primals_63
    buf117 = empty_strided_cpu((16, 8, 1, 16, 2), (256, 32, 32, 2, 1), torch.float32)
    buf115 = reinterpret_tensor(buf117, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 0)  # alias
    buf116 = reinterpret_tensor(buf117, (16, 8, 1, 16, 1), (256, 32, 32, 2, 1), 1)  # alias
    cpp_fused_stack_17(buf106, primals_70, primals_71, buf115, buf116)
    buf108 = buf106; del buf106  # reuse
    # Topologically Sorted Source Nodes: [v_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, reinterpret_tensor(buf105, (16, 256), (256, 1), 0), reinterpret_tensor(primals_66, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf108)
    del primals_67
    buf114 = empty_strided_cpu((16, 513, 256), (131328, 256, 1), torch.float32)
    buf113 = reinterpret_tensor(buf114, (16, 1, 256), (131328, 256, 1), 131072)  # alias
    buf120 = buf86; del buf86  # reuse
    buf118 = reinterpret_tensor(buf120, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf119 = reinterpret_tensor(buf120, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf121 = empty_strided_cpu((128, 1, 512), (512, 512, 1), torch.float32)
    buf123 = buf91; del buf91  # reuse
    buf124 = reinterpret_tensor(buf121, (16, 8, 1, 512), (4096, 512, 65536, 1), 0); del buf121  # reuse
    buf125 = buf89; del buf89  # reuse
    buf126 = reinterpret_tensor(buf124, (16, 8, 1, 512), (4096, 512, 512, 1), 0); del buf124  # reuse
    buf142 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    buf112 = reinterpret_tensor(buf114, (16, 512, 256), (131328, 256, 1), 0)  # alias
    buf127 = empty_strided_cpu((128, 1, 32), (32, 32, 1), torch.float32)
    buf128 = reinterpret_tensor(buf127, (16, 1, 256), (256, 256, 1), 0); del buf127  # reuse
    buf129 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf130 = reinterpret_tensor(buf129, (16, 1, 1), (1, 1, 1), 0); del buf129  # reuse
    buf131 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused__softmax_add_bmm_cat_clone_div_masked_fill_mean_mul_pow_rsqrt_stack_18(buf126, buf128, buf130, buf108, buf111, primals_70, primals_71, buf117, buf120, buf122, primals_69, buf114, buf102, primals_73, buf113, buf118, buf119, buf123, buf125, buf142, buf112, buf131)
    del buf118
    del buf119
    del buf120
    del buf123
    del buf125
    del primals_69
    buf132 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_75, reinterpret_tensor(buf131, (16, 256), (256, 1), 0), reinterpret_tensor(primals_74, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf132)
    del primals_75
    buf133 = empty_strided_cpu((16, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_77, reinterpret_tensor(buf131, (16, 256), (256, 1), 0), reinterpret_tensor(primals_76, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf133)
    del primals_77
    buf134 = empty_strided_cpu((16, 1, 682), (682, 682, 1), torch.float32)
    cpp_fused_mul_silu_19(buf132, buf133, buf134)
    buf135 = buf108; del buf108  # reuse
    # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, reinterpret_tensor(buf134, (16, 682), (682, 1), 0), reinterpret_tensor(primals_78, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf135)
    del primals_79
    buf136 = reinterpret_tensor(buf135, (16, 1, 256), (256, 256, 1), 0); del buf135  # reuse
    buf137 = empty_strided_cpu((16, 1, 1), (1, 16, 16), torch.float32)
    buf138 = reinterpret_tensor(buf137, (16, 1, 1), (1, 1, 1), 0); del buf137  # reuse
    buf139 = empty_strided_cpu((16, 1, 256), (256, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_20(buf136, buf138, buf131, primals_80, buf139)
    buf140 = empty_strided_cpu((16, 10000), (10000, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, reinterpret_tensor(buf139, (16, 256), (256, 1), 0), reinterpret_tensor(primals_1, (256, 10000), (1, 256), 0), alpha=1, beta=1, out=buf140)
    del primals_81
    buf141 = empty_strided_cpu((16, 8, 512, 32), (131072, 16384, 32, 1), torch.float32)
    buf143 = empty_strided_cpu((16, 8, 512, 32), (131072, 16384, 32, 1), torch.float32)
    buf145 = empty_strided_cpu((16, 8, 512, 32), (131072, 16384, 32, 1), torch.float32)
    buf147 = empty_strided_cpu((16, 8, 512, 32), (131072, 16384, 32, 1), torch.float32)
    cpp_fused_clone_21(buf114, buf80, buf46, buf12, buf141, buf143, buf145, buf147)
    return (reinterpret_tensor(buf140, (16, 1, 10000), (10000, 10000, 1), 0), reinterpret_tensor(buf12, (16, 512, 256), (131328, 256, 1), 256), reinterpret_tensor(buf9, (16, 512, 32), (16416, 32, 1), 32), reinterpret_tensor(buf46, (16, 512, 256), (131328, 256, 1), 256), reinterpret_tensor(buf43, (16, 512, 32), (16416, 32, 1), 32), reinterpret_tensor(buf80, (16, 512, 256), (131328, 256, 1), 256), reinterpret_tensor(buf77, (16, 512, 32), (16416, 32, 1), 32), reinterpret_tensor(buf114, (16, 512, 256), (131328, 256, 1), 256), reinterpret_tensor(buf111, (16, 512, 32), (16416, 32, 1), 32), primals_2, primals_4, primals_16, primals_23, primals_35, primals_42, primals_54, primals_61, primals_73, primals_80, buf0, reinterpret_tensor(primals_3, (1, 256), (131072, 1), 130816), buf2, reinterpret_tensor(buf3, (16, 256), (256, 1), 0), reinterpret_tensor(primals_13, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_14, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_13, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_14, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf20, buf24, buf26, buf28, reinterpret_tensor(buf29, (16, 256), (256, 1), 0), buf30, buf31, reinterpret_tensor(buf32, (16, 682), (682, 1), 0), buf34, buf36, reinterpret_tensor(buf37, (16, 256), (256, 1), 0), reinterpret_tensor(primals_32, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_33, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_32, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_33, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf54, buf58, buf60, buf62, reinterpret_tensor(buf63, (16, 256), (256, 1), 0), buf64, buf65, reinterpret_tensor(buf66, (16, 682), (682, 1), 0), buf68, buf70, reinterpret_tensor(buf71, (16, 256), (256, 1), 0), reinterpret_tensor(primals_51, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_52, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_51, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_52, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf88, buf92, buf94, buf96, reinterpret_tensor(buf97, (16, 256), (256, 1), 0), buf98, buf99, reinterpret_tensor(buf100, (16, 682), (682, 1), 0), buf102, buf104, reinterpret_tensor(buf105, (16, 256), (256, 1), 0), reinterpret_tensor(primals_70, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_71, (1, 1, 1, 16), (32, 32, 32, 2), 32), reinterpret_tensor(primals_70, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_71, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf122, buf126, buf128, buf130, reinterpret_tensor(buf131, (16, 256), (256, 1), 0), buf132, buf133, reinterpret_tensor(buf134, (16, 682), (682, 1), 0), buf136, buf138, reinterpret_tensor(buf139, (16, 256), (256, 1), 0), primals_1, primals_78, primals_76, primals_74, reinterpret_tensor(buf141, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf117, (128, 32, 1), (32, 1, 32), 0), reinterpret_tensor(buf142, (128, 512, 32), (16384, 1, 512), 0), primals_66, primals_64, primals_62, primals_59, primals_57, primals_55, reinterpret_tensor(buf143, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf83, (128, 32, 1), (32, 1, 32), 0), reinterpret_tensor(buf144, (128, 512, 32), (16384, 1, 512), 0), primals_47, primals_45, primals_43, primals_40, primals_38, primals_36, reinterpret_tensor(buf145, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf49, (128, 32, 1), (32, 1, 32), 0), reinterpret_tensor(buf146, (128, 512, 32), (16384, 1, 512), 0), primals_28, primals_26, primals_24, primals_21, primals_19, primals_17, reinterpret_tensor(buf147, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf15, (128, 32, 1), (32, 1, 32), 0), reinterpret_tensor(buf148, (128, 512, 32), (16384, 1, 512), 0), primals_9, primals_7, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10000, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 1), (1, 1), device='cpu', dtype=torch.int64)
    primals_3 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((16, 512, 32), (16384, 32, 1), device='cpu', dtype=torch.float16)
    primals_12 = rand_strided((16, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float16)
    primals_13 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_14 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_15 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((16, 512, 32), (16384, 32, 1), device='cpu', dtype=torch.float16)
    primals_31 = rand_strided((16, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float16)
    primals_32 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_33 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_34 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((16, 512, 32), (16384, 32, 1), device='cpu', dtype=torch.float16)
    primals_50 = rand_strided((16, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float16)
    primals_51 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_52 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_53 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_54 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((16, 512, 32), (16384, 32, 1), device='cpu', dtype=torch.float16)
    primals_69 = rand_strided((16, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float16)
    primals_70 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_71 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_72 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((10000, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
