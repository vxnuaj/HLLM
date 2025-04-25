# AOT ID: ['1_forward']
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


cpp_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.cpp_pybinding(['float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                            auto tmp1 = 10000LL;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 10000LL), "index out of bounds: 0 <= tmp7 < 10000LL");
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 256LL*tmp5), static_cast<int64_t>(4));
                            tmp9.store(out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(256LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(256LL)))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x2 + 256LL*x1 + 131072LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2 + 256LL*x1), static_cast<int64_t>(4));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp3 = tmp2 * tmp2;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                }
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(16LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(256LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(256LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x2 + 256LL*x1 + 131072LL*x0), static_cast<int64_t>(4));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2 + 256LL*x1), static_cast<int64_t>(4));
                                auto tmp3 = in_out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)];
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x2), static_cast<int64_t>(4));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp2 * tmp4;
                                auto tmp7 = tmp5 * tmp6;
                                tmp7.store(out_ptr2 + static_cast<int64_t>(x2 + 256LL*x1 + 131072LL*x0));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_clone_stack_1 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'half*', 'half*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const float* in_ptr5,
                       half* out_ptr0,
                       half* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(262144LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(262144LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr2[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                            out_ptr3[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr4[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr5[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_2 = async_compile.cpp_pybinding(['float*', 'const bool*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                                {
                                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0.to<int64_t,2>();
                                    auto tmp2 = static_cast<int64_t>(0);
                                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                    auto tmp6 = static_cast<float>(0.17677669529663687);
                                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                    auto tmp8 = tmp5 * tmp7;
                                    auto tmp9 = -std::numeric_limits<float>::infinity();
                                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                    auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp11);
                                }
                            }
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                auto tmp12 = out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)];
                                auto tmp1 = tmp0.to<int64_t,2>();
                                auto tmp2 = static_cast<int64_t>(0);
                                auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                auto tmp6 = static_cast<float>(0.17677669529663687);
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp5 * tmp7;
                                auto tmp9 = -std::numeric_limits<float>::infinity();
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 - tmp13;
                                auto tmp15 = tmp14.exp();
                                tmp15.store(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_3 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 131072LL*x0), static_cast<int64_t>(4));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp4 = tmp0 + tmp3;
                            tmp4.store(in_out_ptr0 + static_cast<int64_t>(x1 + 131072LL*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(256LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = in_out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(4));
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
}
''')


cpp_fused_mul_silu_4 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(5586944LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(5586944LL)))
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
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused__to_copy_clone_stack_6 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'half*', 'half*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const float* in_ptr5,
                       half* out_ptr0,
                       half* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(262144LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(262144LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr2[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                            out_ptr3[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr4[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr5[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_7 = async_compile.cpp_pybinding(['float*', 'const bool*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                                {
                                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0.to<int64_t,2>();
                                    auto tmp2 = static_cast<int64_t>(0);
                                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                    auto tmp6 = static_cast<float>(0.17677669529663687);
                                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                    auto tmp8 = tmp5 * tmp7;
                                    auto tmp9 = -std::numeric_limits<float>::infinity();
                                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                    auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp11);
                                }
                            }
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                auto tmp12 = out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)];
                                auto tmp1 = tmp0.to<int64_t,2>();
                                auto tmp2 = static_cast<int64_t>(0);
                                auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                auto tmp6 = static_cast<float>(0.17677669529663687);
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp5 * tmp7;
                                auto tmp9 = -std::numeric_limits<float>::infinity();
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 - tmp13;
                                auto tmp15 = tmp14.exp();
                                tmp15.store(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_8 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused_mul_silu_9 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(5586944LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(5586944LL)))
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
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused__to_copy_clone_stack_11 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'half*', 'half*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const float* in_ptr5,
                       half* out_ptr0,
                       half* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(262144LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(262144LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr2[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                            out_ptr3[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr4[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr5[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_12 = async_compile.cpp_pybinding(['float*', 'const bool*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                                {
                                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0.to<int64_t,2>();
                                    auto tmp2 = static_cast<int64_t>(0);
                                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                    auto tmp6 = static_cast<float>(0.17677669529663687);
                                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                    auto tmp8 = tmp5 * tmp7;
                                    auto tmp9 = -std::numeric_limits<float>::infinity();
                                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                    auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp11);
                                }
                            }
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                auto tmp12 = out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)];
                                auto tmp1 = tmp0.to<int64_t,2>();
                                auto tmp2 = static_cast<int64_t>(0);
                                auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                auto tmp6 = static_cast<float>(0.17677669529663687);
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp5 * tmp7;
                                auto tmp9 = -std::numeric_limits<float>::infinity();
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 - tmp13;
                                auto tmp15 = tmp14.exp();
                                tmp15.store(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_13 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused_mul_silu_14 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(5586944LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(5586944LL)))
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
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused__to_copy_clone_stack_16 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const half*', 'const half*', 'const float*', 'half*', 'half*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const float* in_ptr5,
                       half* out_ptr0,
                       half* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(262144LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(262144LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(8LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
                    {
                        auto tmp0 = at::vec::VectorizedN<float,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<half,1,float,2>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr2[static_cast<int64_t>(1LL + 2LL*x1 + 32LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr2[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp8;
                            out_ptr3[static_cast<int64_t>(2LL*x1 + 32LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr3[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr4[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr4[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr5[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_17 = async_compile.cpp_pybinding(['float*', 'const bool*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                                {
                                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                    auto tmp1 = tmp0.to<int64_t,2>();
                                    auto tmp2 = static_cast<int64_t>(0);
                                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                    auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                    auto tmp6 = static_cast<float>(0.17677669529663687);
                                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                    auto tmp8 = tmp5 * tmp7;
                                    auto tmp9 = -std::numeric_limits<float>::infinity();
                                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                    auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp11);
                                }
                            }
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)] = static_cast<float>(tmp_acc0);
                    }
                    for(int64_t x2=static_cast<int64_t>(0LL); x2<static_cast<int64_t>(512LL); x2+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x2 + 512LL*x1));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0), static_cast<int64_t>(4));
                                auto tmp12 = out_ptr0[static_cast<int64_t>(x1 + 512LL*x0)];
                                auto tmp1 = tmp0.to<int64_t,2>();
                                auto tmp2 = static_cast<int64_t>(0);
                                auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                auto tmp4 = at::vec::VecMask<int64_t,2>(tmp1 == tmp3);
                                auto tmp6 = static_cast<float>(0.17677669529663687);
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp5 * tmp7;
                                auto tmp9 = -std::numeric_limits<float>::infinity();
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp4.template cast<float,1>());
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 - tmp13;
                                auto tmp15 = tmp14.exp();
                                tmp15.store(in_out_ptr0 + static_cast<int64_t>(x2 + 512LL*x1 + 262144LL*x0));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(65536LL); x0+=static_cast<int64_t>(1LL))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(512LL); x1+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512LL)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0), static_cast<int64_t>(4));
                            auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512LL*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_18 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


cpp_fused_mul_silu_19 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(5586944LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(5586944LL)))
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
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(2097152LL); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2097152LL)))
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
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
        #pragma omp single
        {
            {
                for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(4LL))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192LL)))
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
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8192LL); x0+=static_cast<int64_t>(1LL))
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
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73 = args
    args.clear()
    assert_size_stride(primals_1, (10000, 256), (256, 1))
    assert_size_stride(primals_2, (16, 512), (512, 1))
    assert_size_stride(primals_3, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, 256), (256, 1))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (32, 256), (256, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (256, 256), (256, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (512, 32), (32, 1))
    assert_size_stride(primals_12, (512, 32), (32, 1))
    assert_size_stride(primals_13, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (682, 256), (256, 1))
    assert_size_stride(primals_16, (682, ), (1, ))
    assert_size_stride(primals_17, (682, 256), (256, 1))
    assert_size_stride(primals_18, (682, ), (1, ))
    assert_size_stride(primals_19, (256, 682), (682, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, 256), (256, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (32, 256), (256, 1))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (256, 256), (256, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (512, 32), (32, 1))
    assert_size_stride(primals_29, (512, 32), (32, 1))
    assert_size_stride(primals_30, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (682, 256), (256, 1))
    assert_size_stride(primals_33, (682, ), (1, ))
    assert_size_stride(primals_34, (682, 256), (256, 1))
    assert_size_stride(primals_35, (682, ), (1, ))
    assert_size_stride(primals_36, (256, 682), (682, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, 256), (256, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (32, 256), (256, 1))
    assert_size_stride(primals_42, (32, ), (1, ))
    assert_size_stride(primals_43, (256, 256), (256, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (512, 32), (32, 1))
    assert_size_stride(primals_46, (512, 32), (32, 1))
    assert_size_stride(primals_47, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (682, 256), (256, 1))
    assert_size_stride(primals_50, (682, ), (1, ))
    assert_size_stride(primals_51, (682, 256), (256, 1))
    assert_size_stride(primals_52, (682, ), (1, ))
    assert_size_stride(primals_53, (256, 682), (682, 1))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, 256), (256, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (32, 256), (256, 1))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (256, 256), (256, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (512, 32), (32, 1))
    assert_size_stride(primals_63, (512, 32), (32, 1))
    assert_size_stride(primals_64, (1, 1, 512, 512), (262144, 262144, 512, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (682, 256), (256, 1))
    assert_size_stride(primals_67, (682, ), (1, ))
    assert_size_stride(primals_68, (682, 256), (256, 1))
    assert_size_stride(primals_69, (682, ), (1, ))
    assert_size_stride(primals_70, (256, 682), (682, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (10000, ), (1, ))
    buf0 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    buf1 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf2 = reinterpret_tensor(buf1, (16, 512, 1), (512, 1, 1), 0); del buf1  # reuse
    buf3 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(buf2, primals_2, primals_1, primals_3, primals_4, buf0, buf3)
    buf4 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_5, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf4)
    del primals_6
    buf5 = empty_strided_cpu((8192, 32), (32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [k], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_7, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf5)
    del primals_8
    buf6 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_9, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf6)
    del primals_10
    buf7 = empty_strided_cpu((16, 512, 32), (16384, 32, 1), torch.float16)
    buf8 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float16)
    buf11 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf9 = reinterpret_tensor(buf11, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf10 = reinterpret_tensor(buf11, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf14 = empty_strided_cpu((16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), torch.float32)
    buf12 = reinterpret_tensor(buf14, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf13 = reinterpret_tensor(buf14, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf15 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused__to_copy_clone_stack_1(buf5, buf6, buf4, primals_11, primals_12, buf14, buf7, buf8, buf9, buf10, buf12, buf13, buf15)
    del buf12
    del buf13
    buf16 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf15, (128, 32, 512), (16384, 512, 1), 0), out=buf16)
    buf17 = empty_strided_cpu((16, 8, 512, 1), (4096, 512, 1, 65536), torch.float32)
    buf18 = reinterpret_tensor(buf16, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf16  # reuse
    buf19 = empty_strided_cpu((16, 8, 512, 1), (4096, 512, 1, 65536), torch.float32)
    buf20 = buf18; del buf18  # reuse
    cpp_fused__softmax_div_eq_masked_fill_2(buf20, primals_13, buf17, buf19)
    buf21 = reinterpret_tensor(buf4, (128, 512, 32), (16384, 32, 1), 0); del buf4  # reuse
    # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf20, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf6, (128, 512, 32), (16384, 32, 1), 0), out=buf21)
    buf22 = reinterpret_tensor(buf21, (16, 512, 256), (131072, 256, 1), 0); del buf21  # reuse
    buf23 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf24 = reinterpret_tensor(buf23, (16, 512, 1), (512, 1, 1), 0); del buf23  # reuse
    buf25 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_3(buf22, buf24, buf0, primals_3, primals_14, buf25)
    buf26 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, reinterpret_tensor(buf25, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_15, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf26)
    del primals_16
    buf27 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, reinterpret_tensor(buf25, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_17, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf27)
    del primals_18
    buf28 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_4(buf26, buf27, buf28)
    buf29 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, reinterpret_tensor(buf28, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_19, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf29)
    del primals_20
    buf30 = reinterpret_tensor(buf29, (16, 512, 256), (131072, 256, 1), 0); del buf29  # reuse
    buf31 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf32 = reinterpret_tensor(buf31, (16, 512, 1), (512, 1, 1), 0); del buf31  # reuse
    buf33 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_5(buf30, buf32, buf25, primals_21, buf33)
    buf34 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, reinterpret_tensor(buf33, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_22, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf34)
    del primals_23
    buf35 = buf5; del buf5  # reuse
    # Topologically Sorted Source Nodes: [k_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_25, reinterpret_tensor(buf33, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_24, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf35)
    del primals_25
    buf36 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, reinterpret_tensor(buf33, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_26, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf36)
    del primals_27
    buf37 = empty_strided_cpu((16, 512, 32), (16384, 32, 1), torch.float16)
    buf38 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float16)
    buf41 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf39 = reinterpret_tensor(buf41, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf40 = reinterpret_tensor(buf41, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf44 = buf14; del buf14  # reuse
    buf42 = reinterpret_tensor(buf44, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf43 = reinterpret_tensor(buf44, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf45 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused__to_copy_clone_stack_6(buf35, buf36, buf34, primals_28, primals_29, buf44, buf37, buf38, buf39, buf40, buf42, buf43, buf45)
    del buf42
    del buf43
    buf46 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf41, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf45, (128, 32, 512), (16384, 512, 1), 0), out=buf46)
    buf47 = buf19; del buf19  # reuse
    buf48 = reinterpret_tensor(buf46, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf46  # reuse
    buf49 = buf17; del buf17  # reuse
    buf50 = buf48; del buf48  # reuse
    cpp_fused__softmax_div_eq_masked_fill_7(buf50, primals_30, buf47, buf49)
    buf51 = reinterpret_tensor(buf34, (128, 512, 32), (16384, 32, 1), 0); del buf34  # reuse
    # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf50, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf36, (128, 512, 32), (16384, 32, 1), 0), out=buf51)
    buf52 = reinterpret_tensor(buf51, (16, 512, 256), (131072, 256, 1), 0); del buf51  # reuse
    buf53 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf54 = reinterpret_tensor(buf53, (16, 512, 1), (512, 1, 1), 0); del buf53  # reuse
    buf55 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_8(buf52, buf54, buf30, primals_31, buf55)
    buf56 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_33, reinterpret_tensor(buf55, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_32, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf56)
    del primals_33
    buf57 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_35, reinterpret_tensor(buf55, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_34, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf57)
    del primals_35
    buf58 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_9(buf56, buf57, buf58)
    buf59 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, reinterpret_tensor(buf58, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_36, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf59)
    del primals_37
    buf60 = reinterpret_tensor(buf59, (16, 512, 256), (131072, 256, 1), 0); del buf59  # reuse
    buf61 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf62 = reinterpret_tensor(buf61, (16, 512, 1), (512, 1, 1), 0); del buf61  # reuse
    buf63 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_10(buf60, buf62, buf55, primals_38, buf63)
    buf64 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, reinterpret_tensor(buf63, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf64)
    del primals_40
    buf65 = reinterpret_tensor(buf44, (8192, 32), (32, 1), 0); del buf44  # reuse
    # Topologically Sorted Source Nodes: [k_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, reinterpret_tensor(buf63, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_41, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf65)
    del primals_42
    buf66 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, reinterpret_tensor(buf63, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_43, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf66)
    del primals_44
    buf67 = empty_strided_cpu((16, 512, 32), (16384, 32, 1), torch.float16)
    buf68 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float16)
    buf71 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf69 = reinterpret_tensor(buf71, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf70 = reinterpret_tensor(buf71, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf74 = reinterpret_tensor(buf35, (16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), 0); del buf35  # reuse
    buf72 = reinterpret_tensor(buf74, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf73 = reinterpret_tensor(buf74, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf75 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused__to_copy_clone_stack_11(buf65, buf66, buf64, primals_45, primals_46, buf74, buf67, buf68, buf69, buf70, buf72, buf73, buf75)
    del buf72
    del buf73
    buf76 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf71, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf75, (128, 32, 512), (16384, 512, 1), 0), out=buf76)
    buf77 = buf49; del buf49  # reuse
    buf78 = reinterpret_tensor(buf76, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf76  # reuse
    buf79 = buf47; del buf47  # reuse
    buf80 = buf78; del buf78  # reuse
    cpp_fused__softmax_div_eq_masked_fill_12(buf80, primals_47, buf77, buf79)
    buf81 = reinterpret_tensor(buf64, (128, 512, 32), (16384, 32, 1), 0); del buf64  # reuse
    # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf80, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf66, (128, 512, 32), (16384, 32, 1), 0), out=buf81)
    buf82 = reinterpret_tensor(buf81, (16, 512, 256), (131072, 256, 1), 0); del buf81  # reuse
    buf83 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf84 = reinterpret_tensor(buf83, (16, 512, 1), (512, 1, 1), 0); del buf83  # reuse
    buf85 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_13(buf82, buf84, buf60, primals_48, buf85)
    buf86 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, reinterpret_tensor(buf85, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_49, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf86)
    del primals_50
    buf87 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, reinterpret_tensor(buf85, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_51, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf87)
    del primals_52
    buf88 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_14(buf86, buf87, buf88)
    buf89 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, reinterpret_tensor(buf88, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_53, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf89)
    del primals_54
    buf90 = reinterpret_tensor(buf89, (16, 512, 256), (131072, 256, 1), 0); del buf89  # reuse
    buf91 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf92 = reinterpret_tensor(buf91, (16, 512, 1), (512, 1, 1), 0); del buf91  # reuse
    buf93 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_15(buf90, buf92, buf85, primals_55, buf93)
    buf94 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, reinterpret_tensor(buf93, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_56, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf94)
    del primals_57
    buf95 = reinterpret_tensor(buf74, (8192, 32), (32, 1), 0); del buf74  # reuse
    # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, reinterpret_tensor(buf93, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_58, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf95)
    del primals_59
    buf96 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_61, reinterpret_tensor(buf93, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_60, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf96)
    del primals_61
    buf97 = empty_strided_cpu((16, 512, 32), (16384, 32, 1), torch.float16)
    buf98 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float16)
    buf101 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf99 = reinterpret_tensor(buf101, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf100 = reinterpret_tensor(buf101, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf104 = reinterpret_tensor(buf65, (16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), 0); del buf65  # reuse
    buf102 = reinterpret_tensor(buf104, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf103 = reinterpret_tensor(buf104, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf105 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused__to_copy_clone_stack_16(buf95, buf96, buf94, primals_62, primals_63, buf104, buf97, buf98, buf99, buf100, buf102, buf103, buf105)
    del buf102
    del buf103
    del buf104
    del buf95
    buf106 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf105, (128, 32, 512), (16384, 512, 1), 0), out=buf106)
    buf107 = buf79; del buf79  # reuse
    buf108 = reinterpret_tensor(buf106, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf106  # reuse
    buf109 = buf77; del buf77  # reuse
    buf110 = buf108; del buf108  # reuse
    cpp_fused__softmax_div_eq_masked_fill_17(buf110, primals_64, buf107, buf109)
    del buf107
    del buf109
    buf111 = reinterpret_tensor(buf94, (128, 512, 32), (16384, 32, 1), 0); del buf94  # reuse
    # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf96, (128, 512, 32), (16384, 32, 1), 0), out=buf111)
    buf112 = reinterpret_tensor(buf111, (16, 512, 256), (131072, 256, 1), 0); del buf111  # reuse
    buf113 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf114 = reinterpret_tensor(buf113, (16, 512, 1), (512, 1, 1), 0); del buf113  # reuse
    buf115 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_18(buf112, buf114, buf90, primals_65, buf115)
    buf116 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, reinterpret_tensor(buf115, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_66, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf116)
    del primals_67
    buf117 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_69, reinterpret_tensor(buf115, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_68, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf117)
    del primals_69
    buf118 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_19(buf116, buf117, buf118)
    buf119 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_71, reinterpret_tensor(buf118, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_70, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf119)
    del primals_71
    buf120 = reinterpret_tensor(buf119, (16, 512, 256), (131072, 256, 1), 0); del buf119  # reuse
    buf121 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf122 = reinterpret_tensor(buf121, (16, 512, 1), (512, 1, 1), 0); del buf121  # reuse
    buf123 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_20(buf120, buf122, buf115, primals_72, buf123)
    buf124 = empty_strided_cpu((8192, 10000), (10000, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_73, reinterpret_tensor(buf123, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_1, (256, 10000), (1, 256), 0), alpha=1, beta=1, out=buf124)
    del primals_73
    return (reinterpret_tensor(buf124, (16, 512, 10000), (5120000, 10000, 1), 0), buf8, buf7, buf38, buf37, buf68, buf67, buf98, buf97, primals_2, primals_3, primals_4, primals_13, primals_14, primals_21, primals_30, primals_31, primals_38, primals_47, primals_48, primals_55, primals_64, primals_65, primals_72, buf0, buf2, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_11, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_12, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_11, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_12, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf20, buf22, buf24, reinterpret_tensor(buf25, (8192, 256), (256, 1), 0), buf26, buf27, reinterpret_tensor(buf28, (8192, 682), (682, 1), 0), buf30, buf32, reinterpret_tensor(buf33, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_28, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_29, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_28, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_29, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf50, buf52, buf54, reinterpret_tensor(buf55, (8192, 256), (256, 1), 0), buf56, buf57, reinterpret_tensor(buf58, (8192, 682), (682, 1), 0), buf60, buf62, reinterpret_tensor(buf63, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_45, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_46, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_45, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_46, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf80, buf82, buf84, reinterpret_tensor(buf85, (8192, 256), (256, 1), 0), buf86, buf87, reinterpret_tensor(buf88, (8192, 682), (682, 1), 0), buf90, buf92, reinterpret_tensor(buf93, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_62, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_63, (1, 1, 1, 16), (32, 32, 32, 2), 0), reinterpret_tensor(primals_62, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_63, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf110, buf112, buf114, reinterpret_tensor(buf115, (8192, 256), (256, 1), 0), buf116, buf117, reinterpret_tensor(buf118, (8192, 682), (682, 1), 0), buf120, buf122, reinterpret_tensor(buf123, (8192, 256), (256, 1), 0), primals_1, primals_70, primals_68, primals_66, reinterpret_tensor(buf96, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf101, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf105, (128, 512, 32), (16384, 1, 512), 0), primals_60, primals_58, primals_56, primals_53, primals_51, primals_49, reinterpret_tensor(buf66, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf71, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf75, (128, 512, 32), (16384, 1, 512), 0), primals_43, primals_41, primals_39, primals_36, primals_34, primals_32, reinterpret_tensor(buf36, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf41, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf45, (128, 512, 32), (16384, 1, 512), 0), primals_26, primals_24, primals_22, primals_19, primals_17, primals_15, reinterpret_tensor(buf6, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf11, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf15, (128, 512, 32), (16384, 1, 512), 0), primals_9, primals_7, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10000, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_3 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_12 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_13 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_29 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_30 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_46 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_47 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((32, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_63 = rand_strided((512, 32), (32, 1), device='cpu', dtype=torch.float16)
    primals_64 = rand_strided((1, 1, 512, 512), (262144, 262144, 512, 1), device='cpu', dtype=torch.bool)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((682, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((682, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, 682), (682, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((10000, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
