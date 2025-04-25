# AOT ID: ['0_forward']
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


cpp_fused_clone_stack_1 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr3[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr3[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
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
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr4 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
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


cpp_fused_clone_stack_6 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr3[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr3[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
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
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr4 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
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


cpp_fused_clone_stack_11 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr3[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr3[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
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
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr4 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
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


cpp_fused_clone_stack_16 = async_compile.cpp_pybinding(['const float*', 'const half*', 'const half*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/var/folders/48/l60ld_dj1kggcvhp0rsys4qm0000gn/T/torchinductor_juanvera/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(128LL); x0+=static_cast<int64_t>(1LL))
            {
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(8192LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr0[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp8 = decltype(tmp3)(tmp3 - tmp7);
                            auto tmp9 = decltype(tmp0)(tmp0 * tmp6);
                            auto tmp10 = decltype(tmp4)(tmp4 * tmp2);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            out_ptr0[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp8;
                            out_ptr1[static_cast<int64_t>(2LL*x1 + 16384LL*x0)] = tmp11;
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
                            auto tmp0 = in_ptr3[static_cast<int64_t>(2LL*x1 + 16384LL*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(2LL*x1)];
                            auto tmp4 = in_ptr3[static_cast<int64_t>(1LL + 2LL*x1 + 16384LL*x0)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(2LL*x1)];
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
                                    at::vec::transpose_mxn<float,static_cast<int64_t>(4),static_cast<int64_t>(4)>(in_ptr4 + static_cast<int64_t>(x2 + 32LL*x3 + 16384LL*x0), static_cast<int64_t>(32LL), tmp0, static_cast<int64_t>(4));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(4); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(4LL*x2_inner), static_cast<int64_t>(4));
                                        tmp1.store(out_ptr4 + static_cast<int64_t>(x3 + 512LL*x2 + 512LL*x2_inner + 16384LL*x1 + 131072LL*x0));
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
    buf9 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf7 = reinterpret_tensor(buf9, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf8 = reinterpret_tensor(buf9, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf12 = empty_strided_cpu((16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), torch.float32)
    buf10 = reinterpret_tensor(buf12, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf11 = reinterpret_tensor(buf12, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf13 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused_clone_stack_1(buf4, primals_11, primals_12, buf5, buf12, buf7, buf8, buf10, buf11, buf13)
    del buf10
    del buf11
    buf14 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf9, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf13, (128, 32, 512), (16384, 512, 1), 0), out=buf14)
    buf15 = empty_strided_cpu((16, 8, 512, 1), (4096, 512, 1, 65536), torch.float32)
    buf16 = reinterpret_tensor(buf14, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf14  # reuse
    buf17 = empty_strided_cpu((16, 8, 512, 1), (4096, 512, 1, 65536), torch.float32)
    buf18 = buf16; del buf16  # reuse
    cpp_fused__softmax_div_eq_masked_fill_2(buf18, primals_13, buf15, buf17)
    buf19 = reinterpret_tensor(buf4, (128, 512, 32), (16384, 32, 1), 0); del buf4  # reuse
    # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf18, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf6, (128, 512, 32), (16384, 32, 1), 0), out=buf19)
    buf20 = reinterpret_tensor(buf19, (16, 512, 256), (131072, 256, 1), 0); del buf19  # reuse
    buf21 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf22 = reinterpret_tensor(buf21, (16, 512, 1), (512, 1, 1), 0); del buf21  # reuse
    buf23 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_3(buf20, buf22, buf0, primals_3, primals_14, buf23)
    buf24 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, reinterpret_tensor(buf23, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_15, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf24)
    del primals_16
    buf25 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, reinterpret_tensor(buf23, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_17, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf25)
    del primals_18
    buf26 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_4(buf24, buf25, buf26)
    buf27 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, reinterpret_tensor(buf26, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_19, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf27)
    del primals_20
    buf28 = reinterpret_tensor(buf27, (16, 512, 256), (131072, 256, 1), 0); del buf27  # reuse
    buf29 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf30 = reinterpret_tensor(buf29, (16, 512, 1), (512, 1, 1), 0); del buf29  # reuse
    buf31 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_5(buf28, buf30, buf23, primals_21, buf31)
    buf32 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, reinterpret_tensor(buf31, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_22, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf32)
    del primals_23
    buf33 = buf5; del buf5  # reuse
    # Topologically Sorted Source Nodes: [k_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_25, reinterpret_tensor(buf31, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_24, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf33)
    del primals_25
    buf34 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, reinterpret_tensor(buf31, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_26, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf34)
    del primals_27
    buf37 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf35 = reinterpret_tensor(buf37, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf36 = reinterpret_tensor(buf37, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf40 = buf12; del buf12  # reuse
    buf38 = reinterpret_tensor(buf40, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf39 = reinterpret_tensor(buf40, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf41 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused_clone_stack_6(buf32, primals_28, primals_29, buf33, buf40, buf35, buf36, buf38, buf39, buf41)
    del buf38
    del buf39
    buf42 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf37, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf41, (128, 32, 512), (16384, 512, 1), 0), out=buf42)
    buf43 = buf17; del buf17  # reuse
    buf44 = reinterpret_tensor(buf42, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf42  # reuse
    buf45 = buf15; del buf15  # reuse
    buf46 = buf44; del buf44  # reuse
    cpp_fused__softmax_div_eq_masked_fill_7(buf46, primals_30, buf43, buf45)
    buf47 = reinterpret_tensor(buf32, (128, 512, 32), (16384, 32, 1), 0); del buf32  # reuse
    # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf46, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf34, (128, 512, 32), (16384, 32, 1), 0), out=buf47)
    buf48 = reinterpret_tensor(buf47, (16, 512, 256), (131072, 256, 1), 0); del buf47  # reuse
    buf49 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf50 = reinterpret_tensor(buf49, (16, 512, 1), (512, 1, 1), 0); del buf49  # reuse
    buf51 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_8(buf48, buf50, buf28, primals_31, buf51)
    buf52 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_33, reinterpret_tensor(buf51, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_32, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf52)
    del primals_33
    buf53 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_35, reinterpret_tensor(buf51, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_34, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf53)
    del primals_35
    buf54 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_9(buf52, buf53, buf54)
    buf55 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, reinterpret_tensor(buf54, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_36, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf55)
    del primals_37
    buf56 = reinterpret_tensor(buf55, (16, 512, 256), (131072, 256, 1), 0); del buf55  # reuse
    buf57 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf58 = reinterpret_tensor(buf57, (16, 512, 1), (512, 1, 1), 0); del buf57  # reuse
    buf59 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_10(buf56, buf58, buf51, primals_38, buf59)
    buf60 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, reinterpret_tensor(buf59, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf60)
    del primals_40
    buf61 = reinterpret_tensor(buf40, (8192, 32), (32, 1), 0); del buf40  # reuse
    # Topologically Sorted Source Nodes: [k_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_42, reinterpret_tensor(buf59, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_41, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf61)
    del primals_42
    buf62 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, reinterpret_tensor(buf59, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_43, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf62)
    del primals_44
    buf65 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf63 = reinterpret_tensor(buf65, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf64 = reinterpret_tensor(buf65, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf68 = reinterpret_tensor(buf33, (16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), 0); del buf33  # reuse
    buf66 = reinterpret_tensor(buf68, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf67 = reinterpret_tensor(buf68, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf69 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused_clone_stack_11(buf60, primals_45, primals_46, buf61, buf68, buf63, buf64, buf66, buf67, buf69)
    del buf66
    del buf67
    buf70 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf69, (128, 32, 512), (16384, 512, 1), 0), out=buf70)
    buf71 = buf45; del buf45  # reuse
    buf72 = reinterpret_tensor(buf70, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf70  # reuse
    buf73 = buf43; del buf43  # reuse
    buf74 = buf72; del buf72  # reuse
    cpp_fused__softmax_div_eq_masked_fill_12(buf74, primals_47, buf71, buf73)
    buf75 = reinterpret_tensor(buf60, (128, 512, 32), (16384, 32, 1), 0); del buf60  # reuse
    # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf62, (128, 512, 32), (16384, 32, 1), 0), out=buf75)
    buf76 = reinterpret_tensor(buf75, (16, 512, 256), (131072, 256, 1), 0); del buf75  # reuse
    buf77 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf78 = reinterpret_tensor(buf77, (16, 512, 1), (512, 1, 1), 0); del buf77  # reuse
    buf79 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_13(buf76, buf78, buf56, primals_48, buf79)
    buf80 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, reinterpret_tensor(buf79, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_49, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf80)
    del primals_50
    buf81 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, reinterpret_tensor(buf79, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_51, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf81)
    del primals_52
    buf82 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_14(buf80, buf81, buf82)
    buf83 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, reinterpret_tensor(buf82, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_53, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf83)
    del primals_54
    buf84 = reinterpret_tensor(buf83, (16, 512, 256), (131072, 256, 1), 0); del buf83  # reuse
    buf85 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf86 = reinterpret_tensor(buf85, (16, 512, 1), (512, 1, 1), 0); del buf85  # reuse
    buf87 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_15(buf84, buf86, buf79, primals_55, buf87)
    buf88 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, reinterpret_tensor(buf87, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_56, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf88)
    del primals_57
    buf89 = reinterpret_tensor(buf68, (8192, 32), (32, 1), 0); del buf68  # reuse
    # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, reinterpret_tensor(buf87, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_58, (256, 32), (1, 256), 0), alpha=1, beta=1, out=buf89)
    del primals_59
    buf90 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_61, reinterpret_tensor(buf87, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_60, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf90)
    del primals_61
    buf93 = empty_strided_cpu((16, 8, 512, 16, 2), (131072, 16384, 32, 2, 1), torch.float32)
    buf91 = reinterpret_tensor(buf93, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 0)  # alias
    buf92 = reinterpret_tensor(buf93, (16, 8, 512, 16, 1), (131072, 16384, 32, 2, 1), 1)  # alias
    buf96 = reinterpret_tensor(buf61, (16, 1, 512, 16, 2), (16384, 1, 32, 2, 1), 0); del buf61  # reuse
    buf94 = reinterpret_tensor(buf96, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 0)  # alias
    buf95 = reinterpret_tensor(buf96, (16, 1, 512, 16, 1), (16384, 1, 32, 2, 1), 1)  # alias
    buf97 = empty_strided_cpu((16, 8, 32, 512), (131072, 16384, 512, 1), torch.float32)
    cpp_fused_clone_stack_16(buf88, primals_62, primals_63, buf89, buf96, buf91, buf92, buf94, buf95, buf97)
    del buf89
    del buf94
    del buf95
    del buf96
    buf98 = empty_strided_cpu((128, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf93, (128, 512, 32), (16384, 32, 1), 0), reinterpret_tensor(buf97, (128, 32, 512), (16384, 512, 1), 0), out=buf98)
    buf99 = buf73; del buf73  # reuse
    buf100 = reinterpret_tensor(buf98, (16, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf98  # reuse
    buf101 = buf71; del buf71  # reuse
    buf102 = buf100; del buf100  # reuse
    cpp_fused__softmax_div_eq_masked_fill_17(buf102, primals_64, buf99, buf101)
    del buf101
    del buf99
    buf103 = reinterpret_tensor(buf88, (128, 512, 32), (16384, 32, 1), 0); del buf88  # reuse
    # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf102, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf90, (128, 512, 32), (16384, 32, 1), 0), out=buf103)
    buf104 = reinterpret_tensor(buf103, (16, 512, 256), (131072, 256, 1), 0); del buf103  # reuse
    buf105 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf106 = reinterpret_tensor(buf105, (16, 512, 1), (512, 1, 1), 0); del buf105  # reuse
    buf107 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_18(buf104, buf106, buf84, primals_65, buf107)
    buf108 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, reinterpret_tensor(buf107, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_66, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf108)
    del primals_67
    buf109 = empty_strided_cpu((8192, 682), (682, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_69, reinterpret_tensor(buf107, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_68, (256, 682), (1, 256), 0), alpha=1, beta=1, out=buf109)
    del primals_69
    buf110 = empty_strided_cpu((16, 512, 682), (349184, 682, 1), torch.float32)
    cpp_fused_mul_silu_19(buf108, buf109, buf110)
    buf111 = empty_strided_cpu((8192, 256), (256, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_71, reinterpret_tensor(buf110, (8192, 682), (682, 1), 0), reinterpret_tensor(primals_70, (682, 256), (1, 682), 0), alpha=1, beta=1, out=buf111)
    del primals_71
    buf112 = reinterpret_tensor(buf111, (16, 512, 256), (131072, 256, 1), 0); del buf111  # reuse
    buf113 = empty_strided_cpu((16, 512, 1), (512, 1, 8192), torch.float32)
    buf114 = reinterpret_tensor(buf113, (16, 512, 1), (512, 1, 1), 0); del buf113  # reuse
    buf115 = empty_strided_cpu((16, 512, 256), (131072, 256, 1), torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_20(buf112, buf114, buf107, primals_72, buf115)
    buf116 = empty_strided_cpu((8192, 10000), (10000, 1), torch.float32)
    # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_73, reinterpret_tensor(buf115, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_1, (256, 10000), (1, 256), 0), alpha=1, beta=1, out=buf116)
    del primals_73
    return (reinterpret_tensor(buf116, (16, 512, 10000), (5120000, 10000, 1), 0), primals_2, primals_3, primals_4, primals_13, primals_14, primals_21, primals_30, primals_31, primals_38, primals_47, primals_48, primals_55, primals_64, primals_65, primals_72, buf0, buf2, reinterpret_tensor(buf3, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_11, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_12, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf18, buf20, buf22, reinterpret_tensor(buf23, (8192, 256), (256, 1), 0), buf24, buf25, reinterpret_tensor(buf26, (8192, 682), (682, 1), 0), buf28, buf30, reinterpret_tensor(buf31, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_28, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_29, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf46, buf48, buf50, reinterpret_tensor(buf51, (8192, 256), (256, 1), 0), buf52, buf53, reinterpret_tensor(buf54, (8192, 682), (682, 1), 0), buf56, buf58, reinterpret_tensor(buf59, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_45, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_46, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf74, buf76, buf78, reinterpret_tensor(buf79, (8192, 256), (256, 1), 0), buf80, buf81, reinterpret_tensor(buf82, (8192, 682), (682, 1), 0), buf84, buf86, reinterpret_tensor(buf87, (8192, 256), (256, 1), 0), reinterpret_tensor(primals_62, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), reinterpret_tensor(primals_63, (1, 1, 512, 16), (16384, 16384, 32, 2), 0), buf102, buf104, buf106, reinterpret_tensor(buf107, (8192, 256), (256, 1), 0), buf108, buf109, reinterpret_tensor(buf110, (8192, 682), (682, 1), 0), buf112, buf114, reinterpret_tensor(buf115, (8192, 256), (256, 1), 0), primals_1, primals_70, primals_68, primals_66, reinterpret_tensor(buf90, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf93, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf97, (128, 512, 32), (16384, 1, 512), 0), primals_60, primals_58, primals_56, primals_53, primals_51, primals_49, reinterpret_tensor(buf62, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf65, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf69, (128, 512, 32), (16384, 1, 512), 0), primals_43, primals_41, primals_39, primals_36, primals_34, primals_32, reinterpret_tensor(buf34, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf37, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf41, (128, 512, 32), (16384, 1, 512), 0), primals_26, primals_24, primals_22, primals_19, primals_17, primals_15, reinterpret_tensor(buf6, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf9, (128, 32, 512), (16384, 1, 32), 0), reinterpret_tensor(buf13, (128, 512, 32), (16384, 1, 512), 0), primals_9, primals_7, primals_5, )


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
