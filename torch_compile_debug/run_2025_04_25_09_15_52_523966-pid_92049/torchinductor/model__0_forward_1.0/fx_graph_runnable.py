
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'constant_functions', 'repro_level', 'skipfiles_inline_module_allowlist', 'repro_after'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0
# torch cuda version: None
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73):
        embedding = torch.ops.aten.embedding.default(primals_1, primals_2)
        add = torch.ops.aten.add.Tensor(embedding, primals_3)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(add, rsqrt)
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        view = torch.ops.aten.view.default(mul_1, [8192, 256]);  mul_1 = None
        permute = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = None
        view_1 = torch.ops.aten.view.default(addmm, [16, 512, 256]);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_8, view, permute_1);  primals_8 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [16, 512, 32]);  addmm_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = None
        view_5 = torch.ops.aten.view.default(addmm_2, [16, 512, 256]);  addmm_2 = None
        view_6 = torch.ops.aten.view.default(view_1, [16, 8, 512, 32]);  view_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        view_7 = torch.ops.aten.view.default(view_5, [16, 8, 512, 32]);  view_5 = None
        slice_3 = torch.ops.aten.slice.Tensor(primals_11, 0, 0, 512);  primals_11 = None
        slice_4 = torch.ops.aten.slice.Tensor(primals_12, 0, 0, 512);  primals_12 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(slice_3, 0);  slice_3 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 0);  unsqueeze_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(slice_4, 0);  slice_4 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 0);  unsqueeze_3 = None
        slice_5 = torch.ops.aten.slice.Tensor(view_6, 3, 0, 9223372036854775807, 2)
        slice_6 = torch.ops.aten.slice.Tensor(view_6, 3, 1, 9223372036854775807, 2);  view_6 = None
        slice_7 = torch.ops.aten.slice.Tensor(unsqueeze_2, 3, 0, 9223372036854775807, 2);  unsqueeze_2 = None
        slice_8 = torch.ops.aten.slice.Tensor(unsqueeze_4, 3, 0, 9223372036854775807, 2);  unsqueeze_4 = None
        mul_2 = torch.ops.aten.mul.Tensor(slice_5, slice_7)
        mul_3 = torch.ops.aten.mul.Tensor(slice_6, slice_8)
        sub = torch.ops.aten.sub.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(slice_5, slice_8);  slice_5 = None
        mul_5 = torch.ops.aten.mul.Tensor(slice_6, slice_7);  slice_6 = None
        add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(sub, 4);  sub = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_2, 4);  add_2 = None
        cat = torch.ops.aten.cat.default([unsqueeze_5, unsqueeze_6], -1);  unsqueeze_5 = unsqueeze_6 = None
        view_8 = torch.ops.aten.view.default(cat, [16, 8, 512, 32]);  cat = None
        slice_11 = torch.ops.aten.slice.Tensor(unsqueeze, 3, 0, 9223372036854775807, 2)
        slice_12 = torch.ops.aten.slice.Tensor(unsqueeze, 3, 1, 9223372036854775807, 2);  unsqueeze = None
        mul_6 = torch.ops.aten.mul.Tensor(slice_11, slice_7)
        mul_7 = torch.ops.aten.mul.Tensor(slice_12, slice_8)
        sub_1 = torch.ops.aten.sub.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(slice_11, slice_8);  slice_11 = None
        mul_9 = torch.ops.aten.mul.Tensor(slice_12, slice_7);  slice_12 = None
        add_3 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(sub_1, 4);  sub_1 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(add_3, 4);  add_3 = None
        cat_1 = torch.ops.aten.cat.default([unsqueeze_11, unsqueeze_12], -1);  unsqueeze_11 = unsqueeze_12 = None
        view_9 = torch.ops.aten.view.default(cat_1, [16, 1, 512, 32]);  cat_1 = None
        permute_3 = torch.ops.aten.permute.default(view_9, [0, 1, 3, 2]);  view_9 = None
        expand = torch.ops.aten.expand.default(view_8, [16, 8, 512, 32]);  view_8 = None
        view_10 = torch.ops.aten.view.default(expand, [128, 512, 32]);  expand = None
        expand_1 = torch.ops.aten.expand.default(permute_3, [16, 8, 32, 512]);  permute_3 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11 = torch.ops.aten.view.default(clone_1, [128, 32, 512]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view_10, view_11)
        view_12 = torch.ops.aten.view.default(bmm, [16, 8, 512, 512]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_12, 5.656854249492381);  view_12 = None
        eq = torch.ops.aten.eq.Scalar(primals_13, 0)
        full_default = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(eq, full_default, div);  eq = div = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2 = torch.ops.aten.expand.default(div_1, [16, 8, 512, 512])
        view_13 = torch.ops.aten.view.default(expand_2, [128, 512, 512]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(view_7, [16, 8, 512, 32]);  view_7 = None
        view_14 = torch.ops.aten.view.default(expand_3, [128, 512, 32]);  expand_3 = None
        bmm_1 = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = None
        view_15 = torch.ops.aten.view.default(bmm_1, [16, 8, 512, 32]);  bmm_1 = None
        view_16 = torch.ops.aten.view.default(view_15, [16, -1, 256]);  view_15 = None
        add_4 = torch.ops.aten.add.Tensor(view_16, add);  view_16 = add = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_4, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_5 = torch.ops.aten.add.Scalar(mean_1, 1.1920928955078125e-07);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(add_4, rsqrt_1)
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, primals_14);  mul_10 = None
        view_17 = torch.ops.aten.view.default(mul_11, [8192, 256])
        permute_4 = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_16, view_17, permute_4);  primals_16 = None
        view_18 = torch.ops.aten.view.default(addmm_3, [16, 512, 682])
        sigmoid = torch.ops.aten.sigmoid.default(view_18)
        mul_12 = torch.ops.aten.mul.Tensor(view_18, sigmoid);  view_18 = sigmoid = None
        permute_5 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        addmm_4 = torch.ops.aten.addmm.default(primals_18, view_17, permute_5);  primals_18 = None
        view_20 = torch.ops.aten.view.default(addmm_4, [16, 512, 682])
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, view_20);  mul_12 = view_20 = None
        view_21 = torch.ops.aten.view.default(mul_13, [8192, 682]);  mul_13 = None
        permute_6 = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_20, view_21, permute_6);  primals_20 = None
        view_22 = torch.ops.aten.view.default(addmm_5, [16, 512, 256]);  addmm_5 = None
        add_6 = torch.ops.aten.add.Tensor(view_22, mul_11);  view_22 = mul_11 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_7 = torch.ops.aten.add.Scalar(mean_2, 1.1920928955078125e-07);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_6, rsqrt_2)
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, primals_21);  mul_14 = None
        view_23 = torch.ops.aten.view.default(mul_15, [8192, 256]);  mul_15 = None
        permute_7 = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        addmm_6 = torch.ops.aten.addmm.default(primals_23, view_23, permute_7);  primals_23 = None
        view_24 = torch.ops.aten.view.default(addmm_6, [16, 512, 256]);  addmm_6 = None
        permute_8 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_25, view_23, permute_8);  primals_25 = None
        view_26 = torch.ops.aten.view.default(addmm_7, [16, 512, 32]);  addmm_7 = None
        permute_9 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_27, view_23, permute_9);  primals_27 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [16, 512, 256]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_24, [16, 8, 512, 32]);  view_24 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_26, 1);  view_26 = None
        view_30 = torch.ops.aten.view.default(view_28, [16, 8, 512, 32]);  view_28 = None
        slice_17 = torch.ops.aten.slice.Tensor(primals_28, 0, 0, 512);  primals_28 = None
        slice_18 = torch.ops.aten.slice.Tensor(primals_29, 0, 0, 512);  primals_29 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(slice_17, 0);  slice_17 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(slice_18, 0);  slice_18 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 0);  unsqueeze_16 = None
        slice_19 = torch.ops.aten.slice.Tensor(view_29, 3, 0, 9223372036854775807, 2)
        slice_20 = torch.ops.aten.slice.Tensor(view_29, 3, 1, 9223372036854775807, 2);  view_29 = None
        slice_21 = torch.ops.aten.slice.Tensor(unsqueeze_15, 3, 0, 9223372036854775807, 2);  unsqueeze_15 = None
        slice_22 = torch.ops.aten.slice.Tensor(unsqueeze_17, 3, 0, 9223372036854775807, 2);  unsqueeze_17 = None
        mul_16 = torch.ops.aten.mul.Tensor(slice_19, slice_21)
        mul_17 = torch.ops.aten.mul.Tensor(slice_20, slice_22)
        sub_3 = torch.ops.aten.sub.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(slice_19, slice_22);  slice_19 = None
        mul_19 = torch.ops.aten.mul.Tensor(slice_20, slice_21);  slice_20 = None
        add_8 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(sub_3, 4);  sub_3 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(add_8, 4);  add_8 = None
        cat_2 = torch.ops.aten.cat.default([unsqueeze_18, unsqueeze_19], -1);  unsqueeze_18 = unsqueeze_19 = None
        view_31 = torch.ops.aten.view.default(cat_2, [16, 8, 512, 32]);  cat_2 = None
        slice_25 = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807, 2)
        slice_26 = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 1, 9223372036854775807, 2);  unsqueeze_13 = None
        mul_20 = torch.ops.aten.mul.Tensor(slice_25, slice_21)
        mul_21 = torch.ops.aten.mul.Tensor(slice_26, slice_22)
        sub_4 = torch.ops.aten.sub.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(slice_25, slice_22);  slice_25 = None
        mul_23 = torch.ops.aten.mul.Tensor(slice_26, slice_21);  slice_26 = None
        add_9 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(sub_4, 4);  sub_4 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(add_9, 4);  add_9 = None
        cat_3 = torch.ops.aten.cat.default([unsqueeze_24, unsqueeze_25], -1);  unsqueeze_24 = unsqueeze_25 = None
        view_32 = torch.ops.aten.view.default(cat_3, [16, 1, 512, 32]);  cat_3 = None
        permute_10 = torch.ops.aten.permute.default(view_32, [0, 1, 3, 2]);  view_32 = None
        expand_4 = torch.ops.aten.expand.default(view_31, [16, 8, 512, 32]);  view_31 = None
        view_33 = torch.ops.aten.view.default(expand_4, [128, 512, 32]);  expand_4 = None
        expand_5 = torch.ops.aten.expand.default(permute_10, [16, 8, 32, 512]);  permute_10 = None
        clone_2 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_34 = torch.ops.aten.view.default(clone_2, [128, 32, 512]);  clone_2 = None
        bmm_2 = torch.ops.aten.bmm.default(view_33, view_34)
        view_35 = torch.ops.aten.view.default(bmm_2, [16, 8, 512, 512]);  bmm_2 = None
        div_2 = torch.ops.aten.div.Tensor(view_35, 5.656854249492381);  view_35 = None
        eq_1 = torch.ops.aten.eq.Scalar(primals_30, 0)
        where_1 = torch.ops.aten.where.self(eq_1, full_default, div_2);  eq_1 = div_2 = None
        amax_1 = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        expand_6 = torch.ops.aten.expand.default(div_3, [16, 8, 512, 512])
        view_36 = torch.ops.aten.view.default(expand_6, [128, 512, 512]);  expand_6 = None
        expand_7 = torch.ops.aten.expand.default(view_30, [16, 8, 512, 32]);  view_30 = None
        view_37 = torch.ops.aten.view.default(expand_7, [128, 512, 32]);  expand_7 = None
        bmm_3 = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = None
        view_38 = torch.ops.aten.view.default(bmm_3, [16, 8, 512, 32]);  bmm_3 = None
        view_39 = torch.ops.aten.view.default(view_38, [16, -1, 256]);  view_38 = None
        add_10 = torch.ops.aten.add.Tensor(view_39, add_6);  view_39 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_11 = torch.ops.aten.add.Scalar(mean_3, 1.1920928955078125e-07);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_10, rsqrt_3)
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, primals_31);  mul_24 = None
        view_40 = torch.ops.aten.view.default(mul_25, [8192, 256])
        permute_11 = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_33, view_40, permute_11);  primals_33 = None
        view_41 = torch.ops.aten.view.default(addmm_9, [16, 512, 682])
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_41)
        mul_26 = torch.ops.aten.mul.Tensor(view_41, sigmoid_1);  view_41 = sigmoid_1 = None
        permute_12 = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        addmm_10 = torch.ops.aten.addmm.default(primals_35, view_40, permute_12);  primals_35 = None
        view_43 = torch.ops.aten.view.default(addmm_10, [16, 512, 682])
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, view_43);  mul_26 = view_43 = None
        view_44 = torch.ops.aten.view.default(mul_27, [8192, 682]);  mul_27 = None
        permute_13 = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_37, view_44, permute_13);  primals_37 = None
        view_45 = torch.ops.aten.view.default(addmm_11, [16, 512, 256]);  addmm_11 = None
        add_12 = torch.ops.aten.add.Tensor(view_45, mul_25);  view_45 = mul_25 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_12, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_13 = torch.ops.aten.add.Scalar(mean_4, 1.1920928955078125e-07);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_28 = torch.ops.aten.mul.Tensor(add_12, rsqrt_4)
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, primals_38);  mul_28 = None
        view_46 = torch.ops.aten.view.default(mul_29, [8192, 256]);  mul_29 = None
        permute_14 = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        addmm_12 = torch.ops.aten.addmm.default(primals_40, view_46, permute_14);  primals_40 = None
        view_47 = torch.ops.aten.view.default(addmm_12, [16, 512, 256]);  addmm_12 = None
        permute_15 = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_42, view_46, permute_15);  primals_42 = None
        view_49 = torch.ops.aten.view.default(addmm_13, [16, 512, 32]);  addmm_13 = None
        permute_16 = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_44, view_46, permute_16);  primals_44 = None
        view_51 = torch.ops.aten.view.default(addmm_14, [16, 512, 256]);  addmm_14 = None
        view_52 = torch.ops.aten.view.default(view_47, [16, 8, 512, 32]);  view_47 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(view_49, 1);  view_49 = None
        view_53 = torch.ops.aten.view.default(view_51, [16, 8, 512, 32]);  view_51 = None
        slice_31 = torch.ops.aten.slice.Tensor(primals_45, 0, 0, 512);  primals_45 = None
        slice_32 = torch.ops.aten.slice.Tensor(primals_46, 0, 0, 512);  primals_46 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(slice_31, 0);  slice_31 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(unsqueeze_27, 0);  unsqueeze_27 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(slice_32, 0);  slice_32 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(unsqueeze_29, 0);  unsqueeze_29 = None
        slice_33 = torch.ops.aten.slice.Tensor(view_52, 3, 0, 9223372036854775807, 2)
        slice_34 = torch.ops.aten.slice.Tensor(view_52, 3, 1, 9223372036854775807, 2);  view_52 = None
        slice_35 = torch.ops.aten.slice.Tensor(unsqueeze_28, 3, 0, 9223372036854775807, 2);  unsqueeze_28 = None
        slice_36 = torch.ops.aten.slice.Tensor(unsqueeze_30, 3, 0, 9223372036854775807, 2);  unsqueeze_30 = None
        mul_30 = torch.ops.aten.mul.Tensor(slice_33, slice_35)
        mul_31 = torch.ops.aten.mul.Tensor(slice_34, slice_36)
        sub_6 = torch.ops.aten.sub.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(slice_33, slice_36);  slice_33 = None
        mul_33 = torch.ops.aten.mul.Tensor(slice_34, slice_35);  slice_34 = None
        add_14 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(sub_6, 4);  sub_6 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(add_14, 4);  add_14 = None
        cat_4 = torch.ops.aten.cat.default([unsqueeze_31, unsqueeze_32], -1);  unsqueeze_31 = unsqueeze_32 = None
        view_54 = torch.ops.aten.view.default(cat_4, [16, 8, 512, 32]);  cat_4 = None
        slice_39 = torch.ops.aten.slice.Tensor(unsqueeze_26, 3, 0, 9223372036854775807, 2)
        slice_40 = torch.ops.aten.slice.Tensor(unsqueeze_26, 3, 1, 9223372036854775807, 2);  unsqueeze_26 = None
        mul_34 = torch.ops.aten.mul.Tensor(slice_39, slice_35)
        mul_35 = torch.ops.aten.mul.Tensor(slice_40, slice_36)
        sub_7 = torch.ops.aten.sub.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(slice_39, slice_36);  slice_39 = None
        mul_37 = torch.ops.aten.mul.Tensor(slice_40, slice_35);  slice_40 = None
        add_15 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(sub_7, 4);  sub_7 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(add_15, 4);  add_15 = None
        cat_5 = torch.ops.aten.cat.default([unsqueeze_37, unsqueeze_38], -1);  unsqueeze_37 = unsqueeze_38 = None
        view_55 = torch.ops.aten.view.default(cat_5, [16, 1, 512, 32]);  cat_5 = None
        permute_17 = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
        expand_8 = torch.ops.aten.expand.default(view_54, [16, 8, 512, 32]);  view_54 = None
        view_56 = torch.ops.aten.view.default(expand_8, [128, 512, 32]);  expand_8 = None
        expand_9 = torch.ops.aten.expand.default(permute_17, [16, 8, 32, 512]);  permute_17 = None
        clone_3 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_57 = torch.ops.aten.view.default(clone_3, [128, 32, 512]);  clone_3 = None
        bmm_4 = torch.ops.aten.bmm.default(view_56, view_57)
        view_58 = torch.ops.aten.view.default(bmm_4, [16, 8, 512, 512]);  bmm_4 = None
        div_4 = torch.ops.aten.div.Tensor(view_58, 5.656854249492381);  view_58 = None
        eq_2 = torch.ops.aten.eq.Scalar(primals_47, 0)
        where_2 = torch.ops.aten.where.self(eq_2, full_default, div_4);  eq_2 = div_4 = None
        amax_2 = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        expand_10 = torch.ops.aten.expand.default(div_5, [16, 8, 512, 512])
        view_59 = torch.ops.aten.view.default(expand_10, [128, 512, 512]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(view_53, [16, 8, 512, 32]);  view_53 = None
        view_60 = torch.ops.aten.view.default(expand_11, [128, 512, 32]);  expand_11 = None
        bmm_5 = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = None
        view_61 = torch.ops.aten.view.default(bmm_5, [16, 8, 512, 32]);  bmm_5 = None
        view_62 = torch.ops.aten.view.default(view_61, [16, -1, 256]);  view_61 = None
        add_16 = torch.ops.aten.add.Tensor(view_62, add_12);  view_62 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_17 = torch.ops.aten.add.Scalar(mean_5, 1.1920928955078125e-07);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, primals_48);  mul_38 = None
        view_63 = torch.ops.aten.view.default(mul_39, [8192, 256])
        permute_18 = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_50, view_63, permute_18);  primals_50 = None
        view_64 = torch.ops.aten.view.default(addmm_15, [16, 512, 682])
        sigmoid_2 = torch.ops.aten.sigmoid.default(view_64)
        mul_40 = torch.ops.aten.mul.Tensor(view_64, sigmoid_2);  view_64 = sigmoid_2 = None
        permute_19 = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        addmm_16 = torch.ops.aten.addmm.default(primals_52, view_63, permute_19);  primals_52 = None
        view_66 = torch.ops.aten.view.default(addmm_16, [16, 512, 682])
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, view_66);  mul_40 = view_66 = None
        view_67 = torch.ops.aten.view.default(mul_41, [8192, 682]);  mul_41 = None
        permute_20 = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_54, view_67, permute_20);  primals_54 = None
        view_68 = torch.ops.aten.view.default(addmm_17, [16, 512, 256]);  addmm_17 = None
        add_18 = torch.ops.aten.add.Tensor(view_68, mul_39);  view_68 = mul_39 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_19 = torch.ops.aten.add.Scalar(mean_6, 1.1920928955078125e-07);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_42 = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_55);  mul_42 = None
        view_69 = torch.ops.aten.view.default(mul_43, [8192, 256]);  mul_43 = None
        permute_21 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_18 = torch.ops.aten.addmm.default(primals_57, view_69, permute_21);  primals_57 = None
        view_70 = torch.ops.aten.view.default(addmm_18, [16, 512, 256]);  addmm_18 = None
        permute_22 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_59, view_69, permute_22);  primals_59 = None
        view_72 = torch.ops.aten.view.default(addmm_19, [16, 512, 32]);  addmm_19 = None
        permute_23 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_61, view_69, permute_23);  primals_61 = None
        view_74 = torch.ops.aten.view.default(addmm_20, [16, 512, 256]);  addmm_20 = None
        view_75 = torch.ops.aten.view.default(view_70, [16, 8, 512, 32]);  view_70 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(view_72, 1);  view_72 = None
        view_76 = torch.ops.aten.view.default(view_74, [16, 8, 512, 32]);  view_74 = None
        slice_45 = torch.ops.aten.slice.Tensor(primals_62, 0, 0, 512);  primals_62 = None
        slice_46 = torch.ops.aten.slice.Tensor(primals_63, 0, 0, 512);  primals_63 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(slice_45, 0);  slice_45 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 0);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(slice_46, 0);  slice_46 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        slice_47 = torch.ops.aten.slice.Tensor(view_75, 3, 0, 9223372036854775807, 2)
        slice_48 = torch.ops.aten.slice.Tensor(view_75, 3, 1, 9223372036854775807, 2);  view_75 = None
        slice_49 = torch.ops.aten.slice.Tensor(unsqueeze_41, 3, 0, 9223372036854775807, 2);  unsqueeze_41 = None
        slice_50 = torch.ops.aten.slice.Tensor(unsqueeze_43, 3, 0, 9223372036854775807, 2);  unsqueeze_43 = None
        mul_44 = torch.ops.aten.mul.Tensor(slice_47, slice_49)
        mul_45 = torch.ops.aten.mul.Tensor(slice_48, slice_50)
        sub_9 = torch.ops.aten.sub.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(slice_47, slice_50);  slice_47 = None
        mul_47 = torch.ops.aten.mul.Tensor(slice_48, slice_49);  slice_48 = None
        add_20 = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(sub_9, 4);  sub_9 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(add_20, 4);  add_20 = None
        cat_6 = torch.ops.aten.cat.default([unsqueeze_44, unsqueeze_45], -1);  unsqueeze_44 = unsqueeze_45 = None
        view_77 = torch.ops.aten.view.default(cat_6, [16, 8, 512, 32]);  cat_6 = None
        slice_53 = torch.ops.aten.slice.Tensor(unsqueeze_39, 3, 0, 9223372036854775807, 2)
        slice_54 = torch.ops.aten.slice.Tensor(unsqueeze_39, 3, 1, 9223372036854775807, 2);  unsqueeze_39 = None
        mul_48 = torch.ops.aten.mul.Tensor(slice_53, slice_49)
        mul_49 = torch.ops.aten.mul.Tensor(slice_54, slice_50)
        sub_10 = torch.ops.aten.sub.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        mul_50 = torch.ops.aten.mul.Tensor(slice_53, slice_50);  slice_53 = None
        mul_51 = torch.ops.aten.mul.Tensor(slice_54, slice_49);  slice_54 = None
        add_21 = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(sub_10, 4);  sub_10 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(add_21, 4);  add_21 = None
        cat_7 = torch.ops.aten.cat.default([unsqueeze_50, unsqueeze_51], -1);  unsqueeze_50 = unsqueeze_51 = None
        view_78 = torch.ops.aten.view.default(cat_7, [16, 1, 512, 32]);  cat_7 = None
        permute_24 = torch.ops.aten.permute.default(view_78, [0, 1, 3, 2]);  view_78 = None
        expand_12 = torch.ops.aten.expand.default(view_77, [16, 8, 512, 32]);  view_77 = None
        view_79 = torch.ops.aten.view.default(expand_12, [128, 512, 32]);  expand_12 = None
        expand_13 = torch.ops.aten.expand.default(permute_24, [16, 8, 32, 512]);  permute_24 = None
        clone_4 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_80 = torch.ops.aten.view.default(clone_4, [128, 32, 512]);  clone_4 = None
        bmm_6 = torch.ops.aten.bmm.default(view_79, view_80)
        view_81 = torch.ops.aten.view.default(bmm_6, [16, 8, 512, 512]);  bmm_6 = None
        div_6 = torch.ops.aten.div.Tensor(view_81, 5.656854249492381);  view_81 = None
        eq_3 = torch.ops.aten.eq.Scalar(primals_64, 0)
        where_3 = torch.ops.aten.where.self(eq_3, full_default, div_6);  eq_3 = full_default = div_6 = None
        amax_3 = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_14 = torch.ops.aten.expand.default(div_7, [16, 8, 512, 512])
        view_82 = torch.ops.aten.view.default(expand_14, [128, 512, 512]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(view_76, [16, 8, 512, 32]);  view_76 = None
        view_83 = torch.ops.aten.view.default(expand_15, [128, 512, 32]);  expand_15 = None
        bmm_7 = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = None
        view_84 = torch.ops.aten.view.default(bmm_7, [16, 8, 512, 32]);  bmm_7 = None
        view_85 = torch.ops.aten.view.default(view_84, [16, -1, 256]);  view_84 = None
        add_22 = torch.ops.aten.add.Tensor(view_85, add_18);  view_85 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_23 = torch.ops.aten.add.Scalar(mean_7, 1.1920928955078125e-07);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_52 = torch.ops.aten.mul.Tensor(add_22, rsqrt_7)
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, primals_65);  mul_52 = None
        view_86 = torch.ops.aten.view.default(mul_53, [8192, 256])
        permute_25 = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_67, view_86, permute_25);  primals_67 = None
        view_87 = torch.ops.aten.view.default(addmm_21, [16, 512, 682])
        sigmoid_3 = torch.ops.aten.sigmoid.default(view_87)
        mul_54 = torch.ops.aten.mul.Tensor(view_87, sigmoid_3);  view_87 = sigmoid_3 = None
        permute_26 = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        addmm_22 = torch.ops.aten.addmm.default(primals_69, view_86, permute_26);  primals_69 = None
        view_89 = torch.ops.aten.view.default(addmm_22, [16, 512, 682])
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, view_89);  mul_54 = view_89 = None
        view_90 = torch.ops.aten.view.default(mul_55, [8192, 682]);  mul_55 = None
        permute_27 = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_71, view_90, permute_27);  primals_71 = None
        view_91 = torch.ops.aten.view.default(addmm_23, [16, 512, 256]);  addmm_23 = None
        add_24 = torch.ops.aten.add.Tensor(view_91, mul_53);  view_91 = mul_53 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_25 = torch.ops.aten.add.Scalar(mean_8, 1.1920928955078125e-07);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_56 = torch.ops.aten.mul.Tensor(add_24, rsqrt_8)
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, primals_72);  mul_56 = None
        view_92 = torch.ops.aten.view.default(mul_57, [8192, 256]);  mul_57 = None
        permute_28 = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm_24 = torch.ops.aten.addmm.default(primals_73, view_92, permute_28);  primals_73 = None
        view_93 = torch.ops.aten.view.default(addmm_24, [16, 512, 10000]);  addmm_24 = None
        permute_29 = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        permute_33 = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_37 = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_42 = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_47 = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
        permute_48 = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_49 = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        permute_51 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_55 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_59 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_63 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_67 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_72 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        permute_77 = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_78 = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_79 = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_81 = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        permute_85 = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_89 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_93 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_97 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_102 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_107 = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_108 = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_109 = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_111 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_115 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_119 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_123 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        permute_127 = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_132 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_137 = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_138 = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_139 = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_141 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_145 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_149 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (view_93, primals_2, primals_3, primals_4, primals_13, primals_14, primals_21, primals_30, primals_31, primals_38, primals_47, primals_48, primals_55, primals_64, primals_65, primals_72, embedding, rsqrt, view, slice_7, slice_8, div_1, add_4, rsqrt_1, view_17, addmm_3, addmm_4, view_21, add_6, rsqrt_2, view_23, slice_21, slice_22, div_3, add_10, rsqrt_3, view_40, addmm_9, addmm_10, view_44, add_12, rsqrt_4, view_46, slice_35, slice_36, div_5, add_16, rsqrt_5, view_63, addmm_15, addmm_16, view_67, add_18, rsqrt_6, view_69, slice_49, slice_50, div_7, add_22, rsqrt_7, view_86, addmm_21, addmm_22, view_90, add_24, rsqrt_8, view_92, permute_29, permute_33, permute_37, permute_42, permute_47, permute_48, permute_49, permute_51, permute_55, permute_59, permute_63, permute_67, permute_72, permute_77, permute_78, permute_79, permute_81, permute_85, permute_89, permute_93, permute_97, permute_102, permute_107, permute_108, permute_109, permute_111, permute_115, permute_119, permute_123, permute_127, permute_132, permute_137, permute_138, permute_139, permute_141, permute_145, permute_149)
        
def load_args(reader):
    buf0 = reader.storage(None, 10240000)
    reader.tensor(buf0, (10000, 256), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 65536, dtype_hint=torch.int64)
    reader.tensor(buf1, (16, 512), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 524288)
    reader.tensor(buf2, (1, 512, 256), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 1024)
    reader.tensor(buf3, (256,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 262144)
    reader.tensor(buf4, (256, 256), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1024)
    reader.tensor(buf5, (256,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 32768)
    reader.tensor(buf6, (32, 256), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 128)
    reader.tensor(buf7, (32,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 262144)
    reader.tensor(buf8, (256, 256), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 1024)
    reader.tensor(buf9, (256,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf10, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf11, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 262144, dtype_hint=torch.bool)
    reader.tensor(buf12, (1, 1, 512, 512), dtype=torch.bool, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1024)
    reader.tensor(buf13, (256,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 698368)
    reader.tensor(buf14, (682, 256), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 2728)
    reader.tensor(buf15, (682,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 698368)
    reader.tensor(buf16, (682, 256), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2728)
    reader.tensor(buf17, (682,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 698368)
    reader.tensor(buf18, (256, 682), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 1024)
    reader.tensor(buf19, (256,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 1024)
    reader.tensor(buf20, (256,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 262144)
    reader.tensor(buf21, (256, 256), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 1024)
    reader.tensor(buf22, (256,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 32768)
    reader.tensor(buf23, (32, 256), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 128)
    reader.tensor(buf24, (32,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 262144)
    reader.tensor(buf25, (256, 256), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 1024)
    reader.tensor(buf26, (256,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf27, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf28, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 262144, dtype_hint=torch.bool)
    reader.tensor(buf29, (1, 1, 512, 512), dtype=torch.bool, is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 1024)
    reader.tensor(buf30, (256,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 698368)
    reader.tensor(buf31, (682, 256), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 2728)
    reader.tensor(buf32, (682,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 698368)
    reader.tensor(buf33, (682, 256), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 2728)
    reader.tensor(buf34, (682,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 698368)
    reader.tensor(buf35, (256, 682), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 1024)
    reader.tensor(buf36, (256,), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 1024)
    reader.tensor(buf37, (256,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 262144)
    reader.tensor(buf38, (256, 256), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 1024)
    reader.tensor(buf39, (256,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 32768)
    reader.tensor(buf40, (32, 256), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 128)
    reader.tensor(buf41, (32,), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 262144)
    reader.tensor(buf42, (256, 256), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 1024)
    reader.tensor(buf43, (256,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf44, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf45, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 262144, dtype_hint=torch.bool)
    reader.tensor(buf46, (1, 1, 512, 512), dtype=torch.bool, is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 1024)
    reader.tensor(buf47, (256,), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 698368)
    reader.tensor(buf48, (682, 256), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 2728)
    reader.tensor(buf49, (682,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 698368)
    reader.tensor(buf50, (682, 256), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 2728)
    reader.tensor(buf51, (682,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 698368)
    reader.tensor(buf52, (256, 682), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 1024)
    reader.tensor(buf53, (256,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 1024)
    reader.tensor(buf54, (256,), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 262144)
    reader.tensor(buf55, (256, 256), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 1024)
    reader.tensor(buf56, (256,), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 32768)
    reader.tensor(buf57, (32, 256), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 128)
    reader.tensor(buf58, (32,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 262144)
    reader.tensor(buf59, (256, 256), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 1024)
    reader.tensor(buf60, (256,), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf61, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 32768, dtype_hint=torch.float16)
    reader.tensor(buf62, (512, 32), dtype=torch.float16, is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 262144, dtype_hint=torch.bool)
    reader.tensor(buf63, (1, 1, 512, 512), dtype=torch.bool, is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 1024)
    reader.tensor(buf64, (256,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 698368)
    reader.tensor(buf65, (682, 256), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 2728)
    reader.tensor(buf66, (682,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 698368)
    reader.tensor(buf67, (682, 256), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 2728)
    reader.tensor(buf68, (682,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 698368)
    reader.tensor(buf69, (256, 682), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1024)
    reader.tensor(buf70, (256,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 1024)
    reader.tensor(buf71, (256,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 40000)
    reader.tensor(buf72, (10000,), is_leaf=True)  # primals_73
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)