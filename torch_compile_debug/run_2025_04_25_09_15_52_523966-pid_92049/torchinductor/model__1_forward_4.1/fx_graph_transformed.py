class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10000, 256]", primals_2: "i64[16, 512]", primals_3: "f32[1, 512, 256]", primals_4: "f32[256]", primals_5: "f32[256, 256]", primals_6: "f32[256]", primals_7: "f32[32, 256]", primals_8: "f32[32]", primals_9: "f32[256, 256]", primals_10: "f32[256]", primals_11: "f16[512, 32]", primals_12: "f16[512, 32]", primals_13: "b8[1, 1, 512, 512]", primals_14: "f32[256]", primals_15: "f32[682, 256]", primals_16: "f32[682]", primals_17: "f32[682, 256]", primals_18: "f32[682]", primals_19: "f32[256, 682]", primals_20: "f32[256]", primals_21: "f32[256]", primals_22: "f32[256, 256]", primals_23: "f32[256]", primals_24: "f32[32, 256]", primals_25: "f32[32]", primals_26: "f32[256, 256]", primals_27: "f32[256]", primals_28: "f16[512, 32]", primals_29: "f16[512, 32]", primals_30: "b8[1, 1, 512, 512]", primals_31: "f32[256]", primals_32: "f32[682, 256]", primals_33: "f32[682]", primals_34: "f32[682, 256]", primals_35: "f32[682]", primals_36: "f32[256, 682]", primals_37: "f32[256]", primals_38: "f32[256]", primals_39: "f32[256, 256]", primals_40: "f32[256]", primals_41: "f32[32, 256]", primals_42: "f32[32]", primals_43: "f32[256, 256]", primals_44: "f32[256]", primals_45: "f16[512, 32]", primals_46: "f16[512, 32]", primals_47: "b8[1, 1, 512, 512]", primals_48: "f32[256]", primals_49: "f32[682, 256]", primals_50: "f32[682]", primals_51: "f32[682, 256]", primals_52: "f32[682]", primals_53: "f32[256, 682]", primals_54: "f32[256]", primals_55: "f32[256]", primals_56: "f32[256, 256]", primals_57: "f32[256]", primals_58: "f32[32, 256]", primals_59: "f32[32]", primals_60: "f32[256, 256]", primals_61: "f32[256]", primals_62: "f16[512, 32]", primals_63: "f16[512, 32]", primals_64: "b8[1, 1, 512, 512]", primals_65: "f32[256]", primals_66: "f32[682, 256]", primals_67: "f32[682]", primals_68: "f32[682, 256]", primals_69: "f32[682]", primals_70: "f32[256, 682]", primals_71: "f32[256]", primals_72: "f32[256]", primals_73: "f32[10000]"):
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/model.py:158 in forward, code: x = self.embeddings(x)
        embedding: "f32[16, 512, 256]" = torch.ops.aten.embedding.default(primals_1, primals_2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:884 in forward, code: x_pe = x + self.positional_embedding[:, :x.size(1), :]
        add: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(embedding, primals_3)
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_1: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add_1: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add, rsqrt)
        mul_1: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        view: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_1, [8192, 256]);  mul_1 = None
        permute: "f32[256, 256]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = None
        view_1: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm, [16, 512, 256]);  addmm = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_1: "f32[256, 32]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_1: "f32[8192, 32]" = torch.ops.aten.addmm.default(primals_8, view, permute_1);  primals_8 = None
        view_3: "f32[16, 512, 32]" = torch.ops.aten.reshape.default(addmm_1, [16, 512, 32]);  addmm_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_2: "f32[256, 256]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        addmm_2: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_10, view, permute_2);  primals_10 = None
        view_5: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_2, [16, 512, 256]);  addmm_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:99 in forward, code: self.k_cache = k.to(torch.float16)
        convert_element_type: "f16[16, 512, 32]" = torch.ops.prims.convert_element_type.default(view_3, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:100 in forward, code: self.v_cache = v.to(torch.float16)
        convert_element_type_1: "f16[16, 512, 256]" = torch.ops.prims.convert_element_type.default(view_5, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:536 in forward, code: q = q.view(b, self.n_heads, q_l, self.d_head)
        view_6: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_1, [16, 8, 512, 32]);  view_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:537 in forward, code: k = k.unsqueeze(1)
        unsqueeze: "f32[16, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:538 in forward, code: v = v.view(b, self.n_heads, v_l, self.d_head)
        view_7: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_5, [16, 8, 512, 32]);  view_5 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:984 in forward, code: cos = self.rope_cos[self.t - 1:self.t]
        slice_3: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_11, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:985 in forward, code: sin = self.rope_sin[self.t - 1:self.t]
        slice_4: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_12, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_1: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_3, 0);  slice_3 = None
        unsqueeze_2: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 0);  unsqueeze_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_3: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_4, 0);  slice_4 = None
        unsqueeze_4: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 0);  unsqueeze_3 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_5: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_6, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_6: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_6, 3, 1, 9223372036854775807, 2);  view_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_7: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_2, 3, 0, 9223372036854775807, 2);  unsqueeze_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_8: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_4, 3, 0, 9223372036854775807, 2);  unsqueeze_4 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_2: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_5, slice_7)
        mul_3: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_6, slice_8)
        sub: "f32[16, 8, 512, 16]" = torch.ops.aten.sub.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_4: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_5, slice_8);  slice_5 = None
        mul_5: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_6, slice_7);  slice_6 = None
        add_2: "f32[16, 8, 512, 16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_5: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub, 4);  sub = None
        unsqueeze_6: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_2, 4);  add_2 = None
        cat: "f32[16, 8, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_5, unsqueeze_6], -1);  unsqueeze_5 = unsqueeze_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_8: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(cat, [16, 8, 512, 32]);  cat = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:987 in forward, code: cos = self.rope_cos[:x.shape[2]]
        slice_9: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_11, 0, 0, 512);  primals_11 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:988 in forward, code: sin = self.rope_sin[:x.shape[2]]
        slice_10: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_12, 0, 0, 512);  primals_12 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_7: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_9, 0);  slice_9 = None
        unsqueeze_8: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 0);  unsqueeze_7 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_9: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_10, 0);  slice_10 = None
        unsqueeze_10: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 0);  unsqueeze_9 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_11: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_12: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze, 3, 1, 9223372036854775807, 2);  unsqueeze = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_13: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_8, 3, 0, 9223372036854775807, 2);  unsqueeze_8 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_14: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_10, 3, 0, 9223372036854775807, 2);  unsqueeze_10 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_6: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_11, slice_13)
        mul_7: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_12, slice_14)
        sub_1: "f32[16, 1, 512, 16]" = torch.ops.aten.sub.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_8: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_11, slice_14);  slice_11 = None
        mul_9: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_12, slice_13);  slice_12 = None
        add_3: "f32[16, 1, 512, 16]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_11: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_1, 4);  sub_1 = None
        unsqueeze_12: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_3, 4);  add_3 = None
        cat_1: "f32[16, 1, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_11, unsqueeze_12], -1);  unsqueeze_11 = unsqueeze_12 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_9: "f32[16, 1, 512, 32]" = torch.ops.aten.reshape.default(cat_1, [16, 1, 512, 32]);  cat_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_3: "f32[16, 1, 32, 512]" = torch.ops.aten.permute.default(view_9, [0, 1, 3, 2]);  view_9 = None
        expand: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_8, [16, 8, 512, 32]);  view_8 = None
        view_10: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand, [128, 512, 32]);  expand = None
        expand_1: "f32[16, 8, 32, 512]" = torch.ops.aten.expand.default(permute_3, [16, 8, 32, 512]);  permute_3 = None
        clone_1: "f32[16, 8, 32, 512]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11: "f32[128, 32, 512]" = torch.ops.aten.reshape.default(clone_1, [128, 32, 512]);  clone_1 = None
        bmm: "f32[128, 512, 512]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "f32[16, 8, 512, 512]" = torch.ops.aten.reshape.default(bmm, [16, 8, 512, 512]);  bmm = None
        div: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(view_12, 5.656854249492381);  view_12 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:547 in forward, code: attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :q_l, :v_l] == 0, float("-inf"))
        eq: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Scalar(primals_13, 0)
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "f32[16, 8, 512, 512]" = torch.ops.aten.where.self(eq, full_default, div);  eq = div = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:548 in forward, code: attn_scores = F.softmax(attn_logits, dim=-1)
        amax: "f32[16, 8, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
        sub_2: "f32[16, 8, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: "f32[16, 8, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[16, 8, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        expand_2: "f32[16, 8, 512, 512]" = torch.ops.aten.expand.default(div_1, [16, 8, 512, 512])
        view_13: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [128, 512, 512]);  expand_2 = None
        expand_3: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_7, [16, 8, 512, 32]);  view_7 = None
        view_14: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_3, [128, 512, 32]);  expand_3 = None
        bmm_1: "f32[128, 512, 32]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = None
        view_15: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(bmm_1, [16, 8, 512, 32]);  bmm_1 = None
        view_16: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(view_15, [16, -1, 256]);  view_15 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:115 in forward, code: x = self.attention(q, k, v, _inference=_inference) + x_res
        add_4: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_16, add);  view_16 = add = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_2: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_4, 2)
        mean_1: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_5: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_1, 1.1920928955078125e-07);  mean_1 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_10: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_4, rsqrt_1)
        mul_11: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, primals_14);  mul_10 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        view_17: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_11, [8192, 256])
        permute_4: "f32[256, 682]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        addmm_3: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_16, view_17, permute_4);  primals_16 = None
        view_18: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_3, [16, 512, 682])
        sigmoid: "f32[16, 512, 682]" = torch.ops.aten.sigmoid.default(view_18)
        mul_12: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(view_18, sigmoid);  view_18 = sigmoid = None
        permute_5: "f32[256, 682]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        addmm_4: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_18, view_17, permute_5);  primals_18 = None
        view_20: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 682])
        mul_13: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(mul_12, view_20);  mul_12 = view_20 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        view_21: "f32[8192, 682]" = torch.ops.aten.reshape.default(mul_13, [8192, 682]);  mul_13 = None
        permute_6: "f32[682, 256]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        addmm_5: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_20, view_21, permute_6);  primals_20 = None
        view_22: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_5, [16, 512, 256]);  addmm_5 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:118 in forward, code: x = self.swigluNN(x) + x
        add_6: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_22, mul_11);  view_22 = mul_11 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_3: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_2: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_7: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_2, 1.1920928955078125e-07);  mean_2 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_14: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_2)
        mul_15: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_14, primals_21);  mul_14 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        view_23: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_15, [8192, 256]);  mul_15 = None
        permute_7: "f32[256, 256]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        addmm_6: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_23, view_23, permute_7);  primals_23 = None
        view_24: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_6, [16, 512, 256]);  addmm_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_8: "f32[256, 32]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_7: "f32[8192, 32]" = torch.ops.aten.addmm.default(primals_25, view_23, permute_8);  primals_25 = None
        view_26: "f32[16, 512, 32]" = torch.ops.aten.reshape.default(addmm_7, [16, 512, 32]);  addmm_7 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_8: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_27, view_23, permute_9);  primals_27 = None
        view_28: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_8, [16, 512, 256]);  addmm_8 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:99 in forward, code: self.k_cache = k.to(torch.float16)
        convert_element_type_2: "f16[16, 512, 32]" = torch.ops.prims.convert_element_type.default(view_26, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:100 in forward, code: self.v_cache = v.to(torch.float16)
        convert_element_type_3: "f16[16, 512, 256]" = torch.ops.prims.convert_element_type.default(view_28, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:536 in forward, code: q = q.view(b, self.n_heads, q_l, self.d_head)
        view_29: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_24, [16, 8, 512, 32]);  view_24 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:537 in forward, code: k = k.unsqueeze(1)
        unsqueeze_13: "f32[16, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(view_26, 1);  view_26 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:538 in forward, code: v = v.view(b, self.n_heads, v_l, self.d_head)
        view_30: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_28, [16, 8, 512, 32]);  view_28 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:984 in forward, code: cos = self.rope_cos[self.t - 1:self.t]
        slice_17: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_28, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:985 in forward, code: sin = self.rope_sin[self.t - 1:self.t]
        slice_18: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_29, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_14: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_17, 0);  slice_17 = None
        unsqueeze_15: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_16: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_18, 0);  slice_18 = None
        unsqueeze_17: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 0);  unsqueeze_16 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_19: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_29, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_20: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_29, 3, 1, 9223372036854775807, 2);  view_29 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_21: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 3, 0, 9223372036854775807, 2);  unsqueeze_15 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_22: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_17, 3, 0, 9223372036854775807, 2);  unsqueeze_17 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_16: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_19, slice_21)
        mul_17: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_20, slice_22)
        sub_3: "f32[16, 8, 512, 16]" = torch.ops.aten.sub.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_18: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_19, slice_22);  slice_19 = None
        mul_19: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_20, slice_21);  slice_20 = None
        add_8: "f32[16, 8, 512, 16]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_18: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_3, 4);  sub_3 = None
        unsqueeze_19: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_8, 4);  add_8 = None
        cat_2: "f32[16, 8, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_18, unsqueeze_19], -1);  unsqueeze_18 = unsqueeze_19 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_31: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(cat_2, [16, 8, 512, 32]);  cat_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:987 in forward, code: cos = self.rope_cos[:x.shape[2]]
        slice_23: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_28, 0, 0, 512);  primals_28 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:988 in forward, code: sin = self.rope_sin[:x.shape[2]]
        slice_24: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_29, 0, 0, 512);  primals_29 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_20: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_23, 0);  slice_23 = None
        unsqueeze_21: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 0);  unsqueeze_20 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_22: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_24, 0);  slice_24 = None
        unsqueeze_23: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_25: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_26: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 1, 9223372036854775807, 2);  unsqueeze_13 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_27: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_21, 3, 0, 9223372036854775807, 2);  unsqueeze_21 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_28: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_23, 3, 0, 9223372036854775807, 2);  unsqueeze_23 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_20: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_25, slice_27)
        mul_21: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_26, slice_28)
        sub_4: "f32[16, 1, 512, 16]" = torch.ops.aten.sub.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_22: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_25, slice_28);  slice_25 = None
        mul_23: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_26, slice_27);  slice_26 = None
        add_9: "f32[16, 1, 512, 16]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_24: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_4, 4);  sub_4 = None
        unsqueeze_25: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_9, 4);  add_9 = None
        cat_3: "f32[16, 1, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_24, unsqueeze_25], -1);  unsqueeze_24 = unsqueeze_25 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_32: "f32[16, 1, 512, 32]" = torch.ops.aten.reshape.default(cat_3, [16, 1, 512, 32]);  cat_3 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_10: "f32[16, 1, 32, 512]" = torch.ops.aten.permute.default(view_32, [0, 1, 3, 2]);  view_32 = None
        expand_4: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_31, [16, 8, 512, 32]);  view_31 = None
        view_33: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_4, [128, 512, 32]);  expand_4 = None
        expand_5: "f32[16, 8, 32, 512]" = torch.ops.aten.expand.default(permute_10, [16, 8, 32, 512]);  permute_10 = None
        clone_2: "f32[16, 8, 32, 512]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_34: "f32[128, 32, 512]" = torch.ops.aten.reshape.default(clone_2, [128, 32, 512]);  clone_2 = None
        bmm_2: "f32[128, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
        view_35: "f32[16, 8, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [16, 8, 512, 512]);  bmm_2 = None
        div_2: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(view_35, 5.656854249492381);  view_35 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:547 in forward, code: attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :q_l, :v_l] == 0, float("-inf"))
        eq_1: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Scalar(primals_30, 0)
        where_1: "f32[16, 8, 512, 512]" = torch.ops.aten.where.self(eq_1, full_default, div_2);  eq_1 = div_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:548 in forward, code: attn_scores = F.softmax(attn_logits, dim=-1)
        amax_1: "f32[16, 8, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_5: "f32[16, 8, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_1: "f32[16, 8, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[16, 8, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        expand_6: "f32[16, 8, 512, 512]" = torch.ops.aten.expand.default(div_3, [16, 8, 512, 512])
        view_36: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(expand_6, [128, 512, 512]);  expand_6 = None
        expand_7: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_30, [16, 8, 512, 32]);  view_30 = None
        view_37: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_7, [128, 512, 32]);  expand_7 = None
        bmm_3: "f32[128, 512, 32]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = None
        view_38: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(bmm_3, [16, 8, 512, 32]);  bmm_3 = None
        view_39: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(view_38, [16, -1, 256]);  view_38 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:115 in forward, code: x = self.attention(q, k, v, _inference=_inference) + x_res
        add_10: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_39, add_6);  view_39 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_4: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
        mean_3: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_11: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_3, 1.1920928955078125e-07);  mean_3 = None
        rsqrt_3: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_24: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_10, rsqrt_3)
        mul_25: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, primals_31);  mul_24 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        view_40: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_25, [8192, 256])
        permute_11: "f32[256, 682]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        addmm_9: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_11);  primals_33 = None
        view_41: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_9, [16, 512, 682])
        sigmoid_1: "f32[16, 512, 682]" = torch.ops.aten.sigmoid.default(view_41)
        mul_26: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(view_41, sigmoid_1);  view_41 = sigmoid_1 = None
        permute_12: "f32[256, 682]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        addmm_10: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_35, view_40, permute_12);  primals_35 = None
        view_43: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 682])
        mul_27: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(mul_26, view_43);  mul_26 = view_43 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        view_44: "f32[8192, 682]" = torch.ops.aten.reshape.default(mul_27, [8192, 682]);  mul_27 = None
        permute_13: "f32[682, 256]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        addmm_11: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_37, view_44, permute_13);  primals_37 = None
        view_45: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_11, [16, 512, 256]);  addmm_11 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:118 in forward, code: x = self.swigluNN(x) + x
        add_12: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_45, mul_25);  view_45 = mul_25 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_5: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_12, 2)
        mean_4: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_13: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_4, 1.1920928955078125e-07);  mean_4 = None
        rsqrt_4: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_28: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_12, rsqrt_4)
        mul_29: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_28, primals_38);  mul_28 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        view_46: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_29, [8192, 256]);  mul_29 = None
        permute_14: "f32[256, 256]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        addmm_12: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_40, view_46, permute_14);  primals_40 = None
        view_47: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_12, [16, 512, 256]);  addmm_12 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_15: "f32[256, 32]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        addmm_13: "f32[8192, 32]" = torch.ops.aten.addmm.default(primals_42, view_46, permute_15);  primals_42 = None
        view_49: "f32[16, 512, 32]" = torch.ops.aten.reshape.default(addmm_13, [16, 512, 32]);  addmm_13 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        addmm_14: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_44, view_46, permute_16);  primals_44 = None
        view_51: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_14, [16, 512, 256]);  addmm_14 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:99 in forward, code: self.k_cache = k.to(torch.float16)
        convert_element_type_4: "f16[16, 512, 32]" = torch.ops.prims.convert_element_type.default(view_49, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:100 in forward, code: self.v_cache = v.to(torch.float16)
        convert_element_type_5: "f16[16, 512, 256]" = torch.ops.prims.convert_element_type.default(view_51, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:536 in forward, code: q = q.view(b, self.n_heads, q_l, self.d_head)
        view_52: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_47, [16, 8, 512, 32]);  view_47 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:537 in forward, code: k = k.unsqueeze(1)
        unsqueeze_26: "f32[16, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(view_49, 1);  view_49 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:538 in forward, code: v = v.view(b, self.n_heads, v_l, self.d_head)
        view_53: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_51, [16, 8, 512, 32]);  view_51 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:984 in forward, code: cos = self.rope_cos[self.t - 1:self.t]
        slice_31: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_45, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:985 in forward, code: sin = self.rope_sin[self.t - 1:self.t]
        slice_32: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_46, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_27: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_31, 0);  slice_31 = None
        unsqueeze_28: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 0);  unsqueeze_27 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_29: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_32, 0);  slice_32 = None
        unsqueeze_30: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 0);  unsqueeze_29 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_33: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_52, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_34: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_52, 3, 1, 9223372036854775807, 2);  view_52 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_35: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_28, 3, 0, 9223372036854775807, 2);  unsqueeze_28 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_36: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_30, 3, 0, 9223372036854775807, 2);  unsqueeze_30 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_30: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_33, slice_35)
        mul_31: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_34, slice_36)
        sub_6: "f32[16, 8, 512, 16]" = torch.ops.aten.sub.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_32: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_33, slice_36);  slice_33 = None
        mul_33: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_34, slice_35);  slice_34 = None
        add_14: "f32[16, 8, 512, 16]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_31: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_6, 4);  sub_6 = None
        unsqueeze_32: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_14, 4);  add_14 = None
        cat_4: "f32[16, 8, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_31, unsqueeze_32], -1);  unsqueeze_31 = unsqueeze_32 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_54: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(cat_4, [16, 8, 512, 32]);  cat_4 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:987 in forward, code: cos = self.rope_cos[:x.shape[2]]
        slice_37: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_45, 0, 0, 512);  primals_45 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:988 in forward, code: sin = self.rope_sin[:x.shape[2]]
        slice_38: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_46, 0, 0, 512);  primals_46 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_33: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_37, 0);  slice_37 = None
        unsqueeze_34: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 0);  unsqueeze_33 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_35: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_38, 0);  slice_38 = None
        unsqueeze_36: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 0);  unsqueeze_35 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_39: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_26, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_40: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_26, 3, 1, 9223372036854775807, 2);  unsqueeze_26 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_41: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_34, 3, 0, 9223372036854775807, 2);  unsqueeze_34 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_42: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_36, 3, 0, 9223372036854775807, 2);  unsqueeze_36 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_34: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_39, slice_41)
        mul_35: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_40, slice_42)
        sub_7: "f32[16, 1, 512, 16]" = torch.ops.aten.sub.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_36: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_39, slice_42);  slice_39 = None
        mul_37: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_40, slice_41);  slice_40 = None
        add_15: "f32[16, 1, 512, 16]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_37: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_7, 4);  sub_7 = None
        unsqueeze_38: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_15, 4);  add_15 = None
        cat_5: "f32[16, 1, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_37, unsqueeze_38], -1);  unsqueeze_37 = unsqueeze_38 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_55: "f32[16, 1, 512, 32]" = torch.ops.aten.reshape.default(cat_5, [16, 1, 512, 32]);  cat_5 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_17: "f32[16, 1, 32, 512]" = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
        expand_8: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_54, [16, 8, 512, 32]);  view_54 = None
        view_56: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_8, [128, 512, 32]);  expand_8 = None
        expand_9: "f32[16, 8, 32, 512]" = torch.ops.aten.expand.default(permute_17, [16, 8, 32, 512]);  permute_17 = None
        clone_3: "f32[16, 8, 32, 512]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_57: "f32[128, 32, 512]" = torch.ops.aten.reshape.default(clone_3, [128, 32, 512]);  clone_3 = None
        bmm_4: "f32[128, 512, 512]" = torch.ops.aten.bmm.default(view_56, view_57)
        view_58: "f32[16, 8, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [16, 8, 512, 512]);  bmm_4 = None
        div_4: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(view_58, 5.656854249492381);  view_58 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:547 in forward, code: attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :q_l, :v_l] == 0, float("-inf"))
        eq_2: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Scalar(primals_47, 0)
        where_2: "f32[16, 8, 512, 512]" = torch.ops.aten.where.self(eq_2, full_default, div_4);  eq_2 = div_4 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:548 in forward, code: attn_scores = F.softmax(attn_logits, dim=-1)
        amax_2: "f32[16, 8, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_8: "f32[16, 8, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_2: "f32[16, 8, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3: "f32[16, 8, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        expand_10: "f32[16, 8, 512, 512]" = torch.ops.aten.expand.default(div_5, [16, 8, 512, 512])
        view_59: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(expand_10, [128, 512, 512]);  expand_10 = None
        expand_11: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_53, [16, 8, 512, 32]);  view_53 = None
        view_60: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_11, [128, 512, 32]);  expand_11 = None
        bmm_5: "f32[128, 512, 32]" = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = None
        view_61: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(bmm_5, [16, 8, 512, 32]);  bmm_5 = None
        view_62: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(view_61, [16, -1, 256]);  view_61 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:115 in forward, code: x = self.attention(q, k, v, _inference=_inference) + x_res
        add_16: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_62, add_12);  view_62 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_6: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_5: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_17: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_5, 1.1920928955078125e-07);  mean_5 = None
        rsqrt_5: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_38: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
        mul_39: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, primals_48);  mul_38 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        view_63: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_39, [8192, 256])
        permute_18: "f32[256, 682]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        addmm_15: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_50, view_63, permute_18);  primals_50 = None
        view_64: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_15, [16, 512, 682])
        sigmoid_2: "f32[16, 512, 682]" = torch.ops.aten.sigmoid.default(view_64)
        mul_40: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(view_64, sigmoid_2);  view_64 = sigmoid_2 = None
        permute_19: "f32[256, 682]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        addmm_16: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_52, view_63, permute_19);  primals_52 = None
        view_66: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 682])
        mul_41: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(mul_40, view_66);  mul_40 = view_66 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        view_67: "f32[8192, 682]" = torch.ops.aten.reshape.default(mul_41, [8192, 682]);  mul_41 = None
        permute_20: "f32[682, 256]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        addmm_17: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_54, view_67, permute_20);  primals_54 = None
        view_68: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_17, [16, 512, 256]);  addmm_17 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:118 in forward, code: x = self.swigluNN(x) + x
        add_18: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_68, mul_39);  view_68 = mul_39 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_7: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
        mean_6: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_19: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_6, 1.1920928955078125e-07);  mean_6 = None
        rsqrt_6: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_42: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
        mul_43: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_42, primals_55);  mul_42 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        view_69: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_43, [8192, 256]);  mul_43 = None
        permute_21: "f32[256, 256]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_18: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_57, view_69, permute_21);  primals_57 = None
        view_70: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_18, [16, 512, 256]);  addmm_18 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_22: "f32[256, 32]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_19: "f32[8192, 32]" = torch.ops.aten.addmm.default(primals_59, view_69, permute_22);  primals_59 = None
        view_72: "f32[16, 512, 32]" = torch.ops.aten.reshape.default(addmm_19, [16, 512, 32]);  addmm_19 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        addmm_20: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_61, view_69, permute_23);  primals_61 = None
        view_74: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_20, [16, 512, 256]);  addmm_20 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:99 in forward, code: self.k_cache = k.to(torch.float16)
        convert_element_type_6: "f16[16, 512, 32]" = torch.ops.prims.convert_element_type.default(view_72, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:100 in forward, code: self.v_cache = v.to(torch.float16)
        convert_element_type_7: "f16[16, 512, 256]" = torch.ops.prims.convert_element_type.default(view_74, torch.float16)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:536 in forward, code: q = q.view(b, self.n_heads, q_l, self.d_head)
        view_75: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_70, [16, 8, 512, 32]);  view_70 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:537 in forward, code: k = k.unsqueeze(1)
        unsqueeze_39: "f32[16, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(view_72, 1);  view_72 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:538 in forward, code: v = v.view(b, self.n_heads, v_l, self.d_head)
        view_76: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(view_74, [16, 8, 512, 32]);  view_74 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:984 in forward, code: cos = self.rope_cos[self.t - 1:self.t]
        slice_45: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_62, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:985 in forward, code: sin = self.rope_sin[self.t - 1:self.t]
        slice_46: "f16[1, 32]" = torch.ops.aten.slice.Tensor(primals_63, 0, 0, 1)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_40: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_45, 0);  slice_45 = None
        unsqueeze_41: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 0);  unsqueeze_40 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_42: "f16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(slice_46, 0);  slice_46 = None
        unsqueeze_43: "f16[1, 1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_47: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_75, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_48: "f32[16, 8, 512, 16]" = torch.ops.aten.slice.Tensor(view_75, 3, 1, 9223372036854775807, 2);  view_75 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_49: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_41, 3, 0, 9223372036854775807, 2);  unsqueeze_41 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_50: "f16[1, 1, 1, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 3, 0, 9223372036854775807, 2);  unsqueeze_43 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_44: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_47, slice_49)
        mul_45: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_48, slice_50)
        sub_9: "f32[16, 8, 512, 16]" = torch.ops.aten.sub.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_46: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_47, slice_50);  slice_47 = None
        mul_47: "f32[16, 8, 512, 16]" = torch.ops.aten.mul.Tensor(slice_48, slice_49);  slice_48 = None
        add_20: "f32[16, 8, 512, 16]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_44: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_9, 4);  sub_9 = None
        unsqueeze_45: "f32[16, 8, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_20, 4);  add_20 = None
        cat_6: "f32[16, 8, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_44, unsqueeze_45], -1);  unsqueeze_44 = unsqueeze_45 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_77: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(cat_6, [16, 8, 512, 32]);  cat_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:987 in forward, code: cos = self.rope_cos[:x.shape[2]]
        slice_51: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_62, 0, 0, 512);  primals_62 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:988 in forward, code: sin = self.rope_sin[:x.shape[2]]
        slice_52: "f16[512, 32]" = torch.ops.aten.slice.Tensor(primals_63, 0, 0, 512);  primals_63 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:990 in forward, code: cos = cos.unsqueeze(0).unsqueeze(0)
        unsqueeze_46: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_51, 0);  slice_51 = None
        unsqueeze_47: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 0);  unsqueeze_46 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:991 in forward, code: sin = sin.unsqueeze(0).unsqueeze(0)
        unsqueeze_48: "f16[1, 512, 32]" = torch.ops.aten.unsqueeze.default(slice_52, 0);  slice_52 = None
        unsqueeze_49: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 0);  unsqueeze_48 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:993 in forward, code: x_even = x[..., 0::2]
        slice_53: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_39, 3, 0, 9223372036854775807, 2)
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:994 in forward, code: x_odd = x[..., 1::2]
        slice_54: "f32[16, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_39, 3, 1, 9223372036854775807, 2);  unsqueeze_39 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:996 in forward, code: cos_even = cos[..., 0::2]
        slice_55: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_47, 3, 0, 9223372036854775807, 2);  unsqueeze_47 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:997 in forward, code: sin_even = sin[..., 0::2]
        slice_56: "f16[1, 1, 512, 16]" = torch.ops.aten.slice.Tensor(unsqueeze_49, 3, 0, 9223372036854775807, 2);  unsqueeze_49 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:999 in forward, code: rotated_x_even = x_even * cos_even - x_odd * sin_even
        mul_48: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_53, slice_55)
        mul_49: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_54, slice_56)
        sub_10: "f32[16, 1, 512, 16]" = torch.ops.aten.sub.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1000 in forward, code: rotated_x_odd = x_even * sin_even + x_odd * cos_even
        mul_50: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_53, slice_56);  slice_53 = None
        mul_51: "f32[16, 1, 512, 16]" = torch.ops.aten.mul.Tensor(slice_54, slice_55);  slice_54 = None
        add_21: "f32[16, 1, 512, 16]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1002 in forward, code: rotated = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        unsqueeze_50: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(sub_10, 4);  sub_10 = None
        unsqueeze_51: "f32[16, 1, 512, 16, 1]" = torch.ops.aten.unsqueeze.default(add_21, 4);  add_21 = None
        cat_7: "f32[16, 1, 512, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_50, unsqueeze_51], -1);  unsqueeze_50 = unsqueeze_51 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:1003 in forward, code: rotated = rotated.flatten(-2)
        view_78: "f32[16, 1, 512, 32]" = torch.ops.aten.reshape.default(cat_7, [16, 1, 512, 32]);  cat_7 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_24: "f32[16, 1, 32, 512]" = torch.ops.aten.permute.default(view_78, [0, 1, 3, 2]);  view_78 = None
        expand_12: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_77, [16, 8, 512, 32]);  view_77 = None
        view_79: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_12, [128, 512, 32]);  expand_12 = None
        expand_13: "f32[16, 8, 32, 512]" = torch.ops.aten.expand.default(permute_24, [16, 8, 32, 512]);  permute_24 = None
        clone_4: "f32[16, 8, 32, 512]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_80: "f32[128, 32, 512]" = torch.ops.aten.reshape.default(clone_4, [128, 32, 512]);  clone_4 = None
        bmm_6: "f32[128, 512, 512]" = torch.ops.aten.bmm.default(view_79, view_80)
        view_81: "f32[16, 8, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [16, 8, 512, 512]);  bmm_6 = None
        div_6: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(view_81, 5.656854249492381);  view_81 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:547 in forward, code: attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :q_l, :v_l] == 0, float("-inf"))
        eq_3: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Scalar(primals_64, 0)
        where_3: "f32[16, 8, 512, 512]" = torch.ops.aten.where.self(eq_3, full_default, div_6);  eq_3 = full_default = div_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:548 in forward, code: attn_scores = F.softmax(attn_logits, dim=-1)
        amax_3: "f32[16, 8, 512, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_11: "f32[16, 8, 512, 512]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_3: "f32[16, 8, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_4: "f32[16, 8, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[16, 8, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        expand_14: "f32[16, 8, 512, 512]" = torch.ops.aten.expand.default(div_7, [16, 8, 512, 512])
        view_82: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(expand_14, [128, 512, 512]);  expand_14 = None
        expand_15: "f32[16, 8, 512, 32]" = torch.ops.aten.expand.default(view_76, [16, 8, 512, 32]);  view_76 = None
        view_83: "f32[128, 512, 32]" = torch.ops.aten.reshape.default(expand_15, [128, 512, 32]);  expand_15 = None
        bmm_7: "f32[128, 512, 32]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = None
        view_84: "f32[16, 8, 512, 32]" = torch.ops.aten.reshape.default(bmm_7, [16, 8, 512, 32]);  bmm_7 = None
        view_85: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(view_84, [16, -1, 256]);  view_84 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:115 in forward, code: x = self.attention(q, k, v, _inference=_inference) + x_res
        add_22: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_85, add_18);  view_85 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_8: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_7: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_23: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_7, 1.1920928955078125e-07);  mean_7 = None
        rsqrt_7: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_52: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_22, rsqrt_7)
        mul_53: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, primals_65);  mul_52 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        view_86: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_53, [8192, 256])
        permute_25: "f32[256, 682]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        addmm_21: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_25);  primals_67 = None
        view_87: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_21, [16, 512, 682])
        sigmoid_3: "f32[16, 512, 682]" = torch.ops.aten.sigmoid.default(view_87)
        mul_54: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(view_87, sigmoid_3);  view_87 = sigmoid_3 = None
        permute_26: "f32[256, 682]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        addmm_22: "f32[8192, 682]" = torch.ops.aten.addmm.default(primals_69, view_86, permute_26);  primals_69 = None
        view_89: "f32[16, 512, 682]" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 682])
        mul_55: "f32[16, 512, 682]" = torch.ops.aten.mul.Tensor(mul_54, view_89);  mul_54 = view_89 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        view_90: "f32[8192, 682]" = torch.ops.aten.reshape.default(mul_55, [8192, 682]);  mul_55 = None
        permute_27: "f32[682, 256]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        addmm_23: "f32[8192, 256]" = torch.ops.aten.addmm.default(primals_71, view_90, permute_27);  primals_71 = None
        view_91: "f32[16, 512, 256]" = torch.ops.aten.reshape.default(addmm_23, [16, 512, 256]);  addmm_23 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:118 in forward, code: x = self.swigluNN(x) + x
        add_24: "f32[16, 512, 256]" = torch.ops.aten.add.Tensor(view_91, mul_53);  view_91 = mul_53 = None
        
         # File: /opt/miniconda3/lib/python3.12/site-packages/torch/nn/functional.py:2929 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_9: "f32[16, 512, 256]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_8: "f32[16, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_25: "f32[16, 512, 1]" = torch.ops.aten.add.Scalar(mean_8, 1.1920928955078125e-07);  mean_8 = None
        rsqrt_8: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_56: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_8)
        mul_57: "f32[16, 512, 256]" = torch.ops.aten.mul.Tensor(mul_56, primals_72);  mul_56 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/model.py:167 in forward, code: x = self.linear(x)
        view_92: "f32[8192, 256]" = torch.ops.aten.reshape.default(mul_57, [8192, 256]);  mul_57 = None
        permute_28: "f32[256, 10000]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm_24: "f32[8192, 10000]" = torch.ops.aten.addmm.default(primals_73, view_92, permute_28);  primals_73 = None
        view_93: "f32[16, 512, 10000]" = torch.ops.aten.reshape.default(addmm_24, [16, 512, 10000]);  addmm_24 = None
        permute_29: "f32[10000, 256]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        permute_33: "f32[256, 682]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        permute_37: "f32[682, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_42: "f32[682, 256]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        permute_47: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_48: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_49: "f32[128, 512, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_51: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_55: "f32[32, 256]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        permute_59: "f32[256, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        permute_63: "f32[256, 682]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        permute_67: "f32[682, 256]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_72: "f32[682, 256]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        permute_77: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_78: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_79: "f32[128, 512, 32]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_81: "f32[256, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_85: "f32[32, 256]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        permute_93: "f32[256, 682]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        permute_97: "f32[682, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_102: "f32[682, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        permute_107: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_108: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_109: "f32[128, 512, 32]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_115: "f32[32, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:821 in forward, code: x = self.linear_out(x)
        permute_123: "f32[256, 682]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:820 in forward, code: x = torch.mul(F.silu(self.swiglu_linear(x)), self.swiglu_gate_linear(x))
        permute_127: "f32[682, 256]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_132: "f32[682, 256]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:549 in forward, code: attn_output = torch.matmul(attn_scores, v).view(b, -1, d_model)
        permute_137: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:546 in forward, code: attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        permute_138: "f32[128, 32, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_139: "f32[128, 512, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:96 in forward, code: v = self.linearV(x)
        permute_141: "f32[256, 256]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:95 in forward, code: k = self.linearK(x)
        permute_145: "f32[32, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/DEEPLEARNING/BUILDING/HLLM/model/blocks.py:94 in forward, code: q = self.linearQ(x)
        permute_149: "f32[256, 256]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (view_93, convert_element_type_1, convert_element_type, convert_element_type_3, convert_element_type_2, convert_element_type_5, convert_element_type_4, convert_element_type_7, convert_element_type_6, primals_2, primals_3, primals_4, primals_13, primals_14, primals_21, primals_30, primals_31, primals_38, primals_47, primals_48, primals_55, primals_64, primals_65, primals_72, embedding, rsqrt, view, slice_7, slice_8, slice_13, slice_14, div_1, add_4, rsqrt_1, view_17, addmm_3, addmm_4, view_21, add_6, rsqrt_2, view_23, slice_21, slice_22, slice_27, slice_28, div_3, add_10, rsqrt_3, view_40, addmm_9, addmm_10, view_44, add_12, rsqrt_4, view_46, slice_35, slice_36, slice_41, slice_42, div_5, add_16, rsqrt_5, view_63, addmm_15, addmm_16, view_67, add_18, rsqrt_6, view_69, slice_49, slice_50, slice_55, slice_56, div_7, add_22, rsqrt_7, view_86, addmm_21, addmm_22, view_90, add_24, rsqrt_8, view_92, permute_29, permute_33, permute_37, permute_42, permute_47, permute_48, permute_49, permute_51, permute_55, permute_59, permute_63, permute_67, permute_72, permute_77, permute_78, permute_79, permute_81, permute_85, permute_89, permute_93, permute_97, permute_102, permute_107, permute_108, permute_109, permute_111, permute_115, permute_119, permute_123, permute_127, permute_132, permute_137, permute_138, permute_139, permute_141, permute_145, permute_149)
        