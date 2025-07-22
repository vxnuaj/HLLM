### Hyperparameter Sweep over Inference Pass.

- Mixed Precision Inference ( not during quantization)
    - fp16
    - bfloat16
    - fp8
    - bfloat8
- Precision during Quantization
	- float32
	- float16
	- float8
	- bfloat16
	- bfloat8
	- int8
	- int4
- Quantization method(s)
	- GPTQ
	- AWQ
	- GGUF
	- EXL12
	- BitsAndBytes
	- Quantize the KV Kache
- Model Execution Backend
	- torch.compile
		- inductor
		- eager 
		- aot_eager
	- onnx runtime
	- tensorrt
	- vllm
- Decoding
	- Speculative Decoding ( distill the model into a model that’s 4x smaller ).


### Quantize

If you're going to run awq, start a venv and then `pip install -e package/llm-awq`.

```zsh
python -m venv awq_venv
source awq_venv/bin/activate
pip install -e package/llm-awq
```

then

```zsh
cd awq/kernels
python setup.py install
```

not needed for other quantization methods.


### TODO

##### `quantize.py`

- [ ] Convert weights to hf format
- [ ] Verify GPTq works
- [ ] Verify AWQ works
- [ ] Verify Quanto works
- [ ] Verify AQLM works
- [ ] Verify VPTq works
- [ ] Verify hqq works
- [ ] Verify bitsandbytes works
- [ ] Verify spqr works