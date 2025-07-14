from transformers import PretrainedConfig
from typing import Union

class AthenaConfig(PretrainedConfig):
    """
    Configuration class for Athena model.
    
    This is the configuration class to store the configuration of a Athena model.
    It is used to instantiate a Athena model according to the specified arguments.
    """
    
    model_type = "athena"
    
    def __init__(
        self,
        context_len: int = 2048,
        d_model: int = 4096,
        n_heads: int = 32,
        n_blocks: int = 32,
        vocab_size: int = 32000,
        pos_emb_dropout_p: float = 0.1,
        pos_emb_type: str = "rope",
        learned: bool = False,
        ntk_rope_scaling: Union[dict, bool] = False,
        dyn_scaling: Union[bool, float] = False,
        attn_type: str = "gqa",
        n_groups: int = None,
        top_k_sparsev: int = None,
        p_threshold: int = None,
        p_threshold_steps_fraction: float = None,
        flash_attn: bool = False,
        flash_attn_dtype: str = "float16",
        supress_warnings: bool = True,
        verbose: bool = False,
        model_name: str = None,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        """
        Initialize AthenaConfig.
        
        Args:
            context_len (int): Maximum sequence length the model can process.
            d_model (int): Dimensionality of the input and output features.
            n_heads (int): Number of attention heads in each transformer block.
            n_blocks (int): Number of transformer blocks.
            vocab_size (int): Size of the vocabulary for token embeddings.
            pos_emb_dropout_p (float): Dropout probability for positional embeddings.
            pos_emb_type (str): Type of positional embedding ('rope' or 'pe').
            learned (bool): Whether to use learned positional embeddings.
            ntk_rope_scaling (Union[dict, bool]): Configuration for NTK RoPE scaling.
            dyn_scaling (Union[bool, float]): Dynamic scaling factor for RoPE.
            attn_type (str): Type of attention mechanism ('mhsa', 'mqa', or 'gqa').
            n_groups (int): Number of groups for grouped query attention.
            top_k_sparsev (int): Top-k sparsity parameter.
            p_threshold (int): Threshold parameter.
            p_threshold_steps_fraction (float): Threshold steps fraction.
            flash_attn (bool): Whether to use FlashAttention.
            flash_attn_dtype (str): Data type for FlashAttention.
            supress_warnings (bool): Whether to suppress warnings.
            verbose (bool): Whether to enable verbose logging.
            model_name (str): Name of the model.
            pad_token_id (int): Token ID for padding.
            bos_token_id (int): Token ID for beginning of sequence.
            eos_token_id (int): Token ID for end of sequence.
            **kwargs: Additional keyword arguments.
        """
        self.context_len = context_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.pos_emb_dropout_p = pos_emb_dropout_p
        self.pos_emb_type = pos_emb_type
        self.learned = learned
        self.ntk_rope_scaling = ntk_rope_scaling
        self.dyn_scaling = dyn_scaling
        self.attn_type = attn_type
        self.n_groups = n_groups
        self.top_k_sparsev = top_k_sparsev
        self.p_threshold = p_threshold
        self.p_threshold_steps_fraction = p_threshold_steps_fraction
        self.flash_attn = flash_attn
        self.flash_attn_dtype = flash_attn_dtype
        self.supress_warnings = supress_warnings
        self.verbose = verbose
        self.model_name = model_name
        
        # Set these for HuggingFace compatibility
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_blocks
        self.max_position_embeddings = context_len
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
