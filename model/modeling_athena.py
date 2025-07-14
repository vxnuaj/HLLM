import torch
import torch.nn as nn
import warnings
import sys
import os
from typing import Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'blocks')))

from blocks import TransformerBlock, PositionalEmbedding
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_athena import AthenaConfig

class AthenaForCausalLM(PreTrainedModel):
    """
    The Athena architecture for causal language modeling, supporting various attention types and positional embeddings.
    
    This model inherits from PreTrainedModel and is compatible with HuggingFace's AutoModelForCausalLM.
    """
    
    config_class = AthenaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    
    def __init__(self, config: AthenaConfig):
        """
        Initialize the AthenaForCausalLM model.
        
        Args:
            config (AthenaConfig): Model configuration containing all hyperparameters.
        """
        super().__init__(config)
        self.config = config
        
        # Initialize model components
        self._init_model_components()
        
        # Initialize weights
        self.post_init()
    
    def _init_model_components(self):
        """Initialize all model components based on the configuration."""
        self._supress_warnings(self.config.supress_warnings)
        self._check_pos_emb_type(self.config.pos_emb_type)
        
        # Store config values as instance variables for backward compatibility
        self.context_len = self.config.context_len
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.n_blocks = self.config.n_blocks
        self.vocab_size = self.config.vocab_size
        self.pos_emb_dropout_p = self.config.pos_emb_dropout_p
        self.pos_emb_type = self.config.pos_emb_type
        self.learned = self.config.learned
        self.ntk_rope_scaling = self.config.ntk_rope_scaling
        self.dyn_scaling = self.config.dyn_scaling
        self.attn_type = self.config.attn_type
        self.n_groups = self.config.n_groups
        self.top_k_sparsev = self.config.top_k_sparsev
        self.p_threshold = self.config.p_threshold
        self.p_threshold_steps_fraction = self.config.p_threshold_steps_fraction
        self.flash_attn = self.config.flash_attn
        self.flash_attn_dtype = getattr(torch, self.config.flash_attn_dtype)
        self.model_name = self.config.model_name
        self.verbose = self.config.verbose
        
        # Model layers
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )
        
        if self.pos_emb_type == 'pe':
            self.pe = PositionalEmbedding(
                context_len=self.context_len,
                d_model=self.d_model,
                dropout_p=self.pos_emb_dropout_p,
                learned=self.learned
            )
        
        self.block = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                context_len=self.context_len,
                ntk_rope_scaling=self.ntk_rope_scaling,
                dyn_scaling=self.dyn_scaling,
                attn_type=self.attn_type,
                n_groups=self.n_groups,
                top_k_sparsev=self.top_k_sparsev,
                p_threshold=self.p_threshold,
                p_threshold_steps_fraction=self.p_threshold_steps_fraction,
                flash_attn=self.flash_attn,
                flash_attn_dtype=self.flash_attn_dtype
            )
            for _ in range(self.n_blocks)
        ])
        
        self.rmsnorm = nn.RMSNorm(normalized_shape=self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tie weights
        self.linear.weight = self.embeddings.weight
    
    def get_input_embeddings(self):
        """Return the input embeddings layer."""
        return self.embeddings
    
    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.embeddings = value
    
    def get_output_embeddings(self):
        """Return the output embeddings layer."""
        return self.linear
    
    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings layer."""
        self.linear = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the AthenaForCausalLM model.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask.
            past_key_values: Past key values for faster generation.
            inputs_embeds (torch.FloatTensor): Input embeddings.
            labels (torch.LongTensor): Labels for computing the loss.
            use_cache (bool): Whether to use caching for generation.
            output_attentions (bool): Whether to output attention weights.
            output_hidden_states (bool): Whether to output hidden states.
            return_dict (bool): Whether to return a dictionary.
            
        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Determine input
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            x = self.embeddings(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            x = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Apply positional embeddings
        if self.pos_emb_type == 'pe':
            x = self.pe(x, _inference=use_cache)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.block):
            x = block(x, _inference=use_cache)
        
        # Apply final layer norm and output projection
        x = self.rmsnorm(x)
        logits = self.linear(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        This function is used to re-order the past_key_values cache if beam search is used.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _check_pos_emb_type(self, pos_emb_type: str):
        """Validate the positional embedding type."""
        assert isinstance(pos_emb_type, str), "pos_emb_type should be a string"
        if pos_emb_type not in ["rope", "pe"]:
            raise ValueError('pos_emb_type should be either "rope" or "pe"')
        if pos_emb_type == 'rope':
            warnings.warn("Using rotary positional embedding, learned will have no effect")
    
    def _supress_warnings(self, supress_warnings: bool):
        """Configure warning suppression."""
        assert isinstance(supress_warnings, bool), ValueError("supress_warnings must be type bool")
        if supress_warnings:
            warnings.filterwarnings("ignore")
