from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_athena import AthenaConfig
from .modeling_athena import AthenaForCausalLM

# Register the configuration
AutoConfig.register("athena", AthenaConfig)

# Register the model
AutoModelForCausalLM.register(AthenaConfig, AthenaForCausalLM)

# Make classes available for import
__all__ = ["AthenaConfig", "AthenaForCausalLM"]