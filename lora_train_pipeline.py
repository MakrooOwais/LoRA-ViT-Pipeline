import torch
import torch.nn as nn
import math
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ViTAdapterConfig:
    """Configuration class for ViT Adapter"""

    adapter_type: str = "LORA"  # Type of adapter (e.g., LORA, PREFIX)
    r: int = 16  # Rank of the adapter
    alpha: float = 32  # Scaling factor
    dropout: float = 0.05  # Dropout probability
    target_modules: Tuple[str, ...] = (
        "query",
        "key",
        "value",
        "proj",
    )  # Modules to apply adaptation
    bias: str = "none"  # Bias type
    inference_mode: bool = False  # Whether to use inference mode
    num_virtual_tokens: int = 0  # Number of virtual tokens for prefix tuning


class LoRALayer(nn.Module):
    """Implementation of LoRA (Low-Rank Adaptation) layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA layers
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer model with feature extraction capabilities and support for
    parameter-efficient fine-tuning methods.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: ViTAdapterConfig,
        adapter_name: str = "default",
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.adapter_name = adapter_name
        self.active_adapter = adapter_name
        self.config = base_model.config

        # Store original forward methods
        self.forward_methods = {}

        # Initialize adapters
        self.adapters = nn.ModuleDict()
        if adapter_config.adapter_type == "LORA":
            self._init_lora_adapters()

    def _find_modules(self, module, target_modules, prefix=""):
        """Recursively find target modules and return their full names and references"""
        modules = {}
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if any(target in name for target in target_modules):
                if isinstance(child, nn.Linear):
                    modules[full_name] = child
            modules.update(self._find_modules(child, target_modules, full_name))
        return modules

    def _create_new_module(self, module):
        """Create a new module with LoRA adaptation"""
        in_features = module.in_features
        out_features = module.out_features

        new_module = nn.Module()
        new_module.original = module
        new_module.lora = LoRALayer(
            in_features,
            out_features,
            self.adapter_config.r,
            self.adapter_config.alpha,
            self.adapter_config.dropout,
        )

        def forward(self, x):
            return self.original(x) + self.lora(x)

        new_module.forward = forward.__get__(new_module)
        return new_module

    def _init_lora_adapters(self):
        """Initialize LoRA adapters for target modules"""
        modules_to_adapt = self._find_modules(
            self.base_model, self.adapter_config.target_modules
        )

        for name, module in modules_to_adapt.items():
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent_module = self.base_model

            if parent_name:
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)

            wrapped_module = self._create_new_module(module)
            setattr(parent_module, child_name, wrapped_module)
            self.adapters[name.replace(".", "_")] = wrapped_module.lora

    def save_adapter(
        self,
        save_directory: Union[str, Path],
        adapter_name: Optional[str] = None,
        save_config: bool = True,
    ) -> None:
        """
        Save LoRA adapter weights and configuration to a directory.

        Args:
            save_directory: Directory to save the weights and config
            adapter_name: Name of the adapter to save. If None, saves the active adapter
            save_config: Whether to save the adapter configuration
        """
        if adapter_name is None:
            adapter_name = self.active_adapter

        save_directory = Path(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        # Create weights dict
        lora_state_dict = {}

        # Extract LoRA weights from model
        for name, module in self.adapters.items():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_up.weight"] = (
                    module.lora_up.weight.data.cpu()
                )
                lora_state_dict[f"{name}.lora_down.weight"] = (
                    module.lora_down.weight.data.cpu()
                )
                lora_state_dict[f"{name}.scaling"] = module.scaling

        # Save weights
        weights_path = save_directory / f"{adapter_name}_lora_weights.pt"
        torch.save(lora_state_dict, weights_path)

        # Save config if requested
        if save_config:
            config_dict = {
                "adapter_type": self.adapter_config.adapter_type,
                "r": self.adapter_config.r,
                "alpha": self.adapter_config.alpha,
                "dropout": self.adapter_config.dropout,
                "target_modules": list(self.adapter_config.target_modules),
                "bias": self.adapter_config.bias,
                "inference_mode": self.adapter_config.inference_mode,
                "num_virtual_tokens": self.adapter_config.num_virtual_tokens,
            }

            config_path = save_directory / f"{adapter_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

    def load_adapter(
        self,
        weights_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        adapter_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Load LoRA adapter weights and optionally configuration from files.

        Args:
            weights_path: Path to the saved weights file
            config_path: Optional path to the saved configuration file
            adapter_name: Name to give to the loaded adapter. If None, uses the name from saving
            device: Device to load the weights to. If None, uses the current model device
        """
        weights_path = Path(weights_path)

        # Load weights
        lora_state_dict = torch.load(weights_path, map_location="cpu")

        # Get adapter name from filename if not provided
        if adapter_name is None:
            adapter_name = weights_path.stem.replace("_lora_weights", "")

        # Load config if provided
        if config_path is not None:
            config_path = Path(config_path)
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            config_dict["target_modules"] = tuple(config_dict["target_modules"])
            self.adapter_config = ViTAdapterConfig(**config_dict)

        # Get device
        if device is None:
            device = next(self.parameters()).device

        # Load weights into model
        for name, module in self.adapters.items():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_up.weight" in lora_state_dict:
                    module.lora_up.weight.data = lora_state_dict[
                        f"{name}.lora_up.weight"
                    ].to(device)
                if f"{name}.lora_down.weight" in lora_state_dict:
                    module.lora_down.weight.data = lora_state_dict[
                        f"{name}.lora_down.weight"
                    ].to(device)
                if f"{name}.scaling" in lora_state_dict:
                    module.scaling = lora_state_dict[f"{name}.scaling"]

    def merge_and_unload(self) -> None:
        """
        Merge LoRA weights into the base model and remove LoRA modules.
        """
        # Find all modules with LoRA
        for name, wrapped_module in self.named_modules():
            if hasattr(wrapped_module, "original") and hasattr(wrapped_module, "lora"):
                # Compute the merged weights
                original_weight = wrapped_module.original.weight.data
                lora_weights = wrapped_module.lora(
                    torch.eye(wrapped_module.original.in_features).to(
                        original_weight.device
                    )
                )
                merged_weights = original_weight + lora_weights

                # Update the original module weights
                wrapped_module.original.weight.data = merged_weights

                # Remove LoRA by replacing wrapped module with original
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = self

                if parent_name:
                    for part in parent_name.split("."):
                        parent_module = getattr(parent_module, part)

                setattr(parent_module, child_name, wrapped_module.original)

        # Clear adapters
        self.adapters.clear()

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # Forward pass through base model
        outputs = self.base_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
