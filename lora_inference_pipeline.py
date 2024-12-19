import torch
import torch.nn as nn
import math
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
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


class MultiLoRALayer(nn.Module):
    """Implementation of Multi-LoRA layer supporting multiple sets of weights"""

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
        self.in_features = in_features
        self.out_features = out_features

        # Dictionary to store multiple LoRA weights
        self.lora_weights: Dict[str, Tuple[nn.Linear, nn.Linear]] = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

    def add_adapter(self, adapter_name: str):
        """Add a new set of LoRA weights"""
        if adapter_name not in self.lora_weights:
            lora_down = nn.Linear(self.in_features, self.r, bias=False)
            lora_up = nn.Linear(self.r, self.out_features, bias=False)

            # Initialize weights
            nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_up.weight)

            self.lora_weights[adapter_name] = nn.ModuleDict(
                {"down": lora_down, "up": lora_up}
            )

    def forward(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        if adapter_name not in self.lora_weights:
            raise ValueError(f"Adapter {adapter_name} not found")

        lora = self.lora_weights[adapter_name]
        return lora["up"](self.dropout(lora["down"](x))) * self.scaling


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer model with feature extraction capabilities and support for
    multiple parameter-efficient fine-tuning methods.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: ViTAdapterConfig = ViTAdapterConfig(),
        default_adapter_name: str = "default",
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.default_adapter_name = default_adapter_name
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
        """Create a new module with Multi-LoRA adaptation"""
        in_features = module.in_features
        out_features = module.out_features

        new_module = nn.Module()
        new_module.original = module
        new_module.lora = MultiLoRALayer(
            in_features,
            out_features,
            self.adapter_config.r,
            self.adapter_config.alpha,
            self.adapter_config.dropout,
        )
        # Add default adapter
        new_module.lora.add_adapter(self.default_adapter_name)

        def forward(self, x):
            # Get adapter name from the parent module's context
            adapter_name = getattr(self, '_current_adapter_name', self.default_adapter_name)
            return self.original(x) + self.lora(x, adapter_name)

        new_module.forward = forward.__get__(new_module)
        new_module.default_adapter_name = self.default_adapter_name
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

    def add_adapter(self, adapter_name: str):
        """Add a new adapter to all LoRA layers"""
        for adapter in self.adapters.values():
            adapter.add_adapter(adapter_name)

    def save_adapter(
        self,
        save_directory: Union[str, Path],
        adapter_name: str,
        save_config: bool = True,
    ) -> None:
        """Save specific LoRA adapter weights and configuration"""
        save_directory = Path(save_directory) / adapter_name
        os.makedirs(save_directory, exist_ok=True)

        # Create weights dict
        lora_state_dict = {}

        # Extract LoRA weights for specific adapter
        for name, module in self.adapters.items():
            if isinstance(module, MultiLoRALayer):
                if adapter_name in module.lora_weights:
                    adapter = module.lora_weights[adapter_name]
                    lora_state_dict[f"{name}.lora_up.weight"] = adapter[
                        "up"
                    ].weight.data.cpu()
                    lora_state_dict[f"{name}.lora_down.weight"] = adapter[
                        "down"
                    ].weight.data.cpu()
                    lora_state_dict[f"{name}.scaling"] = module.scaling

        # Save weights with default name
        weights_path = save_directory / "default_lora_weights.pt"
        torch.save(lora_state_dict, weights_path)

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

            config_path = save_directory / "default_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

    def load_all_adapters(
        self,
        base_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Load all adapters from a directory structure where each subdirectory contains
        an adapter's weights and config files.

        Directory structure:
        base_path/
            adapter1/
                default_lora_weights.pt
                default_config.json
            adapter2/
                default_lora_weights.pt
                default_config.json
            ...
        """
        base_path = Path(base_path)
        if not base_path.exists():
            raise ValueError(f"Base path {base_path} does not exist")

        if device is None:
            device = next(self.parameters()).device

        # Iterate through subdirectories
        for adapter_dir in base_path.iterdir():
            if adapter_dir.is_dir():
                adapter_name = adapter_dir.name
                weights_path = adapter_dir / "default_lora_weights.pt"
                config_path = adapter_dir / "default_config.json"

                if not weights_path.exists():
                    print(f"Warning: No weights file found for adapter {adapter_name}")
                    continue

                # Load config if it exists
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    config_dict["target_modules"] = tuple(config_dict["target_modules"])
                    self.adapter_config = ViTAdapterConfig(**config_dict)

                # Load weights
                self.load_adapter(
                    weights_path=weights_path, adapter_name=adapter_name, device=device
                )
                print(f"Loaded adapter: {adapter_name}")

    def load_adapter(
        self,
        weights_path: Union[str, Path],
        adapter_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load LoRA adapter weights for a specific adapter"""
        weights_path = Path(weights_path)
        lora_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        if adapter_name is None:
            adapter_name = weights_path.parent.name

        # Add new adapter if it doesn't exist
        self.add_adapter(adapter_name)

        if device is None:
            device = next(self.parameters()).device

        for name, module in self.adapters.items():
            if isinstance(module, MultiLoRALayer):
                if f"{name}.lora_up.weight" in lora_state_dict:
                    module.lora_weights[adapter_name]["up"].weight.data = (
                        lora_state_dict[f"{name}.lora_up.weight"].to(device)
                    )
                if f"{name}.lora_down.weight" in lora_state_dict:
                    module.lora_weights[adapter_name]["down"].weight.data = (
                        lora_state_dict[f"{name}.lora_down.weight"].to(device)
                    )
                if f"{name}.scaling" in lora_state_dict:
                    module.scaling = lora_state_dict[f"{name}.scaling"]

    def forward(
        self,
        pixel_values: torch.Tensor,
        adapter_name: Optional[str] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass with specified adapter"""
        # Store the current adapter name in a context that wrapped modules can access
        self._current_adapter_name = adapter_name or self.default_adapter_name
        
        # Set adapter name for all wrapped modules
        for module in self.modules():
            if hasattr(module, 'lora'):
                module._current_adapter_name = self._current_adapter_name

        # No need to modify forward methods anymore
        outputs = self.base_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    def get_trainable_parameters(
        self, adapter_name: Optional[str] = None
    ) -> Tuple[int, int]:
        """Get the number of trainable parameters for a specific adapter"""
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
