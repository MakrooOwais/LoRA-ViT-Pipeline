# **LoRA for Vision Transformers: Training and Inference Pipelines**  

This repository provides a comprehensive solution for lightweight adaptation of Vision Transformers (ViTs) using Low-Rank Adaptation (LoRA). The repository is split into two modules:  

- **Training Pipeline (`lora_train_pipeline.py`)**: A framework to train LoRA adapters and save them for downstream tasks.  
- **Inference Pipeline (`lora_inference_pipeline.py`)**: A system optimized for task-specific inference using multiple sets of LoRA weights in memory.  

## **Why LoRA?**  

Fine-tuning large models, such as Vision Transformers, for different tasks is computationally expensive. LoRA (Low-Rank Adaptation) is a parameter-efficient approach that adapts pre-trained models for new tasks without modifying the original weights. Instead, LoRA introduces low-rank matrices into the architecture, significantly reducing the number of trainable parameters while maintaining high performance.  

---

## **How LoRA Works**  


LoRA assumes that weight updates in pre-trained models are low-rank in nature. For a target weight matrix W:  

$$ W' = W + \alpha \cdot AB $$

where A  and B are low-rank matrices.

By focusing only on A and B, LoRA reduces the number of trainable parameters from $$ d \times k  $$  to  $$ r \cdot (d + k) $$.  

This approach makes LoRA ideal for resource-constrained environments, where fine-tuning the full model is impractical.  

---

## **Repository Overview**  

### **File Structure**  

```
LoRA-ViT-Pipeline/
├── lora_train_pipeline.py       # Training pipeline for LoRA adapters
├── lora_inference_pipeline.py   # Inference pipeline for multiple LoRA adapters
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

### **Components**  

#### **1. Training Pipeline: `lora_train_pipeline.py`**  

This module enables the training of LoRA adapters for Vision Transformers. It includes:  

- **`ViTAdapterConfig`**: A configuration class that specifies LoRA settings, including:  
  - Adapter type (`LORA` by default).  
  - Rank (r), scaling factor α, and dropout.
  - Target modules in the transformer (e.g., query, key, value, proj).  
- **`LoRALayer`**: Implements low-rank adaptations for selected layers in the ViT.  
- **`ViTFeatureExtractor`**: Provides a backbone that integrates LoRA adapters into a Vision Transformer. It supports handling one set of LoRA weights at a time, making it suitable for training.  

**Workflow**:  
1. Train LoRA adapters using the `ViTFeatureExtractor`.  
2. Save the trained weights to a directory specified by the user.  

#### **2. Inference Pipeline: `lora_inference_pipeline.py`**  

This module is optimized for task-specific inference with multiple sets of LoRA weights. It includes:  

- **`ViTAdapterConfig`**: Same as in the training pipeline.  
- **`MultiLoRALayer`**: Extends `LoRALayer` to support multiple LoRA weight sets in memory simultaneously.  
- **`ViTFeatureExtractor`**: Modified for inference to dynamically load and switch between multiple sets of LoRA weights.  

**Key Feature**:  
The `load_all_adapters` function loads all LoRA weights from a specified directory into memory. This reduces I/O overhead during inference and enables seamless task switching.  

---

## **Getting Started**  

### **Prerequisites**  

- Python >= 3.8  
- PyTorch >= 1.10  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

---

### **Training LoRA Weights**  

1. Set up the `ViTFeatureExtractor` and LoRA configuration in `lora_train_pipeline.py`.  
2. Train the model:  
   ```python
   from lora_train_pipeline import ViTFeatureExtractor, ViTAdapterConfig

   config = ViTAdapterConfig(adapter_type="LORA", r=16, alpha=32, dropout=0.1)
   vit_model = ViTFeatureExtractor(config=config)

   # Training code here
   vit_model.train(...)
   ```
3. Save the trained LoRA weights:  
   ```python
   vit_model.save_adapters("/path/to/adapter/directory/")
   ```

---

### **Inference with Multiple LoRA Weights**  

1. Load all LoRA weights into memory using the `load_all_adapters` function:  
   ```python
   from lora_inference_pipeline import ViTFeatureExtractor

   vit_model = ViTFeatureExtractor()
   vit_model.load_all_adapters("/path/to/adapter/directory/")
   ```
2. Perform inference with the desired LoRA weight:  
   ```python
   vit_model.set_active_adapter("task1_adapter")
   predictions = vit_model.infer(input_data)
   ```

---

## **Performance**  

LoRA significantly reduces the memory and compute requirements for fine-tuning Vision Transformers. By keeping multiple LoRA weights in memory, the inference pipeline minimizes I/O overhead and achieves faster task switching.  

---

## **Future Scope**  

- Support for additional adapter types (e.g., prefix tuning).  
- Benchmarks across diverse datasets and tasks.  
- Integration with other pre-trained ViT architectures.  

---

## **License**  

This project is licensed under the MIT License.  

--- 

Let me know if you need further customization!