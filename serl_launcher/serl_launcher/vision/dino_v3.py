from typing import Optional
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
from .resnet_v1 import SpatialLearnedEmbeddings

# Global variables to cache the DINOv3 model and processor
_dinov3_model_cache = {}
_dinov3_processor_cache = {}
# Cache for output shapes: maps (height, width) -> num_tokens
_dinov3_output_shape_cache = {}

def _get_dinov3_model(model_path: str):
    """Get or load DINOv3 model (cached)."""
    if model_path not in _dinov3_model_cache:
        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch
            
            # Load model and processor
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            model.eval()  # Set to evaluation mode
            
            try:
                model = torch.compile(model)
                print("DINOv3 model compiled with torch.compile")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
            
            _dinov3_model_cache[model_path] = model
            _dinov3_processor_cache[model_path] = processor
        except ImportError:
            raise ImportError(
                "transformers and torch libraries are required for DINOv3. "
                "Please install them with: pip install transformers torch"
            )
    
    return _dinov3_model_cache[model_path], _dinov3_processor_cache[model_path]

class PreTrainedDinoV3Encoder(nn.Module):
    """
    DINOv3 Vision Transformer Encoder wrapper, similar to PreTrainedResNetEncoder.
    Uses transformers library to load DINOv3 model, extract features,
    and convert to JAX format for downstream Flax modules.
    """
    pooling_method: str = "flatten_register_tokens"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    bottleneck_dim: Optional[int] = None
    model_path: str = "/home/turingzero/models/dinov3-vitb16-pretrain-lvd1689m"
    num_spatial_blocks: int = 8

    def _extract_features_python(self, images_np: jnp.ndarray) -> jnp.ndarray:
        """
        Extract features from images using DINOv3 model (Python function for callback).
        
        Args:
            images_np: NumPy array or JAX array of shape (B, H, W, C) with values in [0, 255]
            
        Returns:
            NumPy array of features
        """
        import torch

        from PIL import Image

        if not isinstance(images_np, np.ndarray):
            images_np = np.asarray(images_np)

        
        # Convert each image in the batch to PIL Image
        # print(f"images_np.shape: {images_np.shape}")
        pil_images = []
        for i in range(images_np.shape[0]):
            img = images_np[i]  # (H, W, C)
            pil_img = Image.fromarray(img, mode='RGB')
            pil_images.append(pil_img)
        
        # Get model and processor (cached)
        model, processor = _get_dinov3_model(self.model_path)
        device = next(model.parameters()).device
        
        inputs = processor(images=pil_images, return_tensors="pt").to(device)

        with torch.inference_mode():
            # Extract features
            outputs = model(**inputs)
            
            # Get last hidden state: shape (B, num_patches + 1, hidden_size)
            if self.pooling_method == "pooler_output":
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state # （B, 1 cls + 4 register + 196 patches, 768）

            features = features.to(torch.float32)
            
            # Convert to numpy
            features = features.cpu().numpy()

        return features

    def _extract_features(self, images: jnp.ndarray) -> jnp.ndarray:
        """
        Extract features from images using DINOv3 model.
        Uses JAX pure_callback to handle PyTorch calls in JIT-compiled code.
        
        Note: This function uses PyTorch internally, so it must be called via
        JAX callback when used in JIT-compiled functions.
        
        Args:
            images: JAX array of shape (B, H, W, C) or (H, W, C) with values in [0, 255]
            
        Returns:
            JAX array of features
        """
        # Determine batch size based on input shape
        # If input is 3D (H, W, C), batch_size will be 1 (handled in Python function)
        # If input is 4D (B, H, W, C), use the first dimension
        # Batch of images
        if images.ndim == 3:
            images = images[None]

        batch_size = images.shape[0]
        
        hidden_size = 768  # ViT-B hidden size
        
        # Use JAX callback to call Python function (allows PyTorch in JIT)
        # pure_callback will handle the conversion from JAX array to NumPy array
        from jax import pure_callback
        
        output_shape = jax.ShapeDtypeStruct((batch_size, hidden_size), jnp.float32)
        
        # Use pure_callback to call Python function
        # pure_callback will pass the JAX array to the Python function,
        # which will convert it to NumPy internally and handle batch dimension
        features_jax = pure_callback(
            self._extract_features_python,
            output_shape,
            images,
        )
            
        return features_jax

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
    ):
        """
        Encode observations using DINOv3.
        
        Args:
            observations: Input images of shape (B, H, W, C) or (H, W, C)
                          with values in [0, 255]
            encode: Whether to encode with pretrained encoder
            train: Whether in training mode (not used for DINOv3 as it's frozen)
        """
        x = observations
        
        if encode:
            # Extract features using DINOv3
            # Output shape: (B, num_patches + 1, hidden_size)
            # For ViT-B/16: (B, 201, 768) where 201 = 1 CLS token + 4 register token + 196 patches
            x = self._extract_features(x)

        if self.pooling_method == "pooler_output":
            pass
        elif self.pooling_method == "spatial_learned_embeddings":
            # remove cls token and register tokens and reshape B x 14 x 14 x C
            x = x[:, 5:, :]
            x = x.reshape(x.shape[0], 14, 14, x.shape[-1])
            x = SpatialLearnedEmbeddings(
                height=14,
                width=14,
                channel=x.shape[-1],
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
        elif self.pooling_method == "mean_register_tokens":
            x = x[:, 1:5, :]
            x = jnp.mean(x, axis=1)
        elif self.pooling_method == "flatten_register_tokens":
            x = x[:, 1:5, :]
            x = x.reshape(x.shape[0], -1)
        else:
            raise ValueError("pooling method not found")


        # Apply bottleneck if specified
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x