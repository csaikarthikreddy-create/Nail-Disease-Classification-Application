"""
Disease classification module for nail conditions
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
from collections import OrderedDict


class DiseaseClassifier:
    """Disease classifier for nail conditions"""
    
    # Disease classes (matching the training data)
    CLASSES = [
        'Acral_Lentiginous_Melanoma',
        'Healthy_Nail',
        'Onychogryphosis',
        'blue_finger',
        'clubbing',
        'pitting'
    ]
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the disease classifier
        
        Args:
            model_path: Path to trained model weights. If None, uses pretrained DenseNet201
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.CLASSES)
        
        # Initialize model architecture
        self.model = self._create_model()
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                # weights_only=False is needed for PyTorch 2.6+ when loading models with custom classes
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # Handle both checkpoint format and state_dict format
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model checkpoint from {model_path}")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    print(f"Loaded model checkpoint from {model_path}")
                else:
                    # Assume it's a state_dict directly
                    self.model.load_state_dict(checkpoint)
                    print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model weights from {model_path}: {e}")
                print("Using randomly initialized model.")
        else:
            if model_path:
                print(f"Warning: Model path specified but file not found: {model_path}")
            print("Warning: No model weights provided. Using randomly initialized model.")
            print("For production use, please train and save a model first.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup Grad-CAM hooks
        self.gradcam_activations = {}
        self.gradcam_gradients = {}
        self._setup_gradcam_hooks()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _create_model(self):
        """Create DenseNet201 model with custom classifier"""
        # Load pretrained DenseNet201
        model = models.densenet201(weights='DEFAULT')
        
        # Freeze feature extractor
        for param in model.parameters():
            param.requires_grad = False
        
        # Custom classifier (matching training architecture)
        hidden_units = 1024
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1920, hidden_units)),  # DenseNet201 features output 1920
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, 512)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(512, self.num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.classifier = classifier
        return model
    
    def _setup_gradcam_hooks(self):
        """Setup forward and backward hooks for Grad-CAM on DenseNet201"""
        try:
            # Target layer: last dense block in DenseNet201 (features.denseblock4)
            # We'll hook into the last convolution layer before global pooling
            # Try different possible layer paths
            target_layer = None
            
            # Try the most common path first
            try:
                target_layer = self.model.features.denseblock4.denselayer24.conv2
            except AttributeError:
                # Try alternative paths
                try:
                    # If denselayer24 doesn't exist, try to find the last layer
                    denseblock4 = self.model.features.denseblock4
                    # Get the last dense layer
                    last_layer_name = list(denseblock4.named_children())[-1][0]
                    last_layer = getattr(denseblock4, last_layer_name)
                    target_layer = last_layer.conv2
                except (AttributeError, IndexError):
                    # Fallback: use the norm5 layer (last normalization before pooling)
                    try:
                        target_layer = self.model.features.norm5
                    except AttributeError:
                        # Last resort: use the last feature layer
                        target_layer = list(self.model.features.children())[-1]
            
            if target_layer is None:
                raise AttributeError("Could not find suitable target layer for Grad-CAM")
            
            def _forward_hook(module, inp, out):
                """Save activations during forward pass"""
                # Store activations (detach to avoid memory issues)
                self.gradcam_activations['value'] = out.detach().clone()
            
            # Register forward hook to capture activations
            self.gradcam_handle = target_layer.register_forward_hook(_forward_hook)
            
            # Register backward hook to capture gradients (this works even if tensor doesn't require grad)
            def _backward_hook(module, grad_input, grad_output):
                """Backward hook to capture gradients"""
                if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                    self.gradcam_gradients['value'] = grad_output[0].detach().clone()
            
            self.gradcam_backward_handle = target_layer.register_full_backward_hook(_backward_hook)
            print(f"Grad-CAM hooks registered successfully on layer: {target_layer}")
        except Exception as e:
            print(f"Warning: Failed to setup Grad-CAM hooks: {e}")
            print("Grad-CAM visualization will not be available.")
            self.gradcam_handle = None
            self.gradcam_backward_handle = None
    
    def _generate_gradcam(self, image_tensor, class_index=None):
        """
        Generate Grad-CAM heatmap for the given image tensor
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            class_index: Target class index. If None, uses predicted class
            
        Returns:
            heatmap: Normalized heatmap array (224, 224) in range [0, 1]
            class_index: Class index used for Grad-CAM
        """
        self.model.eval()
        self.gradcam_activations.clear()
        self.gradcam_gradients.clear()
        
        # Temporarily enable gradients for feature layers (needed for Grad-CAM)
        # Even though parameters are frozen, we need gradients to flow
        for param in self.model.features.parameters():
            param.requires_grad = True
        
        # Ensure gradients are enabled
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(image_tensor)
            if class_index is None:
                class_index = int(output.argmax(dim=1).item())
            target_score = output[0, class_index]
        
        # Backward pass to get gradients
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Restore frozen state
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Retrieve activations and gradients
        if 'value' not in self.gradcam_activations or 'value' not in self.gradcam_gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        
        activations = self.gradcam_activations['value'].squeeze(0)  # (C, H, W)
        gradients = self.gradcam_gradients['value'].squeeze(0)      # (C, H, W)
        
        # Global average pooling over gradients to get channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        heatmap = cam.cpu().numpy()
        return heatmap, class_index
    
    def _overlay_heatmap(self, heatmap, original_image):
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            heatmap: Normalized heatmap array (H, W) in range [0, 1]
            original_image: Original BGR image (H, W, 3)
            
        Returns:
            overlay: Image with heatmap overlaid (BGR format)
        """
        # Resize heatmap to match original image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to colormap (jet colormap)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        # Convert BGR to RGB for overlay
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Blend heatmap with original image
        overlay = cv2.addWeighted(
            original_rgb.astype(np.float32), 
            0.6, 
            heatmap_rgb.astype(np.float32), 
            0.4, 
            0
        )
        
        # Convert back to BGR
        overlay_bgr = cv2.cvtColor(
            overlay.astype(np.uint8), 
            cv2.COLOR_RGB2BGR
        )
        
        return overlay_bgr
    
    def classify(self, nail_image, return_gradcam=False):
        """
        Classify a nail image
        
        Args:
            nail_image: Nail image (BGR numpy array from OpenCV)
            return_gradcam: If True, also return Grad-CAM visualization
            
        Returns:
            dict with keys:
                - 'predicted_class': Name of predicted class
                - 'probability': Confidence score
                - 'all_probabilities': Dict of all class probabilities
                - 'gradcam_heatmap': Base64 encoded Grad-CAM overlay (if return_gradcam=True)
        """
        # Convert BGR to RGB
        if len(nail_image.shape) == 3 and nail_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(nail_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = nail_image
        
        # Store original for Grad-CAM overlay
        original_image = nail_image.copy()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Convert LogSoftmax to probabilities
            probabilities = torch.exp(output).cpu().numpy()[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        all_probs = {
            self.CLASSES[i]: float(probabilities[i])
            for i in range(len(self.CLASSES))
        }
        
        result = {
            'predicted_class': predicted_class,
            'probability': confidence,
            'all_probabilities': all_probs
        }
        
        # Generate Grad-CAM if requested
        if return_gradcam:
            if self.gradcam_handle is None and self.gradcam_backward_handle is None:
                print("Warning: Grad-CAM hooks not available, skipping visualization")
                result['gradcam_heatmap'] = None
            else:
                try:
                    print("Generating Grad-CAM visualization...")
                    heatmap, _ = self._generate_gradcam(input_tensor, predicted_idx)
                    print(f"Grad-CAM heatmap generated, shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
                    overlay = self._overlay_heatmap(heatmap, original_image)
                    print(f"Grad-CAM overlay created, shape: {overlay.shape}")
                    
                    # Encode overlay to base64
                    success, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not success:
                        raise ValueError("Failed to encode overlay image to JPEG")
                    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
                    result['gradcam_heatmap'] = gradcam_base64
                    print(f"Grad-CAM encoded to base64 successfully, length: {len(gradcam_base64)}")
                except Exception as e:
                    print(f"Warning: Failed to generate Grad-CAM: {e}")
                    import traceback
                    traceback.print_exc()
                    result['gradcam_heatmap'] = None
        
        return result

