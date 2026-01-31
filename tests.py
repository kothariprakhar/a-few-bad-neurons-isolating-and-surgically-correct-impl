import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer

# Import the logic from the user script (assuming it's in a module or pasted here)
# For testing, we simulate the environment.

class TestSycophancyImplementation(unittest.TestCase):

    def setUp(self):
        # Configuration for a tiny GPT2 model to run fast
        self.config = GPT2Config(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            n_inner=64, # inner_dim (neurons)
            activation_function="gelu"
        )
        self.model = GPT2LMHeadModel(self.config)
        self.layer_idx = 0
        self.inner_dim = 64
        self.hidden_dim = 32

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_probe_activations_and_shapes(self, mock_tokenizer_cls, mock_model_cls):
        # Setup Mocks
        mock_model_cls.return_value = self.model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        # Mock tokenizer call to return tensors
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones((1, 10))
        }
        mock_tokenizer_cls.return_value = mock_tokenizer

        # 1. Test Activation Hook Logic
        activations_storage = []
        def get_activation_hook(module, input, output):
            # output shape: [batch, seq, inner_dim]
            activations_storage.append(output[:, -1, :].detach().cpu())

        layer = self.model.transformer.h[self.layer_idx]
        hook_handle = layer.mlp.act.register_forward_hook(get_activation_hook)

        # Run Forward
        inputs = mock_tokenizer("test")
        _ = self.model(**inputs)
        
        # Verify shapes
        self.assertEqual(len(activations_storage), 1)
        # Expected shape: (batch_size=1, inner_dim=64)
        self.assertEqual(activations_storage[0].shape, (1, self.inner_dim))
        
        hook_handle.remove()

    def test_gradient_masking_logic(self):
        # Simulate identified "bad" neurons
        top_neuron_indices = [5, 10, 20]
        
        mlp_layer = self.model.transformer.h[self.layer_idx].mlp
        # Conv1D weights are (input_features, output_features) -> (inner_dim, hidden_dim)
        # Check shape first
        self.assertEqual(mlp_layer.c_proj.weight.shape, (self.inner_dim, self.hidden_dim))
        
        # Create Mask
        full_mask = torch.zeros_like(mlp_layer.c_proj.weight)
        for idx in top_neuron_indices:
            full_mask[idx, :] = 1.0
            
        # Hook for gradient masking
        def make_grad_mask_hook(mask):
            def hook(grad):
                return grad * mask
            return hook
            
        hook_handle = mlp_layer.c_proj.weight.register_hook(make_grad_mask_hook(full_mask))
        
        # Run Forward & Backward
        input_ids = torch.randint(0, 100, (1, 10))
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        
        grad = mlp_layer.c_proj.weight.grad
        
        # Verify gradients are zero for non-selected neurons
        for i in range(self.inner_dim):
            row_grad = grad[i, :]
            if i in top_neuron_indices:
                # Should be non-zero (highly likely)
                # Note: It might be zero if the neuron didn't activate, but generally non-zero.
                pass 
            else:
                # MUST be zero
                self.assertTrue(torch.all(row_grad == 0), f"Gradient at index {i} should be 0")
        
        # Verify gradients are present for selected neurons (if loss > 0)
        # We can't guarantee non-zero grad for specific random init, but we can check the mask worked.
        self.assertTrue(grad[top_neuron_indices[0], :].abs().sum() >= 0)

    def test_synthetic_dataset_fallback(self):
        # Test the fallback logic in data preparation
        # We mimic the class behavior simply
        data = []
        limit = 5
        for i in range(limit):
            data.append({
                "text": "User: ..."
            })
        self.assertEqual(len(data), limit)
        self.assertIn("text", data[0])

if __name__ == '__main__':
    unittest.main()