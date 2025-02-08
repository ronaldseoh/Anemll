#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import unittest
import torch
from pathlib import Path
from anemll.models.llama_model import LlamaModel, LlamaConfig, LlamaForCausalLM

class TestLlamaModel(unittest.TestCase):
    """Test cases for LLAMA model implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = Path("../Meta-Llama-3.2-1B")  # Adjust path as needed
        cls.config = LlamaConfig(
            hidden_size=512,        # Smaller for testing
            intermediate_size=1024,
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=4,
            vocab_size=32000
        )

    def test_model_forward(self):
        """Test basic forward pass of the model."""
        model = LlamaModel(config=self.config)
        batch_size, seq_length = 1, 10
        
        # Create sample inputs
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        position_ids = torch.arange(seq_length).unsqueeze(0)
        
        # Run forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Check output shapes
        expected_shape = (batch_size, seq_length, self.config.hidden_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_causal_lm_forward(self):
        """Test the causal language model forward pass."""
        model = LlamaForCausalLM(config=self.config)
        batch_size, seq_length = 1, 10
        
        # Create sample inputs
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Run forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Check output shapes
        expected_shape = (batch_size, seq_length, self.config.vocab_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_pretrained_weights(self):
        """Test loading and inference with pretrained weights."""
        if not self.model_path.exists():
            self.skipTest("Pretrained model path not found")
        
        # Initialize model with pretrained weights
        model = LlamaModel(
            config=self.config,
            model_path=self.model_path
        )
        model.preprocess()  # This loads the weights
        
        # Test inference
        input_text = torch.tensor([[1, 2, 3, 4, 5]])  # Sample input
        with torch.no_grad():
            output = model(input_text)
        
        # Check output is not all zeros or NaNs
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.isnan(output).any())

    def test_attention_cache(self):
        """Test KV cache functionality."""
        model = LlamaModel(config=self.config)
        
        # First forward pass
        input_ids = torch.tensor([[1, 2, 3]])
        first_output = model(input_ids)
        
        # Second forward pass with new token
        new_input = torch.tensor([[4]])
        second_output = model(new_input)
        
        # Check shapes
        self.assertEqual(first_output.shape, (1, 3, self.config.hidden_size))
        self.assertEqual(second_output.shape, (1, 1, self.config.hidden_size))

    def test_model_validation(self):
        """Test model validation checks."""
        model = LlamaModel(config=self.config)
        
        # Should not raise any assertions
        try:
            model.validate()
        except AssertionError:
            self.fail("Model validation failed")

    def test_compare_with_hf(self):
        """Compare outputs with Hugging Face implementation."""
        try:
            from transformers import LlamaModel as HFLlamaModel
            from transformers import LlamaConfig as HFLlamaConfig
        except ImportError:
            self.skipTest("transformers package not installed")
        
        # Create identical configs
        hf_config = HFLlamaConfig(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_attention_heads=self.config.num_attention_heads,
            num_hidden_layers=self.config.num_hidden_layers,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size
        )
        
        # Initialize both models
        our_model = LlamaModel(config=self.config)
        hf_model = HFLlamaModel(hf_config)
        
        # Copy weights from HF to our model (implement this helper if needed)
        # copy_weights_from_hf(our_model, hf_model)
        
        # Create identical inputs
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        
        # Get outputs
        with torch.no_grad():
            our_output = our_model(input_ids)
            hf_output = hf_model(input_ids).last_hidden_state
        
        # Compare outputs (with some tolerance for numerical differences)
        torch.testing.assert_close(our_output, hf_output, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    unittest.main() 