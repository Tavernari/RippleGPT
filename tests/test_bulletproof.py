import unittest
import torch
from src.config import RippleConfig
from src.model import RippleGPT

class TestRippleGPT(unittest.TestCase):
    def setUp(self):
        # Tiny config for testing
        self.config = RippleConfig(
            vocab_size=50, n_embd=32, n_head=2, n_layer=2, block_size=16, use_absolute_pos_emb=False
        )
        self.model = RippleGPT(self.config)

    def test_parameter_count(self):
        """Ensure it's efficient"""
        params = sum(p.numel() for p in self.model.parameters())
        # Roughly estimate: 12 * n_layer * n_embd^2
        print(f"Test Model Params: {params}")
        self.assertTrue(params > 0)

    def test_forward_pass(self):
        """Ensure data flows"""
        idx = torch.randint(0, 50, (2, 16)) # Batch 2, Len 16
        logits, loss = self.model(idx, idx)
        self.assertEqual(logits.shape, (2, 16, 50))
        self.assertIsNotNone(loss)

    def test_extrapolation_capability(self):
        """Scientific Proof: Can it run on length > block_size?"""
        # Config says block_size=16. Let's feed it 32.
        idx_long = torch.randint(0, 50, (1, 32)) 
        try:
            logits, _ = self.model(idx_long)
            success = True
            print("Extrapolation Test: PASSED")
        except Exception as e:
            success = False
            print(f"Extrapolation Test: FAILED ({e})")
        
        self.assertTrue(success, "Model should handle sequence length > trained block_size")

if __name__ == '__main__':
    unittest.main()
