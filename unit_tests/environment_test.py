import unittest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(PROJECT_ROOT)  # Should print the path to your main directory
sys.path.append(PROJECT_ROOT)

class TestImports(unittest.TestCase):
    def test_imports(self):
        modules = [
            "os",
            "torch",
            "torch.distributed",
            "torch.utils.data",
            "torch.optim.lr_scheduler",
            "torch.nn.parallel",
            "torch.cuda.amp",
            "torch.multiprocessing",
            "multiprocessing",
            "autoattack",
            "ray.air",
            "utils",
            "databases",
            "architectures",
            "losses",
            "hydra",
            "omegaconf",
            "ray",
            "shutil",
            "comet_ml",
        ]
        
        for module in modules:
            with self.subTest(module=module):
                try:
                    __import__(module)
                except ImportError as e:
                    self.fail(f"Failed to import {module}: {e}")

if __name__ == "__main__":
    unittest.main()
