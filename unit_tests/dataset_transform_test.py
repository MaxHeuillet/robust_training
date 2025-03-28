import unittest
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf

import sys 
import os
import warnings
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(PROJECT_ROOT)  # Should print the path to your main directory
sys.path.append(PROJECT_ROOT)

from databases import load_data

# Suppose this is your custom grayscale-to-RGB transform
class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'L':
            # Convert single-channel to 3 channels
            img = img.convert('RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

# Your load_data function:
# from datasets import load_data


class TestDatasetTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the config once, and define the 'expected' transforms."""
        cls.config = OmegaConf.load("./configs/default_config_linearprobe50.yaml")
        # cls.config.datasets_path = './data'  # local dataset path

        # The transforms we expect for the training set
        cls.train_transforms_expected = T.Compose([
            T.Resize((224, 224)),
            GrayscaleToRGB(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.25, 0.25, 0.25),
            T.RandomRotation(2),
            T.ToTensor()
        ])

        # The transforms we expect for the val/test set
        cls.eval_transforms_expected = T.Compose([
            T.Resize((224, 224)),
            GrayscaleToRGB(),
            T.ToTensor()
        ])

    def test_data_transforms(self):
        """Check that each dataset uses the correct transforms for train/val/test."""
        datasets_to_test = [
            'uc-merced-land-use-dataset',
            'flowers-102',
            'caltech101',
            'stanford_cars',
            'fgvc-aircraft-2013b',
            'oxford-iiit-pet'
        ]

        for dataset_name in datasets_to_test:

            with self.subTest(dataset=dataset_name):
                 
                 print('testing', dataset_name)
                 command = ["bash", "./dataset_to_tmpdir.sh", dataset_name]

                 # Run the command
                 result = subprocess.run(command, capture_output=True, text=True)
                 print('imported data to tmpdir')

                 with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    self.config.dataset = dataset_name
                    # load_data is your custom function returning (train_ds, val_ds, test_ds, N)
                    train_ds, val_ds, test_ds, num_classes = load_data(self.config, common_corruption=False)

                    # 1) Check the training set transform
                    # self.assertTrue(hasattr(train_ds, 'transform'), f"Train set missing 'transform' for {dataset_name}")
                    self._compare_transforms(train_ds.transform, self.train_transforms_expected, split='train')

                    # 2) Check the validation set transform
                    # self.assertTrue(hasattr(val_ds, 'transform'), f"Val set missing 'transform' for {dataset_name}")
                    self._compare_transforms(val_ds.transform, self.eval_transforms_expected, split='val')

                    # 3) Check the test set transform
                    # self.assertTrue(hasattr(test_ds, 'transform'), f"Test set missing 'transform' for {dataset_name}")
                    self._compare_transforms(test_ds.transform, self.eval_transforms_expected, split='test')

    def _compare_transforms(self, actual_compose, expected_compose, split):
        """Helper to compare two torchvision.transforms.Compose objects."""
        # 1) They must have the same number of steps
        self.assertEqual(
            len(actual_compose.transforms),
            len(expected_compose.transforms),
            msg=f"{split} transforms length mismatch.\n"
                f"Actual: {actual_compose}\nExpected: {expected_compose}"
        )


if __name__ == '__main__':
    unittest.main()
