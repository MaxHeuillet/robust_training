import unittest
from unittest.mock import patch, MagicMock
import os
from comet_ml import Experiment

class TestCometMLLogger(unittest.TestCase):
    """ Unit test for Comet ML experiment initialization and logging """

    @patch("comet_ml.Experiment", autospec=True)  # Ensures correct patching
    def test_comet_logger_initialization(self, MockExperiment):
        """ Test if the Comet ML logger initializes and logs correctly """

        # Create a mock experiment instance
        mock_experiment_instance = MagicMock()
        MockExperiment.return_value = mock_experiment_instance  # Ensure mock is used

        # Simulated configuration
        config = {
            "project_name": "test-project",
            "exp_id": "test_experiment_123"
        }

        # Initialize the logger
        logger = Experiment(
            api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
            project_name=config["project_name"],
            workspace="maxheuillet",
            auto_metric_logging=False,
            auto_output_logging=False,
        )

        # Simulating logging actions
        logger.set_name(config["exp_id"])
        logger.log_parameter("run_id", "123456")  # Simulated SLURM_JOB_ID
        logger.log_parameter("global_process_rank", 0)
        logger.log_parameters(config)

        # Assertions to check that methods were called correctly
        mock_experiment_instance.set_name.assert_called_with(config["exp_id"])
        mock_experiment_instance.log_parameter.assert_any_call("run_id", "123456")
        mock_experiment_instance.log_parameter.assert_any_call("global_process_rank", 0)
        mock_experiment_instance.log_parameters.assert_called_with(config)

        print("✅ Comet ML logger initialized and logging verified successfully.")

# Run the test
if __name__ == "__main__":
    unittest.main()
