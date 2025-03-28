import unittest
import os
from unittest.mock import patch
from comet_ml import Experiment

class TestCometMLLogger(unittest.TestCase):
    """ Unit test for Comet ML experiment initialization and logging """

    @patch("comet_ml.Experiment")
    def test_comet_logger_initialization(self, MockExperiment):
        """ Test if the Comet ML logger initializes and logs correctly """

        # Mock Comet Experiment to avoid actual API calls
        mock_experiment = MockExperiment.return_value
        
        # Simulated configuration
        config = {
            "project_name": "test_project",
            "exp_id": "test_experiment_123"
        }

        # Initialize the logger
        try:
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
            mock_experiment.set_name.assert_called_with(config["exp_id"])
            mock_experiment.log_parameter.assert_any_call("run_id", "123456")
            mock_experiment.log_parameter.assert_any_call("global_process_rank", 0)
            mock_experiment.log_parameters.assert_called_with(config)

            print("✅ Comet ML logger initialized and logging verified successfully.")

        except Exception as e:
            self.fail(f"🚨 Comet ML logger initialization failed: {e}")

# Run the test
if __name__ == "__main__":
    unittest.main()
