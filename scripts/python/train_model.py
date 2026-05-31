from headpose_estimation.schema import ExperimentConfig
from headpose_estimation.trainer import PredictionHeadTrainer


def main():
    experiment_config: ExperimentConfig = ExperimentConfig.from_args()
    trainer: PredictionHeadTrainer = PredictionHeadTrainer(
        experiment_config=experiment_config
    )
    trainer.train()


if __name__ == "__main__":
    main()
