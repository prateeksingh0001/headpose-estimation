from headpose_estimation.utils.tensorflow_model_handler import TensorflowModelHandler


def main():
    TF_MODELS_CONFIG = "configs/tf_models/tf_models.yaml"
    tf_model_handler = TensorflowModelHandler(tf_models_config_path = TF_MODELS_CONFIG)

    tf_model_handler.get_model("inception_v3", "models/test")



if __name__ == "__main__":
    main()