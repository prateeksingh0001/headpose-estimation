TF_MODELS_CONFIG_PATH = "configs/tf_model_registry/tf_model_registry.yaml"
DEFAULT_SAVE_NP_ARRAY_NAME = "arr_0"

# Tensorflow related constants
TENSORBOARD_DIR_NAME = "tensorboard_logs"
DEFAULT_PREDICTION_HEAD_FROZEN_GRAPH_NAME = "prediction_head.pb"

# Model constants
YAW_ANGLE_KEY = "yaw"
PITCH_ANGLE_KEY = "pitch"
ROLL_ANGLE_KEY = "roll"
TOTAL = "total"

# Since all the angles are in degrees we assume that the total range of angles will between -90 and 90 degrees
# across each axis so we normalize the angles to be between [-1, 1].
# This keeps the training stable as the difference between the predictions and ground truths are little and there
# aren't wild swings in the MSE loss during training.
EULER_ANGLE_NORMALIZATION_FACTOR = 90
