#!/bin/bash

python headpose_estimation/retrain.py
--architecture mobilenet_1.0_128
--model_dir models/
--image_dir data/raw/300W-LP/300W_LP/
--labels_file data/ground_truths.txt
--how_many_training_steps 100
--summaries_dir summaries/
--validation_percentage 10
--testing_percentage 10
--bottleneck_dir bottlenecks/
--saved_model_dir ./models
