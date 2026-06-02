# Architecture

This is the main narrative document. Read it top-to-bottom to rebuild the mental
model of the project.

For per-class detail see [components.md](components.md); for
the reasoning behind the non-obvious choices see [design-decisions.md](design-decisions.md).

---

## 1. Overview

The project trains **head-pose estimators** that predict the three Euler angles —
**yaw, pitch, roll** - of a face from a cropped face image.

Rather than train a CNN from scratch, it uses **transfer learning**:

- A **frozen, pretrained image backbone** (Inception V3 or any MobileNet V1
  variants currently supported) turns an image into a fixed-length feature vector ("image
  representation" / "bottleneck"). Its weights are not updated.
- A small, **trainable MLP head** maps that feature vector to the three angles.
  This is the only part that learns.

Because the backbone is frozen, its output for a given image never changes. So we
run every image through the backbone **once**, cache the result to disk, and then
train the head purely on those cached vectors. This is the central performance
idea of the codebase (see [design-decisions.md](design-decisions.md#two-phase-training-with-bottleneck-caching)).

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │  Phase A — preprocess datasets (one-off, separate script)   │
raw videos/imgs ─┤  decode frames, scale angles to [-1, 1], emit JSON manifest │
                 └─────────────────────────────────────────────────────────────┘
                                          │  manifest of {id, image_path, angles}
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Phase B — bottleneck generation (once per image)                        │
   │  image ──► frozen backbone ──► feature vector ──► cached as .npz on disk │
   └──────────────────────────────────────────────────────────────────────────┘
                                          │  cached feature vectors
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  Phase C — head training (the actual learning)                            │
   │  feature vector ──► MLP head ──► (yaw, pitch, roll) ──► MSE loss ──► Adam │
   └───────────────────────────────────────────────────────────────────────────┘
                                          │  trained head
                                          ▼
                          frozen prediction_head.pb on disk
```

> **TensorFlow 1, graph mode.** The whole project is built on TF1 graph-mode APIs
> (`tf.compat.v1.Session`, `tf.compat.v1.placeholder`, `GraphDef`, `import_graph_def`).
> Do **not** mix in TF2 eager-mode patterns — the pretrained backbones ship as
> frozen TF1 graph definitions and everything downstream assumes the
> build-graph-then-`session.run` model.

---

## 2. How the pieces fit together

### The model contract (`models/base.py`)

Every model implements a small abstract interface so the trainer can treat
backbone and head uniformly. The pattern is deliberately TF1: each model **builds
its TF nodes in `__init__`** and exposes them as **properties**; nothing actually
executes until a caller passes a `tf.Session` into `predict()` / `train()`.

- `BaseModel[ModelInput, PredictionOutput]` - generic base. Exposes
  `input_placeholder`, `prediction_ops`, and `predict(session, inputs)`.
- `BaseAnglePredictionHeadModel` - extends it for trainable heads, adding
  `training_ops`, `ground_truth_placeholders`, and `loss_ops`.

### The class map

- **`PretrainedBackBoneImageModel`** - loads a frozen `.pb` graph, prepends its own
  JPEG-decode/resize/normalize subgraph, and produces a feature vector per image.
  Branches on `supports_batching` because Inception V3 cannot take batched input.
- **`EulerAnglesPredictionHead`** - builds **three independent MLP stacks** (one per
  angle) on top of the feature vector, plus the summed MSE loss, the optimizer
  step, and the TensorBoard summary ops. This is the trainable part.
- **`EulerAnglesPredictionModel`** - a thin container that wires a backbone and a
  head together so you can go straight from image bytes to angles at inference
  time. (Note: it has live bugs - see [components.md](components.md#euleranglespredictionmodel-end_to_end_modelspy).)
- **`PredictionHeadTrainer`** - the orchestrator. Constructs the backbone and head
  from the config, owns the single `tf.Session`, runs bottleneck generation and the
  train/validate/test loop, and exports the frozen head.


```
            ExperimentConfig (schema.py)              TensorflowV1ModelHandler
            loaded from YAML / argparse               downloads + describes a
                     │                                pretrained backbone
                     │                                        │
                     ▼                                        ▼
            ┌─────────────────────────────────────────────────────────────┐
            │                  PredictionHeadTrainer                      │
            │                      (trainer.py)                           │
            │   owns the Session, runs phases B and C, writes TensorBoard │
            └─────────────────────────────────────────────────────────────┘
                     │                                        │
        builds ──────┤                                        ├────── builds
                     ▼                                        ▼
      PretrainedBackBoneImageModel                 EulerAnglesPredictionHead
      (models/pretrained_image_model.py)           (models/angle_prediction_heads.py)
      frozen graph + preprocessing                 3 trainable MLPs + losses + optimizer
                     │                                        │
                     └──────────────┬─────────────────────────┘
                                    ▼
                       EulerAnglesPredictionModel
                       (models/end_to_end_models.py)
                       container that chains backbone ► head
                       for single-call image-to-angles inference
```

---

## 3. The training pipeline

Entry point: `scripts/python/train_model.py` → builds an `ExperimentConfig` →
constructs `PredictionHeadTrainer` → calls `.train()`.

```bash
python scripts/python/train_model.py -c configs/experiments/test_experiment.yml
```

`PredictionHeadTrainer.train()` runs these stages inside one `tf.Session`:

1. **Seed** - `set_global_seed` fixes `random`, `numpy`, and TF seeds for
   reproducibility.
2. **Load the manifest** - `Dataset.load_from_file` reads the preprocessing JSON
   into `Datapoint`s.
3. **Bottleneck generation** *(optional, gated by `run_bottleneck_generations`)* -
   `_create_intermediate_image_representations` runs every image through the
   backbone in batches and writes each feature vector to
   `intermediate_representations_save_dir/<id>.npz`. The manifest is updated with
   the cache path and re-saved. Skip this on later runs to reuse the cache.
4. **Split** - `Dataset.shuffle_and_train_test_split` shuffles and slices into
   train / validation / test by the configured percentages.
5. **Train loop** - for each epoch, iterate the train set in batches: load cached
   vectors + ground-truth angles, call `head.train(...)`, and write the training
   summary. Every `eval_step_interval` global steps, run validation via
   `calculate_validation_loss` and log it.
6. **Test + export** - after all epochs, compute the test loss, then freeze the
   head to `model_save_dir/prediction_head.pb` via `head.save_as_frozen_graph`.

Outputs land under `<experiment_root>/<experiment_name>/` (created by
`_setup_experiment_directories`), including the TensorBoard logs.

---

## 4. Configuration

Experiments are **fully config-driven**. `schema.py` defines a tree of dataclasses
that mirror the YAML; `ExperimentConfig.from_yaml` (or `.from_args`) loads them.


Reference config: [`configs/experiments/test_experiment.yml`](../configs/experiments/test_experiment.yml).


### String-reference injection

The `optimizer_class`, `learning_rate_decay_function`, and the head's activation
function are all specified in YAML as **dotted import strings**, e.g.
`tensorflow.compat.v1.train.AdamOptimizer`. `utils/utils.py` resolves these at
runtime:

- `fn_from_str("a.b.c")` → imports module `a.b` and returns attribute `c` (a callable).
- `cls_from_str("a.b.C")` → same, for a class.
- `optimizer_factory(optimizer_config, global_step)` ties it together: optionally
  builds a learning-rate-decay tensor, instantiates the optimizer class, and
  returns `(optimizer, learning_rate_tensor)`.

This is why you can swap optimizers or activations without touching Python — see
[design-decisions.md](design-decisions.md#string-reference-injection-for-optimizer--activation).

### The backbone registry

`configs/tf_models/tf_models.yaml` is a generated registry of every supported
backbone (Inception V3 + the full MobileNet V1 family) with its download URL,
graph-definition filename, input/output node names, input size, feature-vector
size, and `supports_batching` flag. Regenerate it with
`python scripts/python/generate_tf_model_configs.py`. `TensorflowV1ModelHandler`
reads this file to download and describe a backbone on demand.
