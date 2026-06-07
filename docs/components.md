# Component Reference

Per-class detail for the active codebase. For the big picture read
[architecture.md](architecture.md) first; for the *why* behind the choices see
[design-decisions.md](design-decisions.md).

Each section lists the class's responsibility, what it owns, the key methods, and
any gotchas / known bugs.

---

## `BaseModel` / `BaseAnglePredictionHeadModel` — `models/base.py`

- `BaseModel[ModelInput, PredictionOutput]` — generic **ABC**. Declares the abstract
  surface every model must provide: the `prediction_ops` and `input_placeholder`
  properties and `predict(session, prediction_input)`.
- `BaseAnglePredictionHeadModel(BaseModel[np.ndarray, Dict[str, float]])` — the
  shared base for **all angle heads**. Defines the dict keys
  (`YAW_ANGLE_KEY` / `PITCH_ANGLE_KEY` / `ROLL_ANGLE_KEY` / `TOTAL`) and provides
  **concrete** implementations of the entire head surface, each backed by a
  `self._*` attribute the subclass populates in `__init__`:
  - properties: `prediction_ops` ← `_prediction_ops`, `input_placeholder` ←
    `_input_placeholder`, `ground_truth_placeholders` ← `_ground_truth_placeholder`,
    `training_ops` ← `_training_ops`, `loss_ops` ← `_loss_ops`, `summary_ops` ←
    `_summary_ops`.
  - `predict(session, prediction_input)` — `vstack`s the inputs, runs the three
    prediction ops, and returns a `List[Dict[str, float]]` of yaw/pitch/roll, each
    scaled by `self.EULER_ANGLE_NORMALIZATION_FACTOR`.
  - `save_as_frozen_graph(session, output_graphdef_path)` — folds variables into
    constants (`convert_variables_to_constants`) over the prediction ops and writes
    a frozen `.pb`.

**Pattern:** a head subclass only has to build its TF nodes in `__init__` and assign
the `self._prediction_ops` / `_input_placeholder` / `_ground_truth_placeholder` /
`_training_ops` / `_loss_ops` / `_summary_ops` attributes (and a `train` method) —
the base supplies all the accessors, `predict`, and `save_as_frozen_graph`. Nothing
runs until a `tf.Session` is passed into a method.

---

## `PretrainedBackBoneImageModel` — `models/pretrained_image_model.py`

**Responsibility:** load a frozen pretrained backbone and turn raw JPEG bytes into
a fixed-length feature vector ("image representation" / bottleneck).

**Constructor:** takes a `TensorflowV1ModelConfig`. It then:

1. Creates a `tf.string` placeholder for raw image bytes (`input_image_bytes`).
2. Builds a **preprocessing subgraph** (`_create_preprocessing_graph`):
   `decode_jpeg` → `expand_dims` → `resize_bilinear` to the model's input size →
   pixel normalization to `[-1, 1]` via
   `(x - PIXEL_NORMALIZATION_MEAN) * PIXEL_NORMALIZATION_MULTIPLIER`
   (`mean=128`, `mult=0.0078125 = 1/128`).
3. Loads the frozen graph from disk (`_load_graph_definition` + `import_graph_def`)
   and grabs the configured input/output nodes (`_create_model_graph`).

**Key methods:**

- `predict(session, prediction_input)` — preprocesses each image, then runs the
  backbone. **Branches on `model_config.supports_batching`:** if the backbone
  supports batching it `vstack`s all images and runs once, then `vsplit`s the
  result; otherwise it loops one image at a time. Returns a `List[np.ndarray]`,
  one feature vector per input.
- `prediction_ops` — `{"image_representation": <output tensor>}`.

**Gotchas:**

- The preprocessing subgraph and the frozen backbone graph are **separate** — the
  trainer feeds raw bytes to the preprocessing placeholder, then feeds the
  preprocessed array into the backbone's input node. They are not joined into one
  graph here.
- `supports_batching=false` for Inception V3 because its resize node rejects
  batched input — see [design-decisions.md](design-decisions.md#per-backbone-batching-the-supports_batching-flag).

---

## `EulerAnglesPredictionHead` — `models/angle_prediction_heads.py`

**Responsibility:** the trainable part. Map a feature vector to yaw, pitch, roll.

**Constructor inputs:** `input_representation_size`, `layer_sizes` (hidden widths),
`optimizer`, `learning_rate_tensor`, `global_step`, and an `activation_function`
import string (default `tensorflow.nn.relu`).

**Structure:** builds **three independent MLP stacks**, one per angle
(`yaw_predictor`, `pitch_predictor`, `roll_predictor`), via
`_intialize_weights_for_an_angle`. Each stack is the configured `layer_sizes`
followed by a final width-1 layer. **Layers use `use_bias=False`.** Hidden layers
use the configured `activation_function`; the **final width-1 layer is linear
(`activation=None`)** so the head can output negative angles.

The Dense layers do **not** receive an input dimension — TF1's `Dense` infers it
lazily from the input tensor's shape when variables are first initialized
(`session.run(tf.global_variables_initializer())`). The class docstring calls this
out explicitly.

**Angle normalization** (`EULER_ANGLE_NORMALIZATION_FACTOR = 90`): the head assumes
angles lie in `[-90, 90]` degrees and normalizes to `[-1, 1]` for stable MSE
training. This is done **inside the head**, not in preprocessing — `train` divides
the incoming ground truth by 90, and `predict` multiplies the raw output back by 90
to return degrees.

**Training ops** (`_initialize_training_ops`): three `tf.float32` ground-truth
placeholders; per-angle MSE losses summed into `total_loss`; one optimizer
`.minimize(total_loss, global_step=...)`; and a merged TensorBoard summary
(learning rate + total/per-angle losses).

**Key method:**

- `train(session, image_representations, ground_truths)` — normalizes the ground
  truth by `EULER_ANGLE_NORMALIZATION_FACTOR`, `vstack`s the batch, feeds
  representations + per-angle ground truth, runs the optimizer + loss + summary ops.
  Returns `(total_loss, yaw_loss, roll_loss, pitch_loss, summary)`.

`predict` (which scales outputs back to degrees) and `save_as_frozen_graph` are
inherited from `BaseAnglePredictionHeadModel`.

**Gotchas:**

- `train(...)` returns losses in order **total, yaw, roll, pitch** — note roll
  comes before pitch. The trainer unpacks them as
  `total, yaw, roll, pitch` to match, but be careful if you reuse the return value.
- Order matters because losses/ops are read off dict `.values()`; keep the dicts in
  sync if you edit them.
- `train` mutates the `ground_truths` dict in place while normalizing (it rebinds
  each list to a scaled copy), so don't reuse the same dict expecting raw degrees.

---

## `HopeNetPredictionHead` — `models/angle_prediction_heads.py`

**Responsibility:** an alternative angle head implementing the
[HopeNet](https://arxiv.org/pdf/1710.00925) multi-loss formulation — coarse-grained
**bin classification** plus a fine-grained **expected-value regression** — instead
of `EulerAnglesPredictionHead`'s pure MSE.

**Bin scheme (class constants):**

- `NUM_ANGLE_BUCKETS = 67` — bins indexed `[0, 66]`, covering `[-99, 99]` degrees
  (one more bin than the original paper, to span the full `±99°` range).
- `ANGLE_PER_BUCKET = 3` — 3° per bin.
- `MAX_EULER_ANGLE = 99` — offset mapping bin index ↔ degrees.
- `ALPHA = 1` — weight on the regression term in the combined loss.
- `YAW_CLASS_KEY` / `PITCH_CLASS_KEY` / `ROLL_CLASS_KEY` — dict keys for the one-hot
  bin ground-truth placeholders.

**Structure:** three independent branches (`yaw_predictor`, `pitch_predictor`,
`roll_predictor`) built by `_initialize_weights_for_angle_prediction`. Each branch
is `layer_sizes` followed by a final `NUM_ANGLE_BUCKETS`-wide layer; hidden layers
use the activation, the **final bin-logits layer is linear**. Each branch returns
two tensors:

- the raw **bin logits** (for cross-entropy), and
- a continuous **angle prediction** = `reduce_sum(softmax(logits) · bin_index) ·
  ANGLE_PER_BUCKET − MAX_EULER_ANGLE`, i.e. the softmax-weighted expected bin
  recentered to degrees. Kept as shape `[batch, 1]` (`keepdims=True`) for parity
  with `EulerAnglesPredictionHead`.

`__init__` builds the three branches, stashes the predictions in `_prediction_ops`,
and calls `_initialize_training_ops` with the per-angle logit tensors.

**Loss** (`_calculate_loss_for_angle`): `softmax_cross_entropy_with_logits` on the
bin logits (vs. the one-hot bin label) **plus** `ALPHA ·` MSE
(`reduction=Reduction.NONE`) between the squeezed continuous angle prediction and
the degree-valued ground truth.

**Training ops** (`_initialize_training_ops`): per angle, a `[None]` continuous-GT
placeholder and a `[None, NUM_ANGLE_BUCKETS]` one-hot bin-GT placeholder (all six
stored in `_ground_truth_placeholder`); sums the three per-angle losses,
`reduce_mean`s into `total_loss`, creates the optimizer step, and appends a merged
TensorBoard summary to the summary-ops list.

**Key method:**

- `train(session, image_representations, ground_truths)` — takes ground truth in
  **raw degrees** and derives the bin targets internally via
  `(angle + MAX_EULER_ANGLE) / ANGLE_PER_BUCKET`, then feeds both the continuous and
  bin ground truths. Returns `(total_loss, yaw_loss, roll_loss, pitch_loss,
  summary)`. Because the bins are computed here, the data pipeline does **not** need
  to emit bin labels.

`predict` and `save_as_frozen_graph`, plus all the property accessors, are inherited
from `BaseAnglePredictionHeadModel`.

---

## `EulerAnglesPredictionModel` — `models/end_to_end_models.py`

**Responsibility:** container that chains a backbone and a head so a single
`predict` call goes from raw image bytes to angles (backbone → feature vector →
head → angles).

**Constructor:** `(backbone_image_model, euler_angle_prediction_head)`.

---

## `PredictionHeadTrainer` — `trainer.py`

**Responsibility:** orchestrate an end-to-end experiment. Owns the single
`tf.Session` and runs bottleneck generation + the train/validate/test loop +
export.

**Constructor:** from an `ExperimentConfig`, it creates the experiment
directories, the `global_step` variable, the optimizer (via `optimizer_factory`),
the backbone (`PretrainedBackBoneImageModel`, looked up through
`TensorflowV1ModelHandler`), and the head (`EulerAnglesPredictionHead`).

**Key methods:**

- `_load_dataset()` — loads the manifest. **Truncates to the first 1000 datapoints**
  (`dataset.datapoints[:1000]`) — a dev convenience; remove/raise for full runs.
- `_create_intermediate_image_representations(session, dataset)` — Phase B. Reads
  images in `batch_size` chunks, runs the backbone, writes each feature vector to
  `<intermediate_representations_save_dir>/<id>.npz` (compressed), and records the
  cache path back on each `Datapoint`.
- `get_img_representation_and_gt_batch(dataset, start_index, batch_size)`
  *(staticmethod)* — loads cached `.npz` vectors (array key
  `DEFAULT_SAVE_NP_ARRAY_NAME = "arr_0"`) plus per-angle ground truth for a batch.
- `calculate_validation_loss(session, dataset, tag_prefix)` — runs the head over a
  split, computes per-angle and total MSE in NumPy, and packages a `tf.Summary`.
  Used for both validation and test.
- `train()` — the full loop described in
  [architecture.md §3](architecture.md#3-the-training-pipeline): seed → load →
  optional bottleneck gen → split → epochs (with periodic validation) → test →
  freeze head to `prediction_head.pb`.

**Gotchas:**

- Bottleneck generation is gated by `training_config.run_bottleneck_generations`.
  After the first run, set it false to reuse the cache (the manifest then already
  carries the `intermediate_representation_path`s).
- `checkpoint_save_frequency` is configured but not currently used in `train()`.

---

## `Dataset` / `Datapoint` — `schema.py`

**Responsibility:** in-memory representation of the dataset manifest.

- `Datapoint` — `id`, `image_path`, `yaw/pitch/roll_ground_truth`, and an optional
  `intermediate_representation_path` (filled in during bottleneck generation).
- `Dataset` — wraps `List[Datapoint]`; supports `len()` and indexing.
  - `load_from_file(path)` — reads the JSON manifest into `Datapoint`s.
  - `shuffle_and_train_test_split(train%, val%, test%)` — shuffles in place and
    slices into three `Dataset`s.
  - `save(path)` — writes the manifest back to JSON (used after bottleneck gen so
    the cache paths persist).

**Manifest format** (list of objects): `{id, image_path, yaw_ground_truth,
pitch_ground_truth, roll_ground_truth}`, angles already scaled to `[-1, 1]`.

The config dataclasses (`ExperimentConfig`, `TrainingConfig`, `OptimizerConfig`,
`PredictionHeadModelConfig`, `PretrainedImageRepresentationModelConfig`) also live
in `schema.py` — see [architecture.md §4](architecture.md#4-configuration).

---

## `TensorflowV1ModelHandler` / `TensorflowV1ModelConfig` — `utils/tensorflow_model_handler.py`

**Responsibility:** download and describe pretrained backbones.

- `TensorflowV1ModelConfig` (dataclass) — one backbone's metadata: `architecture`,
  `graph_definition_path`, `download_url`, `input_node_name`, `output_node_name`,
  `output_tensor_size`, `image_input_size`, `image_depth`, `supports_batching`.
- `TensorflowV1ModelHandler(model_storage_path, tf_models_config_path)` — loads the
  registry (`configs/tf_models/tf_models.yaml`) into a `{architecture: config}`
  map and rewrites each `graph_definition_path` to an absolute on-disk path.
  - `get_model_info(architecture)` — returns the config (raises if unknown).
  - `get_model(architecture)` — returns the local model dir, downloading and
    extracting the tarball (with a logged progress bar) on a cache miss.

---

## `UPNAPreprocessor` — `dataset_preprocessors.py`

**Responsibility:** turn a raw dataset into the JSON manifest the trainer consumes.

Handles the UPNA datasets (real + synthetic), which ship as `.mp4` videos of 300
frames each plus tab-delimited ground-truth files
(`user_n_video_k_groundtruth3D.txt`) whose columns are
`Tx, Ty, Tz, Roll, Yaw, Pitch`. It uses the **non-zeroed** ground truth (absolute
angles, not relative-to-first-frame).

`preprocess_dataset()` walks each user's videos, extracts every frame as a JPEG,
and appends a manifest entry with the angles **divided by `ANGLE_SCALING_FACTOR =
180`** to scale degrees into `[-1, 1]`. Writes `preprocessed_ground_truth.json`.

**Entry point:** `scripts/python/run_dataset_preprocessing.py -c <config.yml>`. The
config lists datasets and their preprocessor import strings (resolved via
`cls_from_str`), so adding a preprocessor is a config edit plus a new class — see
[`configs/experiments/data_preprocessing_config.yml`](../configs/experiments/data_preprocessing_config.yml).
