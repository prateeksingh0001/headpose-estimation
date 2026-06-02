# Design Decisions

The *why* behind the non-obvious choices in this project - the things that won't be
self-evident from the code. Each entry is: the decision, the
reasoning, and the consequence it imposes on the rest of the codebase.

For the big picture see [architecture.md](architecture.md); for class detail see
[components.md](components.md).

---

## Two-phase training with bottleneck caching

**Decision.** Split training into (B) run every image through the frozen backbone
once and cache the feature vector to disk, then (C) train the head purely on the
cached vectors.

**Why.** The backbone forward pass is the expensive part, and because the backbone
is frozen its output never changes between epochs. Recomputing it every epoch would
waste almost all of the training time. Caching means the backbone runs exactly once
per image regardless of how many epochs the head trains for.

**Consequence.**
- Cached vectors are stored as compressed `.npz` files (NumPy array key `"arr_0"`,
  i.e. `DEFAULT_SAVE_NP_ARRAY_NAME`), one per datapoint.
- The manifest is updated with each cache path and re-saved, so later runs can skip
  generation by setting `run_bottleneck_generations: false`.
- The backbone and the head can be trained / reasoned about almost independently.

---

## Three independent heads with a summed loss

**Decision.** Predict yaw, pitch, and roll with **three separate MLP stacks**
(`yaw_predictor`, `pitch_predictor`, `roll_predictor`) rather than one shared head
with a 3-wide output. Train them jointly by summing their per-angle MSE losses into
a single `total_loss` minimized by one optimizer.

**Why.** Separate stacks let each angle learn its own feature weighting without
forcing weight-sharing across angles. Summing the losses into one optimizer step
keeps training simple (one `global_step`, one update) while still letting every head
learn each batch.

**Consequence.**
- Each head ends in a width-1 layer; hidden widths come from `layer_sizes`.
- `loss_ops` / `prediction_ops` are dicts; several methods read them via
  `.values()`, so **dict ordering matters**. In particular `train()` returns losses
  as `total, yaw, roll, pitch` (roll before pitch) — match that ordering when
  unpacking.

---

## Normalize angles to [-1, 1] during preprocessing

**Decision.** Scale ground-truth Euler angles from degrees into `[-1, 1]` by
dividing by `ANGLE_SCALING_FACTOR = 90` in the `EulerAnglePredictionHead`.

**Why.** Bounded, small-magnitude targets are friendlier for MSE regression and
keep the loss on a comparable scale across the three angles. Doing it once during
preprocessing means the cached manifest already carries normalized targets.

**Consequence.** Model outputs are in `[-1, 1]` units; multiply predictions by 180
to recover degrees. Any new dataset preprocessor must apply the same scaling so
targets stay consistent.

---

## Normalize pixels inside the backbone's preprocessing subgraph

**Decision.** Decode/resize/normalize images to `[-1, 1]` inside a TF subgraph that
`PretrainedBackBoneImageModel` prepends, using
`(pixel - 128) * 0.0078125` (i.e. `1/128`).

**Why.** This is the pixel range the MobileNet / Inception backbones were trained
with. Folding it into the graph keeps preprocessing co-located with the model that
requires it, rather than scattering it across the data pipeline, and guarantees
every image fed to the backbone is normalized identically.

**Consequence.** Callers feed **raw JPEG bytes** to the preprocessing placeholder;
they must not pre-normalize. The preprocessing subgraph and the frozen backbone
graph are kept separate (raw bytes → preprocess → feed backbone input node).

---

## Per-backbone batching: the `supports_batching` flag

**Decision.** Carry a `supports_batching` boolean per backbone in the registry, and
branch on it in `PretrainedBackBoneImageModel.predict`.

**Why.** **Inception V3 cannot accept batched input** at its resize node, so it must
be fed one image at a time; MobileNet V1 variants accept batched input and run much
faster batched. Encoding this as data (a config flag) rather than special-casing
class logic keeps the model code generic.

**Consequence.** `predict` `vstack`s and runs once for batching backbones, else
loops per image. When adding a backbone to `tf_models.yaml`, set this flag
correctly or you'll get shape errors (or leave throughput on the table).

---

## Export only the head as a frozen graph (for now)

**Decision.** `save_as_frozen_graph` exports just the trained prediction head to
`prediction_head.pb`, not a combined backbone+head graph.

**Why.** The head is the only part that changed during training; the backbone is a
known, separately downloadable artifact. Exporting only the head keeps the saved
model small. A combined export (stitching backbone + head into one inference graph)
is the intended V2 and is what `EulerAnglesPredictionModel` is meant to enable.

**Consequence.** Inference today requires loading both the backbone and the saved
head and chaining them. `EulerAnglesPredictionModel` is the container for that
chaining but currently has bugs (see
[components.md](components.md#euleranglespredictionmodel-end_to_end_modelspy)).
