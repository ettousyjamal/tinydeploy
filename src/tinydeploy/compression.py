"""Optional TensorFlow compression adapter: INT8 PTQ, structured pruning, distillation.

Install with ``pip install tinydeploy[tensorflow]``. TensorFlow is imported
lazily, so the rest of TinyDeploy keeps working without it.

Scope: full-integer INT8 post-training quantization,
magnitude-based structured (unit) pruning for Dense-only ``Sequential``
models, and teacher-student training. INT4 kernels, CNN channel pruning and
neural architecture search are out of scope and are rejected explicitly rather
than approximated. In the reported workflow these services came from Edge
Impulse / EON Tuner; this adapter lets the same grid run on open tooling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

from ._util import TinyDeployError


def _tf():
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TinyDeployError("this feature needs TensorFlow: pip install tinydeploy[tensorflow]") from exc
    return np, tf


def int8_ptq(model: Any, calibration: Iterable[Any], output: str | Path) -> bytes:
    """Convert a single-input Keras model to a full-integer INT8 ``.tflite`` file.

    ``calibration`` holds representative input samples (without batch axis),
    taken from the training split. Inputs and outputs are INT8, as expected by
    TFLM/CMSIS-NN. The model is exported as a SavedModel first, which works for
    both Keras 2 and Keras 3.
    """
    np, tf = _tf()
    calibration = list(calibration)
    if len(model.inputs) != 1 or not calibration:
        raise TinyDeployError("INT8 conversion needs a single-input model and calibration samples")

    def representative():
        for sample in calibration:
            yield [np.asarray(sample, dtype=np.float32)[np.newaxis]]

    with tempfile.TemporaryDirectory() as tmp:
        if hasattr(model, "export"):  # Keras 3
            try:
                model.export(tmp, verbose=False)
            except TypeError:
                model.export(tmp)
        else:  # Keras 2
            tf.saved_model.save(model, tmp)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        data = converter.convert()
    Path(output).write_bytes(data)
    return data


def prune_dense(model: Any, ratio: float) -> Any:
    """Remove the lowest-magnitude hidden units of a Dense-only ``Sequential`` model.

    Each hidden layer keeps ``round(units * (1 - ratio))`` units ranked by the
    L1 norm of their incoming weights; the next layer's kernel is sliced to
    match, and the output layer keeps its size. The result is a smaller dense
    model (no sparse kernels needed). Fine-tuning afterwards is the caller's
    choice.
    """
    np, tf = _tf()
    if not 0 <= ratio < 1:
        raise TinyDeployError("pruning ratio must be in [0, 1)")
    layers = model.layers
    if (not isinstance(model, tf.keras.Sequential) or not layers
            or any(not isinstance(layer, tf.keras.layers.Dense) for layer in layers)):
        raise TinyDeployError("structured pruning supports only Dense-only Sequential models")
    if any(not layer.use_bias for layer in layers):
        raise TinyDeployError("structured pruning requires bias-enabled Dense layers")
    pruned = tf.keras.Sequential([tf.keras.Input(shape=model.input_shape[1:])])
    incoming = np.arange(model.input_shape[-1])
    for index, layer in enumerate(layers):
        kernel, bias = layer.get_weights()
        keep = np.arange(layer.units)
        if index < len(layers) - 1:
            count = max(1, round(layer.units * (1 - ratio)))
            importance = np.sum(np.abs(kernel[incoming, :]), axis=0)
            keep = np.sort(np.argsort(importance, kind="stable")[-count:])
        config = layer.get_config()
        config["units"] = len(keep)
        reduced = tf.keras.layers.Dense.from_config(config)
        pruned.add(reduced)
        reduced.set_weights([kernel[np.ix_(incoming, keep)], bias[keep]])
        incoming = keep
    return pruned


def distill(teacher: Any, student: Any, x: Any, y: Any, *, epochs: int = 10,
            batch_size: int = 32, temperature: float = 3.0, alpha: float = 0.5,
            learning_rate: float = 1e-3, seed: int = 0,
            task: str = "classification") -> list[dict[str, float]]:
    """Train ``student`` against labels and a frozen ``teacher``.

    Classification: loss = alpha * CE(y, student) + (1 - alpha) * T^2 *
    KL(teacher_T || student_T), where both models output probabilities that
    are re-softened at temperature T. Reconstruction (autoencoders): loss =
    alpha * MSE(y, student) + (1 - alpha) * MSE(teacher, student).
    Returns the mean training loss per epoch.
    """
    np, tf = _tf()
    if temperature <= 0 or not 0 <= alpha <= 1 or epochs < 1 or batch_size < 1:
        raise TinyDeployError("invalid distillation settings")
    if task not in ("classification", "reconstruction"):
        raise TinyDeployError("task must be 'classification' or 'reconstruction'")
    tf.keras.utils.set_random_seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    rng = np.random.default_rng(seed)

    def soften(p):
        return tf.nn.softmax(tf.math.log(tf.clip_by_value(p, 1e-7, 1.0)) / temperature)

    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = tf.convert_to_tensor(x[idx]), tf.convert_to_tensor(y[idx])
            target = tf.stop_gradient(teacher(xb, training=False))
            with tf.GradientTape() as tape:
                pred = student(xb, training=True)
                if task == "classification":
                    hard = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(yb, pred))
                    p, q = soften(target), soften(pred)
                    kl = tf.reduce_sum(p * (tf.math.log(p + 1e-7) - tf.math.log(q + 1e-7)), axis=-1)
                    soft = tf.reduce_mean(kl) * temperature ** 2
                else:
                    hard = tf.reduce_mean(tf.square(tf.cast(yb, pred.dtype) - pred))
                    soft = tf.reduce_mean(tf.square(target - pred))
                loss = alpha * hard + (1 - alpha) * soft
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            total += float(loss) * len(idx)
        history.append({"epoch": epoch + 1, "loss": total / len(order)})
    return history
