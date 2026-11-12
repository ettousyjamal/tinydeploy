"""TensorFlow adapter tests; skipped automatically when TensorFlow is not installed."""

import pytest

tf = pytest.importorskip("tensorflow")
np = pytest.importorskip("numpy")

from tinydeploy import TinyDeployError  # noqa: E402
from tinydeploy.cli import main  # noqa: E402
from tinydeploy.compression import distill, int8_ptq, prune_dense  # noqa: E402

pytestmark = pytest.mark.tensorflow


def small_model(units=32):
    tf.keras.utils.set_random_seed(0)
    return tf.keras.Sequential([tf.keras.Input(shape=(16,)),
                                tf.keras.layers.Dense(units, activation="relu"),
                                tf.keras.layers.Dense(units, activation="relu"),
                                tf.keras.layers.Dense(4, activation="softmax")])


def data(n=128):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 16)).astype("float32")
    return x, (x[:, 0] > 0).astype("int64") + 2 * (x[:, 1] > 0).astype("int64")


def test_int8_ptq_produces_full_integer_model(tmp_path):
    x, _ = data()
    blob = int8_ptq(small_model(), x[:32], tmp_path / "m.tflite")
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]["dtype"] == np.int8
    assert interpreter.get_output_details()[0]["dtype"] == np.int8


def test_structured_pruning_shrinks_hidden_layers_only():
    model = small_model(units=40)
    pruned = prune_dense(model, 0.5)
    assert [layer.units for layer in pruned.layers] == [20, 20, 4]
    assert pruned.count_params() < model.count_params()
    x, _ = data(8)
    assert pruned(x).shape == (8, 4)


def test_pruning_rejects_unsupported_models():
    conv = tf.keras.Sequential([tf.keras.Input(shape=(8, 8, 1)), tf.keras.layers.Conv2D(4, 3)])
    with pytest.raises(TinyDeployError, match="Dense-only"):
        prune_dense(conv, 0.3)


def test_distillation_reduces_loss():
    x, y = data(256)
    teacher = small_model(64)
    teacher.compile("adam", "sparse_categorical_crossentropy")
    teacher.fit(x, y, epochs=5, verbose=0)
    history = distill(teacher, small_model(16), x, y, epochs=5, batch_size=32)
    assert history[-1]["loss"] < history[0]["loss"]


def test_compress_cli_writes_model_and_record(tmp_path):
    x, y = data()
    small_model().save(tmp_path / "m.keras")
    np.savez(tmp_path / "d.npz", x_calibration=x[:32], x_train=x, y_train=y)
    code = main(["compress", str(tmp_path / "m.keras"), "--data", str(tmp_path / "d.npz"),
                 "--prune", "0.5", "--output", str(tmp_path / "out.tflite")])
    assert code == 0
    assert (tmp_path / "out.tflite").stat().st_size > 0
    assert (tmp_path / "out.json").exists()
    assert main(["compress", str(tmp_path / "m.keras"), "--data", str(tmp_path / "d.npz"),
                 "--bits", "4", "--output", str(tmp_path / "int4.tflite")]) == 1
