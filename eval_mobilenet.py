import pathlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = pathlib.Path(tf.keras.utils.get_file("flower_photos", origin=url, untar=True))
if (data_dir / "flower_photos").exists():
    data_dir = data_dir / "flower_photos"

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(224, 224), batch_size=32)
class_names = val_ds.class_names

model = tf.keras.models.load_model("flower_mobilenet_model.keras", compile=False)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\nValidation Accuracy: {acc * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}\n")

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=class_names, digits=2))