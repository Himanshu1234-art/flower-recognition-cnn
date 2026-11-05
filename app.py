import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Validation accuracies of the two trained models
CNN_ACC = 77.66
MOBILENET_ACC = 91.96  # from flower_transfer.py (set to None to hide it)

MOBILENET_LABEL = "MobileNetV2 transfer learning" + (
    f" ({MOBILENET_ACC:.2f}%)" if MOBILENET_ACC is not None else ""
)

MODEL_FILES = {
    f"Custom CNN ({CNN_ACC:.2f}%)": "flower_cnn_model.keras",
    MOBILENET_LABEL: "flower_mobilenet_model.keras",
}

FLOWER_INFO = {
    "daisy": "Small flowers with white petals around a yellow center. Common in lawns and meadows.",
    "dandelion": "Bright yellow flowers that turn into round white seed heads you can blow away.",
    "roses": "Layered, cup-shaped blooms in red, pink, white and yellow. A classic symbol of love.",
    "sunflowers": "Tall plants with large yellow heads that turn to face the sun.",
    "tulips": "Cup-shaped spring flowers in many bright colors, grown from bulbs.",
}

FLOWER_FACTS = {
    "daisy": {"Colors": "White with a yellow center", "Season": "Spring to autumn",
              "Fun fact": "The name comes from 'day's eye' because the flower opens at dawn."},
    "dandelion": {"Colors": "Bright yellow, then white seed heads", "Season": "Spring to autumn",
                  "Fun fact": "Every part of a dandelion is edible, from the leaves to the roots."},
    "roses": {"Colors": "Red, pink, white, yellow, orange", "Season": "Late spring to autumn",
              "Fun fact": "There are over 300 species and thousands of cultivated varieties."},
    "sunflowers": {"Colors": "Yellow petals, brown center", "Season": "Summer to early autumn",
                   "Fun fact": "Young sunflowers follow the sun across the sky during the day."},
    "tulips": {"Colors": "Red, yellow, pink, purple, white", "Season": "Spring",
               "Fun fact": "Tulips were once so valuable in the Netherlands that single bulbs sold for huge sums."},
}

st.set_page_config(page_title="Flower Recognition", page_icon="🌸", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 700;}
    .sub-title {color: #6b7280; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
@st.cache_resource
def load_models():
    models, errors = {}, {}
    for label, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[label] = tf.keras.models.load_model(path)
            except Exception as e:  # noqa: BLE001
                errors[label] = str(e)
    return models, errors


def find_dataset():
    candidates = [
        Path("dataset/flower_photos/flower_photos"),
        Path("dataset/flower_photos"),
        Path.home() / ".keras/datasets/flower_photos/flower_photos",
        Path.home() / ".keras/datasets/flower_photos",
    ]
    for c in candidates:
        if c.exists() and (c / "roses").exists():
            return c
    return None


@st.cache_data
def class_counts(path_str):
    base = Path(path_str)
    return {c: len(list((base / c).glob("*.jpg"))) for c in CLASS_NAMES}


def to_probs(out):
    """Turn model output into probabilities.

    MobileNetV2 (flower_transfer.py) outputs raw logits, so softmax is needed.
    If a model already outputs probabilities (non-negative, sums to 1),
    they are used as they are, so softmax is not applied twice.
    """
    out = np.asarray(out, dtype="float64")
    if np.all(out >= 0) and np.isclose(out.sum(), 1.0, atol=1e-3):
        return out.astype("float32")
    e = np.exp(out - out.max())
    return (e / e.sum()).astype("float32")


def predict(m, img):
    # Both models contain their own preprocessing (Rescaling / preprocess_input),
    # so raw 0-255 pixel values are passed in.
    size = m.input_shape[1]
    arr = np.array(img.convert("RGB").resize((size, size)), dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    preds = m.predict(arr, verbose=0)
    return to_probs(preds[0])


def log_history(key, image_name, model_name, name, conf):
    seen = st.session_state.setdefault("seen_keys", set())
    if key in seen:
        return
    seen.add(key)
    st.session_state.setdefault("history", []).append(
        {
            "Time": time.strftime("%H:%M:%S"),
            "Image": image_name,
            "Model": model_name,
            "Prediction": name.capitalize(),
            "Confidence (%)": round(conf, 2),
        }
    )


def color_analysis(img):
    arr = np.array(img.resize((200, 200)))
    means = arr.reshape(-1, 3).mean(axis=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Average red", f"{means[0]:.0f}")
    c2.metric("Average green", f"{means[1]:.0f}")
    c3.metric("Average blue", f"{means[2]:.0f}")
    hist = {}
    for i, ch in enumerate(["Red", "Green", "Blue"]):
        counts, _ = np.histogram(arr[:, :, i], bins=32, range=(0, 255))
        hist[ch] = counts
    st.line_chart(pd.DataFrame(hist))


def make_report(image_name, model_name, probs):
    lines = [
        "Flower Recognition Report",
        "-------------------------",
        f"Image: {image_name}",
        f"Model: {model_name}",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Probabilities:",
    ]
    for i in np.argsort(probs)[::-1]:
        lines.append(f"  {CLASS_NAMES[i].capitalize():<12} {probs[i] * 100:6.2f}%")
    top = int(np.argmax(probs))
    lines.append("")
    lines.append(f"Prediction: {CLASS_NAMES[top].capitalize()} ({probs[top] * 100:.1f}% confidence)")
    return "\n".join(lines)


def show_result(img, image_name):
    probs = predict(model, img)
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    name = CLASS_NAMES[top]
    conf = probs[top] * 100
    log_history(f"{image_name}|{model_label}|{img.size}", image_name, model_label, name, conf)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=image_name, use_container_width=True)
    with col2:
        st.success(f"Prediction: **{name.capitalize()}**")
        st.caption(f"Model: {model_label}")
        st.metric("Confidence", f"{conf:.1f}%")
        st.progress(float(probs[top]))
        st.write(FLOWER_INFO[name])
        facts = FLOWER_FACTS[name]
        st.markdown(
            f"**Colors:** {facts['Colors']}  \n"
            f"**Season:** {facts['Season']}  \n"
            f"**Fun fact:** {facts['Fun fact']}"
        )
        if conf < 60:
            st.warning(
                "The model is not very sure about this image. Try a clearer, "
                "closer photo of a single flower."
            )

    st.subheader("Top 3 predictions")
    for i in order[:3]:
        st.write(f"{CLASS_NAMES[i].capitalize()}: {probs[i] * 100:.2f}%")

    st.subheader("All class probabilities")
    df = pd.DataFrame(
        {"Probability (%)": probs * 100},
        index=[c.capitalize() for c in CLASS_NAMES],
    )
    st.bar_chart(df)

    with st.expander("Image analysis (colors)"):
        color_analysis(img)

    st.download_button(
        "Download prediction report (.txt)",
        make_report(image_name, model_label, probs).encode("utf-8"),
        file_name="flower_report.txt",
        mime="text/plain",
        key=f"report_{image_name}",
    )


# ---------------- Quiz callbacks ----------------
def new_quiz():
    if dataset_dir is None:
        return
    cls = random.choice(CLASS_NAMES)
    files = list((dataset_dir / cls).glob("*.jpg"))
    st.session_state["quiz_path"] = str(random.choice(files))
    st.session_state["quiz_result"] = None


def submit_quiz():
    path = st.session_state.get("quiz_path")
    if not path:
        return
    actual = Path(path).parent.name
    guess = st.session_state.get("quiz_guess", CLASS_NAMES[0])
    probs = predict(model, Image.open(path).convert("RGB"))
    top = int(np.argmax(probs))
    model_guess = CLASS_NAMES[top]
    st.session_state["quiz_rounds"] = st.session_state.get("quiz_rounds", 0) + 1
    if guess == actual:
        st.session_state["quiz_user"] = st.session_state.get("quiz_user", 0) + 1
    if model_guess == actual:
        st.session_state["quiz_model"] = st.session_state.get("quiz_model", 0) + 1
    st.session_state["quiz_result"] = {
        "actual": actual,
        "guess": guess,
        "model_guess": model_guess,
        "model_conf": probs[top] * 100,
    }


# ---------------- Load models ----------------
models, load_errors = load_models()
if not models:
    st.error(
        "No trained model found. Run `python flower_cnn.py` (custom CNN) or "
        "`python flower_transfer.py` (MobileNetV2) first."
    )
    for k, v in load_errors.items():
        st.code(f"{k}: {v}")
    st.stop()

dataset_dir = find_dataset()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About this project")
    # Default to the last model in the list (MobileNetV2, the more accurate one)
    model_label = st.selectbox("Model", list(models.keys()), index=len(models) - 1)
    model = models[model_label]
    for k, v in load_errors.items():
        st.warning(f"Could not load {k}")
    st.write(
        "Deep learning models built with TensorFlow/Keras that recognize 5 types of flowers: "
        "a custom CNN and a MobileNetV2 transfer learning model."
    )
    st.metric("Custom CNN accuracy", f"{CNN_ACC:.2f}%")
    if MOBILENET_ACC is not None:
        st.metric(
            "MobileNetV2 accuracy",
            f"{MOBILENET_ACC:.2f}%",
            delta=f"+{MOBILENET_ACC - CNN_ACC:.2f} vs custom CNN",
        )
    st.write("**Classes:** " + ", ".join(c.capitalize() for c in CLASS_NAMES))
    st.write("**Dataset:** TensorFlow Flowers (about 3,600 images)")
    st.metric("Predictions this session", len(st.session_state.get("history", [])))
    st.caption("Made by Himanshu Mishra")

# ---------------- Header ----------------
st.markdown('<div class="main-title">🌸 Flower Recognition using CNN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload one or more flower photos, or try a sample image, and the model will tell you what they are.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🔍 Predict",
        "⚖️ Compare Models",
        "🎮 Flower Quiz",
        "🖼️ Dataset Explorer",
        "📊 Model Performance",
        "🕘 History",
        "ℹ️ About",
    ]
)

# ---------------- Tab 1: Predict ----------------
with tab1:
    uploaded = st.file_uploader(
        "Choose one or more flower images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.session_state.pop("sample_path", None)
        if len(uploaded) == 1:
            show_result(Image.open(uploaded[0]).convert("RGB"), uploaded[0].name)
        else:
            rows = []
            cols = st.columns(4)
            for i, f in enumerate(uploaded):
                img = Image.open(f).convert("RGB")
                probs = predict(model, img)
                top = int(np.argmax(probs))
                conf = probs[top] * 100
                log_history(f"{f.name}|{model_label}|{img.size}", f.name, model_label, CLASS_NAMES[top], conf)
                rows.append(
                    {
                        "Image": f.name,
                        "Prediction": CLASS_NAMES[top].capitalize(),
                        "Confidence (%)": round(conf, 2),
                    }
                )
                with cols[i % 4]:
                    st.image(
                        img,
                        caption=f"{CLASS_NAMES[top].capitalize()} ({conf:.1f}%)",
                        use_container_width=True,
                    )
            st.subheader("Results table")
            results = pd.DataFrame(rows)
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="flower_predictions.csv",
                mime="text/csv",
            )
    else:
        st.subheader("Or try a sample image")
        if dataset_dir is None:
            st.info("Dataset folder not found, so sample images are unavailable.")
        else:
            samples = []
            for c in CLASS_NAMES:
                imgs = sorted((dataset_dir / c).glob("*.jpg"))
                samples.extend(imgs[:2])
            cols = st.columns(5)
            for i, p in enumerate(samples):
                with cols[i % 5]:
                    st.image(str(p), use_container_width=True)
                    if st.button("Predict", key=f"sample_{i}"):
                        st.session_state["sample_path"] = str(p)

            sp = st.session_state.get("sample_path")
            if sp:
                st.divider()
                actual = Path(sp).parent.name
                st.caption(f"Actual class of this sample: {actual}")
                show_result(Image.open(sp).convert("RGB"), Path(sp).name)

# ---------------- Tab 2: Compare Models ----------------
with tab2:
    st.subheader("Compare models on one image")
    if len(models) < 2:
        st.info(
            "Only one model is loaded. Train flower_transfer.py to create "
            "flower_mobilenet_model.keras, then you can compare both models here."
        )
    cmp_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="cmp_upload")
    if cmp_file is not None:
        cmp_img = Image.open(cmp_file).convert("RGB")
        left, right = st.columns([1, 2])
        with left:
            st.image(cmp_img, caption=cmp_file.name, use_container_width=True)
        with right:
            cols = st.columns(len(models))
            predicted_names = []
            for col, (label, m) in zip(cols, models.items()):
                p = predict(m, cmp_img)
                t = int(np.argmax(p))
                predicted_names.append(CLASS_NAMES[t])
                with col:
                    st.markdown(f"**{label}**")
                    st.success(CLASS_NAMES[t].capitalize())
                    st.metric("Confidence", f"{p[t] * 100:.1f}%")
                    st.bar_chart(
                        pd.DataFrame(
                            {"Probability (%)": p * 100},
                            index=[c.capitalize() for c in CLASS_NAMES],
                        )
                    )
            if len(predicted_names) > 1:
                if len(set(predicted_names)) == 1:
                    st.success(f"Both models agree: {predicted_names[0].capitalize()}.")
                else:
                    st.warning("The models disagree on this image.")

# ---------------- Tab 3: Flower Quiz ----------------
with tab3:
    st.subheader("Can you beat the model?")
    if dataset_dir is None:
        st.info("Dataset folder not found, so the quiz is unavailable.")
    else:
        if "quiz_path" not in st.session_state:
            new_quiz()

        rounds = st.session_state.get("quiz_rounds", 0)
        s1, s2, s3 = st.columns(3)
        s1.metric("Rounds played", rounds)
        s2.metric("Your score", st.session_state.get("quiz_user", 0))
        s3.metric(f"Model score ({model_label.split(' (')[0]})", st.session_state.get("quiz_model", 0))

        q1, q2 = st.columns([1, 1])
        with q1:
            st.image(st.session_state["quiz_path"], use_container_width=True)
        with q2:
            result = st.session_state.get("quiz_result")
            if result is None:
                st.radio(
                    "Which flower is this?",
                    [c.capitalize() for c in CLASS_NAMES],
                    key="quiz_guess_label",
                )
                st.session_state["quiz_guess"] = st.session_state.get(
                    "quiz_guess_label", CLASS_NAMES[0].capitalize()
                ).lower()
                st.button("Submit answer", on_click=submit_quiz)
            else:
                if result["guess"] == result["actual"]:
                    st.success(f"You said {result['guess'].capitalize()}. Correct!")
                else:
                    st.error(f"You said {result['guess'].capitalize()}. Wrong.")
                if result["model_guess"] == result["actual"]:
                    st.success(
                        f"The model said {result['model_guess'].capitalize()} "
                        f"({result['model_conf']:.1f}%). Correct!"
                    )
                else:
                    st.error(
                        f"The model said {result['model_guess'].capitalize()} "
                        f"({result['model_conf']:.1f}%). Wrong."
                    )
                st.write(f"Actual flower: **{result['actual'].capitalize()}**")
                st.button("Next flower", on_click=new_quiz)

# ---------------- Tab 4: Dataset Explorer ----------------
with tab4:
    if dataset_dir is None:
        st.info("Dataset folder not found. Expected dataset/flower_photos/flower_photos in the project folder.")
    else:
        counts = class_counts(str(dataset_dir))
        st.subheader("Images per class")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(pd.DataFrame({"Images": list(counts.values())}, index=list(counts.keys())))
        with c2:
            st.metric("Total images", sum(counts.values()))
            st.metric("Classes", len(counts))
            st.write("Training/validation split: 80% / 20%")

        st.subheader("Browse the dataset")
        g1, g2, g3 = st.columns([2, 1, 1])
        with g1:
            chosen = st.selectbox("Flower class", CLASS_NAMES)
        with g2:
            n = st.slider("Images", 4, 12, 8)
        with g3:
            st.write("")
            if st.button("Shuffle"):
                st.session_state["gallery_seed"] = random.randint(0, 10_000)
        files = sorted((dataset_dir / chosen).glob("*.jpg"))
        rnd = random.Random(st.session_state.get("gallery_seed", 0))
        picked = rnd.sample(files, min(n, len(files)))
        cols = st.columns(4)
        for i, p in enumerate(picked):
            with cols[i % 4]:
                st.image(str(p), use_container_width=True)

# ---------------- Tab 5: Model Performance ----------------
with tab5:
    st.subheader("Classification report (Custom CNN)")
    report = pd.DataFrame(
        {
            "Precision": [0.77, 0.83, 0.65, 0.88, 0.73],
            "Recall": [0.74, 0.84, 0.68, 0.83, 0.76],
            "F1-score": [0.75, 0.83, 0.67, 0.85, 0.75],
            "Support": [129, 176, 120, 152, 157],
        },
        index=[c.capitalize() for c in CLASS_NAMES],
    )
    st.dataframe(report, use_container_width=True)
    st.caption(f"Overall validation accuracy: {CNN_ACC:.2f}% on 734 images.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Training curves")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.info("training_curves.png not found in the project folder.")
    with c2:
        st.subheader("Confusion matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("confusion_matrix.png not found in the project folder.")

    st.write(
        "Sunflowers and dandelions are recognized best. Roses are the hardest class "
        "because they look similar to tulips and daisies."
    )

    st.divider()
    st.subheader("Model comparison")
    comp_rows = [{"Model": "Custom CNN", "Approach": "Trained from scratch", "Validation accuracy (%)": CNN_ACC}]
    if MOBILENET_ACC is not None:
        comp_rows.append(
            {"Model": "MobileNetV2", "Approach": "Transfer learning + fine-tuning", "Validation accuracy (%)": MOBILENET_ACC}
        )
    comp = pd.DataFrame(comp_rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    if MOBILENET_ACC is not None:
        st.bar_chart(comp.set_index("Model")["Validation accuracy (%)"])
        st.write(
            f"Transfer learning improved validation accuracy by about "
            f"{MOBILENET_ACC - CNN_ACC:.1f} percentage points. Roses, the weakest class "
            f"of the custom CNN (F1 0.67), improved to an F1 of 0.88."
        )
    else:
        st.info("Train flower_transfer.py, then set MOBILENET_ACC at the top of app.py to compare both models.")

    m1, m2 = st.columns(2)
    with m1:
        if os.path.exists("mobilenet_training_curves.png"):
            st.subheader("MobileNetV2 training curves")
            st.image("mobilenet_training_curves.png", use_container_width=True)
    with m2:
        if os.path.exists("mobilenet_confusion_matrix.png"):
            st.subheader("MobileNetV2 confusion matrix")
            st.image("mobilenet_confusion_matrix.png", use_container_width=True)

# ---------------- Tab 6: History ----------------
with tab6:
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions yet. Upload an image or try a sample on the Predict tab.")
    else:
        hdf = pd.DataFrame(hist)
        st.dataframe(hdf, use_container_width=True)
        st.subheader("Predictions per flower")
        st.bar_chart(hdf["Prediction"].value_counts())
        h1, h2 = st.columns(2)
        with h1:
            st.download_button(
                "Download history as CSV",
                hdf.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with h2:
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.session_state["seen_keys"] = set()
                st.rerun()

# ---------------- Tab 7: About ----------------
with tab7:
    st.subheader("How it works")
    st.markdown(
        """
        1. The image is resized to the size the model expects (180 x 180 for the custom CNN, 224 x 224 for MobileNetV2).
        2. Convolution and pooling layers extract features such as edges, colors and petal shapes.
        3. Dropout and dense layers combine the features.
        4. The last layer gives a score for each of the 5 flowers, and softmax turns the scores into percentages.
        5. The class with the highest percentage is the prediction.
        """
    )
    st.subheader("Training details (custom CNN)")
    st.markdown(
        """
        - Optimizer: Adam
        - Loss: Sparse categorical cross-entropy
        - Data augmentation: flip, rotation, zoom, contrast
        - Callbacks: Early stopping and learning rate reduction
        """
    )
    st.subheader("Training details (MobileNetV2 transfer learning)")
    st.markdown(
        """
        - Base model: MobileNetV2 pretrained on ImageNet, input 224 x 224
        - Phase 1 (8 epochs): base frozen, only the new classifier is trained (Adam, learning rate 1e-3)
        - Phase 2 (up to 20 epochs in total): last 30 layers of the base unfrozen and fine-tuned (Adam, learning rate 1e-5)
        - Data augmentation: flip, rotation, zoom
        - Callbacks: Early stopping and learning rate reduction
        """
    )
    st.subheader("Limitations")
    st.write(
        "The models only know 5 flower types, so any other flower will still be "
        "forced into one of them. The custom CNN is about 78% accurate and MobileNetV2 "
        "about 92%, so some predictions will still be wrong."
    )