"""A flower care guide for the garden centre, with a model saved in step 5 of the project.

Run it from the repository folder: uv run streamlit run app_garden.py

Like app.py, it needs a model that takes the raw photos (pixel values 0 to 255) and outputs one probability per flower.
"""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

CLASSES = ["rose", "sunflower", "tulip"]  # the folder names, in alphabetical order, as in the training data

# Below this top probability the app does not name a flower. It was chosen on the 60 validation photos, never on the
# test set: 4 of the 9 validation mistakes of the fine-tuned model had a top probability below 0.6.
SURE = 0.6

# One palette per mood: page background, text, accent, card. "Auto" takes the mood of the predicted flower.
MOODS = {
    "Grounded": ("#121317", "#ece7dc", "#d9b44a", "#1b1d24"),
    "Smitten": ("#2a1220", "#f6e4ec", "#ff7aa8", "#3a1a2e"),
    "Cheerful": ("#2b2410", "#fff3c4", "#ffc61a", "#3a3116"),
    "Airy": ("#eef4fb", "#1e2a3a", "#5b8def", "#ffffff"),
    "Earth": ("#1d1a14", "#eadfcb", "#c27c4a", "#2a251c"),
}
FLOWER_MOOD = {"rose": "Smitten", "sunflower": "Cheerful", "tulip": "Airy"}

# General care guidance for the three guides the garden centre starts with
CARE = {
    "rose": [
        "**Water:** deeply at the base, in the morning, once or twice a week. Wet leaves invite black spot.",
        "**Cut back:** prune in late winter or early spring, and remove dead or crossing stems.",
        "**Watch for:** aphids on new shoots, black spot and powdery mildew on the leaves.",
    ],
    "sunflower": [
        "**Water:** deeply but not often, at the base. Roots reach deep once the plant is established.",
        "**Cut back:** nothing until the flower fades. Stake tall varieties against the wind.",
        "**Watch for:** slugs on seedlings, birds at the seed heads, powdery mildew late in summer.",
    ],
    "tulip": [
        "**Water:** lightly while the shoots grow, and not at all in soggy soil. Bulbs rot when they stay wet.",
        "**Cut back:** remove the flower, but leave the leaves until they turn yellow so the bulb can recharge.",
        "**Watch for:** bulb rot in wet ground, aphids, and grey mould on leaves in a damp spring.",
    ],
}


@st.cache_resource  # load each model only once, not again every time the page changes
def load_model(path):
    return keras.models.load_model(path)


def prepare_photo(photo_bytes, model):
    """Decode and resize a photo with TensorFlow, as image_dataset_from_directory did for the training photos."""
    image = tf.io.decode_image(photo_bytes, channels=3, expand_animations=False)
    height, width = model.input_shape[1:3]
    return tf.image.resize(image, (height, width))  # bilinear, the default of image_dataset_from_directory


def apply_mood(name):
    """Colour the page with one palette. The colours are set here, so the page needs no other style file."""
    background, text, accent, card = MOODS[name]
    st.markdown(
        f"""<style>
        .stApp {{ background: {background}; color: {text}; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp p, .stApp li {{ color: {text}; }}
        .stApp h1 {{ font-family: Georgia, serif; text-align: center; }}
        .card {{ background: {card}; border-left: 4px solid {accent}; padding: 0.8rem 1rem; border-radius: 6px;
                 margin: 0.6rem 0; color: {text}; }}
        .accent {{ color: {accent}; font-weight: 700; }}
        </style>""",
        unsafe_allow_html=True,
    )
    return accent


def card(text):
    st.markdown(f'<div class="card">{text}</div>', unsafe_allow_html=True)


model_files = sorted(Path("models").glob("*.keras"))
if not model_files:
    st.error("No model found in `models/`. Save a model there in step 5 of the notebook first.")
    st.stop()

# Photo first, because the "Auto" mood needs the prediction before the page is coloured
st.title("Know your flower. The honest way.")
mood_choice = st.pills("Choose a mood", ["Auto", *MOODS], default="Auto") or "Auto"
source = st.segmented_control("Photo from", ["Upload", "Camera"], default="Upload") or "Upload"
if source == "Camera":
    photo = st.camera_input("Take a photo of a rose, a sunflower or a tulip")
else:
    photo = st.file_uploader("A photo of a rose, a sunflower or a tulip", type=["jpg", "jpeg", "png"])

with st.expander("Model"):
    default = next((i for i, path in enumerate(model_files) if path.name == "fine_tuned.keras"), 0)
    model_path = st.selectbox("Model", model_files, index=default, format_func=lambda path: path.name)
model = load_model(str(model_path))

probabilities = None
if photo is not None:
    pixels = prepare_photo(photo.getvalue(), model)
    probabilities = model.predict(pixels[np.newaxis], verbose=0)[0]  # a batch of one photo
    flower = CLASSES[probabilities.argmax()]
    sure = probabilities.max() >= SURE

mood = mood_choice
if mood_choice == "Auto":  # until there is a photo, or while the model is not sure, the page rests in Grounded
    mood = FLOWER_MOOD[flower] if probabilities is not None and sure else "Grounded"
accent = apply_mood(mood)

if probabilities is None:
    card('<span class="accent">Tip</span> Move closer. One bloom, plain light, steady hand.')
else:
    st.image(photo.getvalue(), width=320)
    if sure:
        st.subheader(f"This looks like a {flower} ({probabilities.max():.0%})")
        for line in CARE[flower]:
            st.markdown(f"- {line}")
    else:
        first, second = np.argsort(probabilities)[::-1][:2]
        st.subheader("Not sure")
        card(
            f"The model's best guess is a {CLASSES[first]} ({probabilities[first]:.0%}), then a {CLASSES[second]} "
            f"({probabilities[second]:.0%}). That is too close to name a flower: a member of staff should look. "
            '<span class="accent">Tip</span> Move closer. One bloom, plain light, steady hand.'
        )
    st.bar_chart(pd.Series(probabilities, index=CLASSES, name="probability"), color=accent)
    card(
        "<b>Honest note:</b> the model knows only roses, sunflowers and tulips, and it is not always right. Photos of "
        "bunches or whole gardens fool it most, and a high percentage is no proof: some of its mistakes were "
        "confident ones."
    )
