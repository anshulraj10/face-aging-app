import streamlit as st
import torch
from PIL import Image
from model.pipeline import load_pipeline
from utils.face_mask import detect_face_mask
from model.quantize_prune import apply_structured_pruning

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource
def get_model():
    pipe = load_pipeline(device)
    pipe.unet = apply_structured_pruning(pipe.unet, amount=0.3)
    return pipe

def get_prompt_for_age(age: int) -> str:
    age_prompts = {
        range(0, 3): (
            "a baby with soft, delicate skin, round cheeks, a small nose, sparkling eyes full of curiosity, and a gentle expression, "
            "realistic lighting, based on the original photo, showing age {age}"
        ),
        range(3, 7): (
            "a cheerful toddler with a joyful smile, round face, smooth skin, wide bright eyes, and playful posture, "
            "based on the original photo, showing age {age}"
        ),
        range(7, 13): (
            "a lively child with smooth skin, expressive eyes, a curious and energetic look, neatly combed hair, and a natural pose, "
            "based on the original photo, showing age {age}"
        ),
        range(13, 17): (
            "a young teenager with youthful skin, confident gaze, subtle facial features developing maturity, natural hairstyle, "
            "balanced lighting, based on the original photo, showing age {age}"
        ),
        range(17, 21): (
            "an older teenager with fresh skin, vibrant features, trendy hairstyle, relaxed expression, slight smile, and realistic lighting, "
            "based on the original photo, showing age {age}"
        ),
        range(21, 30): (
            "a young adult with clear, glowing skin, symmetrical facial features, expressive eyes, casual stylish outfit, "
            "high-resolution image, based on the original photo, showing age {age}"
        ),
        range(30, 40): (
            "an adult with a mature and confident expression, subtle smile lines, even skin tone, well-groomed hair, and a composed demeanor, "
            "realistic portrait, based on the original photo, showing age {age}"
        ),
        range(40, 50): (
            "a middle-aged adult with visible laugh lines, slight wrinkles around the eyes and mouth, a poised expression, and a touch of gray hair, "
            "realistic textures, based on the original photo, showing age {age}"
        ),
        range(50, 60): (
            "a late middle-aged person with noticeable facial lines, mature skin, gentle eyes, composed look, wearing simple yet elegant clothing, "
            "clear lighting, based on the original photo, showing age {age}"
        ),
        range(60, 70): (
            "a senior with gray hair, defined wrinkles, kind eyes, gentle smile, wearing traditional or modest clothing, dignified posture, "
            "natural lighting, based on the original photo, showing age {age}"
        ),
        range(70, 85): (
            "an elderly person with deep facial lines, thinning white hair, calm and wise expression, warm eyes, upright or slightly hunched posture, "
            "photorealistic portrait with soft tones, based on the original photo, showing age {age}"
        ),
        range(85, 101): (
            "a very elderly person with sagging skin, pronounced wrinkles, serene face, fragile features, soft and wise gaze, and realistic lighting, "
            "captured in a respectful, detailed portrait style, based on the original photo, showing age {age}"
        )
    }

    for age_range, prompt in age_prompts.items():
        if age in age_range:
            return prompt.format(age=age)
    return f"a person of age {age}, detailed, realistic portrait with accurate age characteristics, based on the original photo"

pipe = get_model()

st.title("Face Aging App")
st.write("Upload your face image and adjust the slider to make yourself younger or older.")

uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
age_target = st.slider("Age Target (years)", min_value=10, max_value=80, value=30, step=1)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    mask = detect_face_mask(image)

    prompt = get_prompt_for_age(age_target)

    with st.spinner("Processing..."):
        result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5).images[0]
        st.image(result, caption="Transformed Face", use_column_width=True)