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
    pipe.unet = apply_structured_pruning(pipe.unet, amount=0.3)  # This still works
    return pipe

def get_prompt_for_age(age: int) -> str:
    if age <= 19:
        return f"a teenager, youthful face, smooth skin, bright eyes, soft lighting, photorealistic of age {age}"
    elif age <= 30:
        return f"a young adult, healthy skin, DSLR portrait, soft background, vibrant expression of age {age}"
    elif age <= 50:
        return f"a middle-aged person, slight wrinkles, mature expression, natural lighting of age {age}"
    elif age <= 70:
        return f"an older person, visible wrinkles, age spots, graying hair, photorealistic studio photo of age {age}"
    else:
        return f"an elderly person, deep wrinkles, sagging skin, white hair, wise expression, realistic of age {age}"

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