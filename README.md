# Face Aging App

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-enabled-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
The **Face Aging App** is a deep learning-powered web application built with **Streamlit** that allows users to visualize how they might look older or younger. By combining tools like **Stable Diffusion inpainting** and facial masking techniques, this app seamlessly modifies facial features while preserving realism.

This project serves as both a functional demonstration of face transformation technology and a personal portfolio piece showcasing model integration, preprocessing pipelines, and UI deployment.

## Features
- Upload an image and visualize aged or de-aged versions
- Facial region detection using custom facial landmark logic
- Image masking and inpainting with diffusion models
- Optimized for local inference
- Lightweight architecture with optional model quantization and pruning

## Demo
> 📸 *Add a screenshot here showing before and after transformation*

## Project Structure
```
face-aging-app/
├── app.py                    # Streamlit web application
├── requirements.txt         # Python dependencies
├── models/
│   ├── pipeline.py          # Main face aging pipeline
│   └── quantize_prune.py    # Optional: model optimization code
├── utils/
│   └── face_mask.py         # Face region extraction and mask creation
├── README.md
```

## How It Works
1. **Upload Image**: The user uploads a frontal facial photo.
2. **Face Detection**: `face_mask.py` identifies facial landmarks using custom logic.
3. **Mask Creation**: A region to be aged is masked while preserving background.
4. **Transformation**: `pipeline.py` applies a pretrained diffusion-based model to inpaint the masked region.
5. **Stable Diffusion Inpainting**: We use the `stabilityai/stable-diffusion-2-inpainting` model from Hugging Face. This model takes a masked image, a binary mask, and a text prompt to guide the transformation. The prompt is necessary to instruct the model on what kind of inpainting is desired—e.g., "make this person look 20 years older". The combination of the visual input and prompt allows the model to generate age-progressed or regressed images with realistic textures and details.
6. **Quantization (Optional)**: For optimized deployment, `quantize_prune.py` enables lightweight inference.

## Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/face-aging-app.git
cd face-aging-app
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## Tech Stack
- **Python 3.10+**
- **Streamlit** for frontend
- **Stable Diffusion (via diffusers)** for image transformation
- **Torch + Transformers** for deep learning
- **OpenCV, NumPy, Matplotlib** for preprocessing and image operations

## Limitations
- Performance is hardware-dependent (no GPU = slower inference)
- Best results with clear, frontal face images
- May not generalize to extreme expressions or occluded faces

## Future Work
- Add option to control aging intensity
- Enable webcam support
- Expand to include hairstyle or lighting style changes
- Deploy to HuggingFace Spaces or Streamlit Cloud

## Acknowledgements & Credits
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion](https://stability.ai/)
- **Author**: [Anshul Raj](https://anshulraj.com/)

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

---
> Feel free to fork this project and modify it for your own face transformation tools or artistic applications!
