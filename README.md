# Image Upscale

**Note**: This application is a modified version of the [Tile-Upscaler](https://huggingface.co/spaces/gokaygokay/Tile-Upscaler) project. Special thanks to the original creator for their work in developing the initial version.

---

## Overview

Image Upscale is an AI-powered application designed to enhance and upscale images using advanced techniques like Stable Diffusion and Tile ControlNet. It provides high-quality image enhancement with options for HDR effects and customizable settings.

---

## Features

- **Image Upscaling**: Increase the resolution of images while maintaining quality.
- **Detail Enhancement**: Improve image details and overall quality.
- **HDR Effects**: Apply HDR-like effects for more vibrant and dynamic images.
- **Customizable Settings**: Adjust resolution, enhancement strength, inference steps, HDR effect, and guidance scale.

---

## Technical Details

- **Stable Diffusion with Tile ControlNet**: Utilizes advanced AI models for image processing.
- **RealESRGAN**: Provides high-quality upscaling.
- **Custom HDR Processing**: Enhances dynamic range for better image quality.

---

## Prerequisites

- Python 3.7 or higher
- NVIDIA GPU with CUDA support (recommended for better performance)
- **Torch**: You will need the following specific versions for GPU support:
  - `torch==2.0.0`
  - `torchvision==0.15.1`
  - `torchaudio==2.0.1`
  - Install via: 
    ```bash
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SUP3RMASS1VE/Image-Upscale.git
   cd Image-Upscale
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models**: The application will automatically download the necessary models on the first run.

---

## Usage

1. **Run the application**:

   ```bash
   python app.py
   ```

2. **Access the Gradio interface**:
   Open your web browser and go to [http://127.0.0.1:7860](http://127.0.0.1:7860).

3. **Upload and enhance images**:
   - Upload an image using the interface.
   - Adjust the settings as needed.
   - Click "Enhance Image" to process the image.

---

## Environment Variables

- **FORCE_CPU**: Set to `1` to force CPU usage even if a GPU is available.
- **VERBOSE_OUTPUT**: Set to `1` to enable detailed logging.

---

## Troubleshooting

- **CUDA not available**: Ensure that NVIDIA drivers and CUDA-compatible PyTorch are installed correctly.
- **Model download issues**: Ensure you have a working internet connection and try again if there are download failures.

---

## License

This project is licensed under the MIT License.

---

This update ensures that credit is given to the original creator, and the necessary dependency information for GPU support is included. Let me know if you need any further modifications!
