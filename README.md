# Tile Upscaler

Tile Upscaler is an AI-powered application designed to enhance and upscale images using advanced techniques like Stable Diffusion and Tile ControlNet. It provides high-quality image enhancement with options for HDR effects and customizable settings.

## Features

- **Image Upscaling**: Increase the resolution of images while maintaining quality.
- **Detail Enhancement**: Improve image details and quality.
- **HDR Effects**: Apply HDR-like effects for more vibrant images.
- **Customizable Settings**: Adjust resolution, enhancement strength, inference steps, HDR effect, and guidance scale.

## Technical Details

- **Stable Diffusion with Tile ControlNet**: Utilizes advanced AI models for image processing.
- **RealESRGAN**: Provides high-quality upscaling.
- **Custom HDR Processing**: Enhances dynamic range for better image quality.

## Prerequisites

- Python 3.7 or higher
- NVIDIA GPU with CUDA support (recommended for better performance)
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Tile-Upscaler
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models**:
   The application will automatically download necessary models on the first run.

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Access the Gradio interface**:
   - Open your web browser and go to `http://127.0.0.1:7860`.

3. **Upload and enhance images**:
   - Upload an image using the interface.
   - Adjust settings as needed.
   - Click "Enhance Image" to process.

## Environment Variables

- `FORCE_CPU`: Set to `1` to force CPU usage even if a GPU is available.
- `VERBOSE_OUTPUT`: Set to `1` to enable detailed logging.

## Troubleshooting

- **CUDA not available**: Ensure NVIDIA drivers and CUDA-compatible PyTorch are installed.
- **Model download issues**: Check internet connection and retry.

## License

This project is licensed under the MIT License. 
