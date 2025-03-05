import os
import sys
import requests
import time
import torch
from pathlib import Path
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

# GPU diagnostics
print("\n=== GPU Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Explicitly set CUDA device and empty cache
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
else:
    print("WARNING: CUDA is not available. Using CPU instead (this will be much slower).")
    print("If you have a NVIDIA GPU, please check your drivers and PyTorch installation.")
    print("Recommended fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# Set device and precision
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
if FORCE_CPU:
    device = torch.device("cpu")
    print("Forcing CPU usage as requested via FORCE_CPU environment variable.")
else:
    # Try to force CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set the current CUDA device
        torch.cuda.set_device(0)
        # Empty cache to free up memory
        torch.cuda.empty_cache()
        print(f"Successfully set device to CUDA. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead (this will be much slower).")
        print("If you have a NVIDIA GPU, please install CUDA-compatible PyTorch:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

# Use float16 only if CUDA is available, otherwise use float32
dtype = torch.float32 if device.type == "cpu" else torch.float16
print(f"Using device: {device}")
print(f"Using precision: {dtype}")
print("=" * 50 + "\n")

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0

from PIL import Image
import cv2
import numpy as np

# Try to import RealESRGAN, but provide a fallback if it fails
try:
    from RealESRGAN.model import RealESRGAN
    REALESRGAN_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: RealESRGAN not available or incompatible with current PyTorch version: {e}")
    print("Using PIL for upscaling instead (lower quality).")
    REALESRGAN_AVAILABLE = False

import gradio as gr

# Try to import ImageSlider, but provide a fallback if it fails
USE_IMAGE_SLIDER = True
try:
    from gradio_imageslider import ImageSlider
except ImportError:
    # Silently fall back to standard image output
    USE_IMAGE_SLIDER = False

from huggingface_hub import hf_hub_download

# Configuration
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Create necessary directories
def create_directories():
    """Create necessary directories for model storage."""
    directories = [
        os.path.join(MODELS_DIR, "models", "Stable-diffusion"),
        os.path.join(MODELS_DIR, "upscalers"),
        os.path.join(MODELS_DIR, "embeddings"),
        os.path.join(MODELS_DIR, "Lora"),
        os.path.join(MODELS_DIR, "ControlNet"),
        os.path.join(MODELS_DIR, "VAE"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Check if models exist and download if they don't
def download_models():
    """Download required models if they don't exist locally."""
    create_directories()
    
    models = {
        "MODEL": ("dantea1118/juggernaut_reborn", "juggernaut_reborn.safetensors", os.path.join(MODELS_DIR, "models", "Stable-diffusion")),
        "UPSCALER_X2": ("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth", os.path.join(MODELS_DIR, "upscalers")),
        "UPSCALER_X4": ("ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth", os.path.join(MODELS_DIR, "upscalers")),
        "NEGATIVE_1": ("philz1337x/embeddings", "verybadimagenegative_v1.3.pt", os.path.join(MODELS_DIR, "embeddings")),
        "NEGATIVE_2": ("philz1337x/embeddings", "JuggernautNegative-neg.pt", os.path.join(MODELS_DIR, "embeddings")),
        "LORA_1": ("philz1337x/loras", "SDXLrender_v2.0.safetensors", os.path.join(MODELS_DIR, "Lora")),
        "LORA_2": ("philz1337x/loras", "more_details.safetensors", os.path.join(MODELS_DIR, "Lora")),
        "CONTROLNET": ("lllyasviel/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", os.path.join(MODELS_DIR, "ControlNet")),
        "VAE": ("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors", os.path.join(MODELS_DIR, "VAE")),
    }

    # Count how many models need to be downloaded
    models_to_download = []
    for model_name, (repo_id, filename, local_dir) in models.items():
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            models_to_download.append((model_name, repo_id, filename, local_dir))
    
    # If there are models to download, show a progress bar
    if models_to_download:
        print(f"Need to download {len(models_to_download)} models...")
        download_progress = tqdm(total=len(models_to_download), desc="Downloading models", unit="model")
        
        for model_name, repo_id, filename, local_dir in models_to_download:
            print(f"Downloading {model_name} from {repo_id} ({filename})...")
            download_progress.set_description(f"Downloading {model_name}")
            try:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
                print(f"✓ {model_name} downloaded successfully")
                download_progress.update(1)
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                raise
        
        download_progress.close()
        print("All models downloaded successfully!")
    else:
        print("All required models are already downloaded.")

def timer_func(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # Only print if VERBOSE_OUTPUT is enabled
        if os.environ.get("VERBOSE_OUTPUT", "0") == "1":
            print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class LazyLoadPipeline:
    """Lazy-loaded pipeline to reduce memory usage until needed."""
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        if self.pipe is None:
            load_progress = tqdm(total=2, desc="Loading pipeline", unit="step")
            self.pipe = self.setup_pipeline()
            load_progress.update(1)
            
            load_progress.set_description(f"Moving to {device}")
            self.pipe.to(device)
            
            if USE_TORCH_COMPILE and device.type == "cuda":
                load_progress.set_description("Compiling model")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            
            load_progress.update(1)
            load_progress.close()

    @timer_func
    def setup_pipeline(self):
        setup_progress = tqdm(total=10, desc="Setting up pipeline", unit="step")
        
        setup_progress.set_description("Loading ControlNet")
        controlnet = ControlNetModel.from_single_file(
            os.path.join(MODELS_DIR, "ControlNet", "control_v11f1e_sd15_tile.pth"), 
            torch_dtype=dtype
        )
        setup_progress.update(1)
        
        # Disable safety checker to prevent NSFW filtering
        safety_checker = None
        
        setup_progress.set_description("Loading base model")
        model_path = os.path.join(MODELS_DIR, "models", "Stable-diffusion", "juggernaut_reborn.safetensors")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=safety_checker,
            feature_extractor=None
        )
        setup_progress.update(3)
        
        setup_progress.set_description("Loading VAE")
        vae = AutoencoderKL.from_single_file(
            os.path.join(MODELS_DIR, "VAE", "vae-ft-mse-840000-ema-pruned.safetensors"),
            torch_dtype=dtype
        )
        pipe.vae = vae
        setup_progress.update(1)
        
        setup_progress.set_description("Loading embeddings")
        pipe.load_textual_inversion(os.path.join(MODELS_DIR, "embeddings", "verybadimagenegative_v1.3.pt"))
        pipe.load_textual_inversion(os.path.join(MODELS_DIR, "embeddings", "JuggernautNegative-neg.pt"))
        setup_progress.update(1)
        
        setup_progress.set_description("Loading LoRA 1")
        pipe.load_lora_weights(os.path.join(MODELS_DIR, "Lora", "SDXLrender_v2.0.safetensors"))
        pipe.fuse_lora(lora_scale=0.5)
        setup_progress.update(1)
        
        setup_progress.set_description("Loading LoRA 2")
        pipe.load_lora_weights(os.path.join(MODELS_DIR, "Lora", "more_details.safetensors"))
        pipe.fuse_lora(lora_scale=1.)
        setup_progress.update(1)
        
        setup_progress.set_description("Setting up scheduler")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        setup_progress.update(1)
        
        setup_progress.set_description("Enabling FreeU")
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        setup_progress.update(1)
        
        setup_progress.close()
        return pipe

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)

class LazyRealESRGAN:
    """Lazy-loaded RealESRGAN model."""
    def __init__(self, device, scale):
        self.device = device
        self.scale = scale
        self.model = None

    def load_model(self):
        if self.model is None and REALESRGAN_AVAILABLE:
            try:
                load_progress = tqdm(total=1, desc=f"Loading RealESRGAN x{self.scale}", unit="model")
                model_path = os.path.join(MODELS_DIR, "upscalers", f'RealESRGAN_x{self.scale}.pth')
                self.model = RealESRGAN(scale=self.scale, device=self.device)
                self.model.load_weights(model_path)
                load_progress.update(1)
                load_progress.close()
            except Exception as e:
                print(f"Error loading RealESRGAN: {e}")
                self.model = None
    
    def predict(self, img):
        self.load_model()
        if self.model is not None:
            try:
                # Convert PIL image to numpy array
                img_np = np.array(img)
                # Process with RealESRGAN
                result = self.model.predict(img_np)
                # Check if result is already a PIL Image
                if isinstance(result, Image.Image):
                    return result
                # Otherwise convert numpy array to PIL
                return Image.fromarray(result)
            except Exception as e:
                print(f"Error during RealESRGAN prediction: {e}")
                print("Falling back to PIL upscaling")
                w, h = img.size
                return img.resize((w * self.scale, h * self.scale), Image.LANCZOS)
        else:
            # Fallback to simple PIL upscaling if RealESRGAN is not available
            w, h = img.size
            return img.resize((w * self.scale, h * self.scale), Image.LANCZOS)

@timer_func
def resize_and_upscale(input_image, resolution):
    """Resize and upscale the input image to the target resolution."""
    upscale_progress = tqdm(total=3, desc="Upscaling", unit="step")
    
    scale = 2 if resolution <= 1024 else 4
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    
    upscale_progress.set_description("Resizing")
    upscale_progress.update(1)
    
    k = float(resolution) / min(H, W)
    H = int(round(H * k / 64.0)) * 64
    W = int(round(W * k / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    
    upscale_progress.set_description(f"RealESRGAN x{scale}")
    upscale_progress.update(1)
    
    if scale == 2:
        img = lazy_realesrgan_x2.predict(img)
    else:
        img = lazy_realesrgan_x4.predict(img)
    
    upscale_progress.set_description("Finalizing")
    upscale_progress.update(1)
    upscale_progress.close()
    
    return img

@timer_func
def create_hdr_effect(original_image, hdr):
    """Create an HDR-like effect on the image."""
    if hdr == 0:
        return original_image
        
    # Convert to numpy array for processing
    img_np = np.array(original_image)
    
    # Simple HDR effect using contrast and saturation enhancement
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Enhance saturation
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + 0.5 * hdr)
    
    # Enhance value (brightness)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + 0.3 * hdr)
    
    # Clip values to valid range
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(enhanced)

def prepare_image(input_image, resolution, hdr):
    """Prepare the image for processing by resizing, upscaling, and applying HDR effect."""
    condition_image = resize_and_upscale(input_image, resolution)
    condition_image = create_hdr_effect(condition_image, hdr)
    return condition_image

@timer_func
def process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, progress=gr.Progress()):
    """Process the input image with the pipeline."""
    if input_image is None:
        return None
    
    # Always show processing time in console
    print(f"\nProcessing image with resolution={resolution}, steps={num_inference_steps}, strength={strength:.2f}")
    processing_start_time = time.time()
    
    # Create console progress bar
    console_progress = tqdm(total=100, desc="Processing", unit="%", leave=True)
    
    # Save original stdout/stderr if we're in non-verbose mode
    if os.environ.get("VERBOSE_OUTPUT", "0") != "1":
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Create a null device to redirect output
        class NullIO:
            def write(self, *args, **kwargs):
                pass
            def flush(self, *args, **kwargs):
                pass
        # Redirect to null device (but keep tqdm output)
        sys.stdout = NullIO()
        sys.stderr = NullIO()
    
    # Custom progress callback that updates both Gradio and console progress bars
    def update_progress(step, total_steps, desc):
        # Update Gradio progress
        progress(step/total_steps, desc=desc)
        # Update console progress
        percent_complete = int(step/total_steps * 100)
        console_progress.n = percent_complete
        console_progress.set_description(desc)
        console_progress.refresh()
    
    update_progress(0, 1, "Initializing...")
    
    # Note: NSFW filtering is disabled in this application
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    update_progress(0.1, 1, "Preparing image...")
    condition_image = prepare_image(input_image, resolution, hdr)
    
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": condition_image,
        "control_image": condition_image,
        "width": condition_image.size[0],
        "height": condition_image.size[1],
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(device=device).manual_seed(0),
        "callback": lambda i, t, latents: update_progress(i + 1, num_inference_steps, f"Processing image... Step {i+1}/{num_inference_steps}"),
        "callback_steps": 1
    }
    
    update_progress(0.2, 1, "Running inference...")
    
    # Temporarily disable all warnings during inference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lazy_pipe(**options).images[0]
    
    update_progress(0.9, 1, "Finalizing...")
    
    # Convert input_image and result to numpy arrays
    input_array = np.array(input_image)
    result_array = np.array(result)
    
    # Restore stdout/stderr if we redirected them
    if os.environ.get("VERBOSE_OUTPUT", "0") != "1":
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    # Close the progress bar
    console_progress.close()
    
    # Calculate and display total processing time
    processing_time = time.time() - processing_start_time
    print(f"Total processing time: {processing_time:.2f} seconds")
    
    update_progress(1.0, 1, "Done!")
    if USE_IMAGE_SLIDER:
        return [input_array, result_array]
    else:
        return result_array

def show_info():
    """Display information about the application."""
    print("\n" + "="*80)
    print("Tile Upscaler - Image Upscaler with Tile Controlnet".center(80))
    print("="*80)
    print("\nThis application uses AI to enhance and upscale images.")
    print("It combines Stable Diffusion with Tile ControlNet for high-quality image enhancement.")
    print("\nFeatures:")
    print("- Upscale images to higher resolutions")
    print("- Enhance image details and quality")
    print("- Apply HDR effects")
    print("="*80 + "\n")

def main():
    """Main function to run the application."""
    # Set verbose output to false by default
    os.environ["VERBOSE_OUTPUT"] = "0"
    
    # Create a null device to redirect output if needed
    class NullIO:
        def write(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
    
    # Redirect stdout and stderr to null device if not in verbose mode
    if os.environ.get("VERBOSE_OUTPUT", "0") != "1":
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Redirect to null device
        sys.stdout = NullIO()
        sys.stderr = NullIO()
        
        # Function to restore output streams
        def restore_output():
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("Starting Gradio interface. Press Ctrl+C to exit.")
    else:
        def restore_output():
            pass
    
    show_info()
    
    # Temporarily restore output for model downloading
    if os.environ.get("VERBOSE_OUTPUT", "0") != "1":
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("Checking for required models...")
    
    # Download models if they don't exist
    download_models()
    
    # Redirect output again if not in verbose mode
    if os.environ.get("VERBOSE_OUTPUT", "0") != "1":
        sys.stdout = NullIO()
        sys.stderr = NullIO()
    
    # Initialize models
    global lazy_realesrgan_x2, lazy_realesrgan_x4, lazy_pipe
    lazy_realesrgan_x2 = LazyRealESRGAN(device, scale=2)
    lazy_realesrgan_x4 = LazyRealESRGAN(device, scale=4)
    lazy_pipe = LazyLoadPipeline()
    lazy_pipe.load()
    
    # Restore output for Gradio interface messages
    restore_output()
    
    # CSS for custom styling - using a dark theme
    css = """
    /* Dark theme enhancements */
    body {
        background-color: #111827 !important;
    }
    
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
        color: #E5E7EB !important;
        background-color: #111827 !important;
    }

    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 20px;
        border-radius: 10px;
        background-color: #1F2937;
        border: 1px solid #4B5563;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    }

    .main-header h1 {
        color: #3B82F6;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #E5E7EB;
        font-size: 1.1rem;
    }

    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #9CA3AF;
        padding: 15px;
        background-color: #1F2937;
        border-radius: 8px;
        border: 1px solid #4B5563;
    }

    .custom-button {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .custom-button:hover {
        background-color: #60A5FA !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    .info-block {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #4B5563;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }

    .info-block h3 {
        color: #3B82F6;
        margin-top: 0;
        font-weight: 600;
    }

    .info-block ol, .info-block ul {
        color: #E5E7EB;
        padding-left: 1.5rem;
        margin-bottom: 0;
    }

    .info-block li {
        margin-bottom: 0.5rem;
    }

    .tabs {
        margin-top: 20px;
    }

    /* Ensure all text has good contrast */
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #E5E7EB !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1F2937;
    }

    ::-webkit-scrollbar-thumb {
        background: #4B5563;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3B82F6;
    }
    """

    # Create the Gradio interface with improved UI
    with gr.Blocks(css=css, theme="dark") as demo:
        gr.HTML("""
            <div class="main-header">
                <h1>✨ Tile Upscaler ✨</h1>
                <p>Enhance and upscale your images with AI</p>
            </div>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class="info-block">
                        <h3>📋 Instructions</h3>
                        <ol>
                            <li>Upload an image using the panel on the left</li>
                            <li>Adjust settings in the tabs below if needed</li>
                            <li>Click "Enhance Image" to process</li>
                        </ol>
                    </div>
                """)
                
                input_image = gr.Image(
                    type="pil", 
                    label="Upload Image", 
                    elem_id="input_image",
                    height=400
                )
                
                with gr.Row():
                    run_button = gr.Button("✨ Enhance Image", elem_classes="custom-button")
                    clear_button = gr.Button("🗑️ Clear", variant="secondary")
                
            with gr.Column(scale=1):
                # Use ImageSlider if available, otherwise use standard Image output
                if USE_IMAGE_SLIDER:
                    output = ImageSlider(
                        label="Before / After (Slide to compare)", 
                        type="numpy",
                        height=500,
                        elem_id="output_slider"
                    )
                else:
                    output = gr.Image(
                        label="Enhanced Image", 
                        type="numpy",
                        height=500,
                        elem_id="output_image"
                    )
        
        with gr.Tabs(elem_classes="tabs") as tabs:
            with gr.TabItem("🔧 Basic Settings"):
                with gr.Row():
                    with gr.Column():
                        resolution = gr.Slider(
                            minimum=256, 
                            maximum=1024, 
                            value=512, 
                            step=256, 
                            label="Resolution",
                            info="Higher values produce larger images but require more processing time"
                        )
                    with gr.Column():
                        strength = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=0.4, 
                            step=0.01, 
                            label="Enhancement Strength",
                            info="Controls how much the AI modifies your image"
                        )
            
            with gr.TabItem("🔬 Advanced Settings"):
                with gr.Row():
                    with gr.Column():
                        num_inference_steps = gr.Slider(
                            minimum=1, 
                            maximum=50, 
                            value=20, 
                            step=1, 
                            label="Inference Steps",
                            info="More steps = better quality but slower processing"
                        )
                        hdr = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            value=0, 
                            step=0.1, 
                            label="HDR Effect",
                            info="Adds an HDR-like effect to the image"
                        )
                    with gr.Column():
                        guidance_scale = gr.Slider(
                            minimum=0, 
                            maximum=20, 
                            value=3, 
                            step=0.5, 
                            label="Guidance Scale",
                            info="Controls how closely the AI follows the prompt"
                        )
            
            with gr.TabItem("ℹ️ About"):
                gr.HTML("""
                    <div style="text-align: left; max-width: 800px; margin: 0 auto;">
                        <h3>Tile Upscaler</h3>
                        <p>This application uses AI to enhance and upscale images by combining Stable Diffusion with Tile ControlNet.</p>
                        
                        <h4>Features:</h4>
                        <ul>
                            <li>Upscale images to higher resolutions</li>
                            <li>Enhance image details and quality</li>
                            <li>Apply HDR effects for more vibrant images</li>
                            <li>Preserve the original structure while improving details</li>
                        </ul>
                        
                        <h4>Technical Details:</h4>
                        <p>The application uses:</p>
                        <ul>
                            <li>Stable Diffusion with Tile ControlNet</li>
                            <li>RealESRGAN for high-quality upscaling</li>
                            <li>Custom HDR processing for enhanced dynamic range</li>
                        </ul>
                    </div>
                """)
        
        gr.HTML("""
            <div class="footer">
                <p>Tile Upscaler v1.0 | Powered by Stable Diffusion and ControlNet</p>
            </div>
        """)
        
        # Set up event handlers
        run_button.click(
            fn=process_image, 
            inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale],
            outputs=output
        )
        
        clear_button.click(
            fn=lambda: (None, None) if USE_IMAGE_SLIDER else None,
            outputs=[input_image, output]
        )

    print("Starting Gradio interface. Press Ctrl+C to exit.")
    try:
        # Use server_name and server_port to avoid potential issues with some Gradio versions
        # Suppress Gradio warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            demo.launch(share=False, server_name="127.0.0.1", server_port=7860, quiet=True)
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
        print("\nTry installing compatible versions of gradio and related packages:")
        print("pip uninstall -y gradio pydantic")
        print("pip install gradio==3.32.0 pydantic<2.0.0")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["-v", "--verbose"]:
        os.environ["VERBOSE_OUTPUT"] = "1"
        print("Verbose output enabled")
    main()
