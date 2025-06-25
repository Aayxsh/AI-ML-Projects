import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import os
import sys
import torch
from PIL import Image
from datetime import datetime
import time
import clip
import tensorflow_hub as hub
import whisper
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline
)

# Configuration for local execution
AUDIO_FILE_PATH = "path/to/your/audio.wav"  # Update this path
OUTPUT_DIR = "./AI_Generated_Images"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure GPU
print("Configuring GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU configured: {len(physical_devices)} GPU(s) found")
        device = "cuda"
        torch_dtype = torch.float16
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        device = "cpu"
        torch_dtype = torch.float32
else:
    print("No GPU found. Running on CPU.")
    device = "cpu"
    torch_dtype = torch.float32

# Load CLIP model globally for efficiency
print("Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded successfully")

# Function to download YAMNet class map
def get_yamnet_class_map():
    class_map_path = 'yamnet_class_map.csv'
    if not os.path.exists(class_map_path):
        print("Downloading YAMNet class map...")
        import urllib.request
        url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        urllib.request.urlretrieve(url, class_map_path)
    return class_map_path

# Load YAMNet model
print("Loading YAMNet model...")
try:
    with tf.device('/cpu:0'):
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet model loaded successfully on CPU")
except Exception as e:
    print(f"Error loading YAMNet model: {e}")
    yamnet_model = None

def load_yamnet_class_names():
    class_names_path = get_yamnet_class_map()
    df = pd.read_csv(class_names_path, header=0)
    return df.iloc[:, 2].tolist()

def classify_environmental_sounds(audio_path, threshold=0.15):
    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = waveform.astype(np.float32)
        chunk_size = 5 * sr
        all_scores = []

        for i in range(0, len(waveform), chunk_size):
            chunk = waveform[i:i + chunk_size]
            if len(chunk) < sr:
                continue
            chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
            with tf.device('/cpu:0'):
                scores, _, _ = yamnet_model(chunk_tensor)
            all_scores.append(scores)

        if not all_scores:
            return []

        combined_scores = tf.concat(all_scores, axis=0)
        mean_scores = tf.reduce_mean(combined_scores, axis=0).numpy()
        class_names = load_yamnet_class_names()
        detected_classes = [
            (class_names[i], float(mean_scores[i]))
            for i in np.where(mean_scores > threshold)[0]
            if i < len(class_names)
        ]
        return sorted(detected_classes, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Error in classify_environmental_sounds: {e}")
        return []

# Load Whisper model
print("Loading Whisper model...")
try:
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("base", device=whisper_device)
    print(f"Whisper model loaded on {whisper_device}")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

def transcribe_with_validation(audio_path, min_confidence=0.5, min_words=3):
    try:
        if whisper_model is None:
            return {'text': '', 'confidence': 0, 'is_valid': False}
        result = whisper_model.transcribe(audio_path)
        transcription = result.get("text", "").strip()
        segments = result.get("segments", [])
        confidences = [seg.get('confidence', 0.7) for seg in segments]
        avg_confidence = np.mean(confidences) if confidences else 0.8
        is_valid = (
            avg_confidence >= min_confidence and
            len(transcription.split()) >= min_words and
            any(c.isalpha() for c in transcription)
        )
        return {
            'text': transcription,
            'confidence': avg_confidence,
            'is_valid': is_valid
        }
    except Exception as e:
        print(f"Error in transcribe_with_validation: {e}")
        return {'text': '', 'confidence': 0, 'is_valid': False}

def create_image_prompt(audio_path, env_threshold=0.15):
    try:
        env_results = classify_environmental_sounds(audio_path, threshold=env_threshold) if yamnet_model else []
        stt_result = transcribe_with_validation(audio_path)

        # Compute audio metrics
        avg_sound_confidence = np.mean([score for _, score in env_results]) if env_results else 0
        audio_metrics = {
            'avg_sound_confidence': avg_sound_confidence,
            'transcription_confidence': stt_result['confidence'],
            'is_transcription_valid': stt_result['is_valid']
        }

        # Create prompt based on analysis
        if stt_result['is_valid']:
            prompt = f"A realistic scene depicting: {stt_result['text']}"
            if env_results:
                env_desc = ", ".join([label for label, _ in env_results[:3]])
                prompt += f", with environmental sounds of {env_desc}"
        elif env_results:
            env_desc = ", ".join([label for label, _ in env_results[:5]])
            prompt = f"A realistic scene with {env_desc}"
        else:
            prompt = "A realistic environmental scene"

        # Add quality enhancers
        prompt += ", high quality, detailed, photorealistic"

        return prompt, audio_metrics
    except Exception as e:
        print(f"Error in create_image_prompt: {e}")
        return "Realistic natural environment scene", {
            'avg_sound_confidence': 0, 
            'transcription_confidence': 0, 
            'is_transcription_valid': False
        }

class ImageGenerator:
    def __init__(self):
        self.base_model_loaded = False
        self.upscaler_loaded = False
        self.base_model = None
        self.upscaler = None

    def load_base_model(self):
        if not self.base_model_loaded:
            print("Loading Stable Diffusion XL model...")
            try:
                self.base_model = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch_dtype,
                    variant="fp16" if torch_dtype == torch.float16 else None,
                    use_safetensors=True
                )
                self.base_model.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.base_model.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True
                )
                self.base_model = self.base_model.to(device)
                self.base_model.enable_attention_slicing()
                if torch.cuda.is_available():
                    self.base_model.enable_model_cpu_offload()
                self.base_model_loaded = True
                print("Base model loaded successfully")
            except Exception as e:
                print(f"Error loading base model: {e}")
                raise

    def generate_image(self, prompt, negative_prompt=None, guidance_scale=7.5, steps=30, width=1024, height=1024, num_images=1):
        if not self.base_model_loaded:
            self.load_base_model()
        
        if negative_prompt is None:
            negative_prompt = "deformed, bad anatomy, blurry, low quality, low resolution, distorted"

        start_time = time.time()
        try:
            images = self.base_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_images_per_prompt=num_images
            ).images
        except Exception as e:
            print(f"Error generating images: {e}")
            return [], {'generation_time': 0, 'image_paths': []}
        
        gen_time = time.time() - start_time

        # Save images and collect paths
        image_paths = []
        for idx, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"image_{timestamp}_{idx}.png")
            img.save(filename)
            image_paths.append(filename)
            print(f"Saved image: {filename}")

        return images, {'generation_time': gen_time, 'image_paths': image_paths}

def compute_clip_score(image, text):
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            similarity = (image_features @ text_features.T).item()
        return similarity
    except Exception as e:
        print(f"Error computing CLIP score: {e}")
        return 0.0

def audio_to_image(audio_file_path, num_images=1):
    start_time = time.time()

    # Validate audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    print(f"Processing audio file: {audio_file_path}")

    # Generate prompt and collect audio metrics
    prompt, audio_metrics = create_image_prompt(audio_file_path)
    print(f"Generated prompt: {prompt}")

    # Generate images
    generator = ImageGenerator()
    images, image_metrics = generator.generate_image(prompt, num_images=num_images)

    if not images:
        print("No images were generated")
        return [], prompt, {
            'total_execution_time': time.time() - start_time,
            'success': False,
            'error': 'Image generation failed'
        }

    # Compute CLIP scores
    clip_scores = [compute_clip_score(img, prompt) for img in images]

    # Compile all metrics
    total_time = time.time() - start_time
    metrics = {
        'total_execution_time': total_time,
        'generation_time': image_metrics['generation_time'],
        'avg_sound_confidence': audio_metrics['avg_sound_confidence'],
        'transcription_confidence': audio_metrics['transcription_confidence'],
        'is_transcription_valid': audio_metrics['is_transcription_valid'],
        'prompt_length': len(prompt.split()),
        'clip_scores': clip_scores,
        'avg_clip_score': np.mean(clip_scores) if clip_scores else 0,
        'image_paths': image_metrics['image_paths'],
        'success': True
    }

    return images, prompt, metrics

def main():
    # Update this path to your audio file
    audio_file = input("Enter path to audio file (or press Enter for default): ").strip()
    if not audio_file:
        audio_file = AUDIO_FILE_PATH
    
    if not os.path.exists(audio_file):
        print(f"ERROR: File not found at {audio_file}")
        print("Please update the AUDIO_FILE_PATH variable or provide a valid path")
        return

    print(f"\nUsing audio file: {audio_file}")
    print("\nProcessing audio to generate images...")
    
    try:
        images, prompt, metrics = audio_to_image(audio_file, num_images=1)

        print("\n=== AUDIO ANALYSIS AND IMAGE GENERATION COMPLETE ===")
        print(f"Prompt: {prompt}")
        print(f"Generated {len(images)} images in {OUTPUT_DIR}")

        print("\n=== METRICS ===")
        print(f"Total Execution Time: {metrics['total_execution_time']:.2f} seconds")
        print(f"Image Generation Time: {metrics.get('generation_time', 0):.2f} seconds")
        print(f"Avg Sound Confidence: {metrics['avg_sound_confidence']:.2f}")
        print(f"Transcription Confidence: {metrics['transcription_confidence']:.2f}")
        print(f"Transcription Valid: {metrics['is_transcription_valid']}")
        print(f"Prompt Length: {metrics['prompt_length']} words")
        print(f"CLIP Scores: {[f'{score:.2f}' for score in metrics.get('clip_scores', [])]}")
        print(f"Avg CLIP Score: {metrics['avg_clip_score']:.2f}")
        print(f"Success: {metrics['success']}")
        
        if 'image_paths' in metrics:
            print(f"Saved images:")
            for path in metrics['image_paths']:
                print(f"  - {path}")
        
        print("=================")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Metrics collection incomplete due to failure")

if __name__ == "__main__":
    main()
