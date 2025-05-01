import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

# ------------------- Core Otsu Functions -------------------
def calculate_local_otsu_threshold(image_block):
    pixels = image_block.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=[0, 256])
    total_pixels = pixels.size

    if total_pixels == 0 or np.all(pixels == pixels[0]):
        return 127

    prob = hist / total_pixels
    max_variance = 0
    optimal_threshold = 0
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))

    for t in range(256):
        prob0 = cum_prob[t]
        prob1 = 1.0 - prob0
        if prob0 == 0 or prob1 == 0:
            continue

        mean0 = cum_mean[t] / prob0 if prob0 > 0 else 0
        mean1 = (cum_mean[255] - cum_mean[t]) / prob1 if prob1 > 0 else 0
        between_class_variance = prob0 * prob1 * (mean0 - mean1) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t

    return optimal_threshold

def apply_local_otsu_thresholding(image: np.ndarray, block_size: int) -> np.ndarray:
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size must be odd and ≥3")
    
    height, width = image.shape
    binary_output = np.zeros_like(image)
    half_block = block_size // 2

    for i in range(height):
        for j in range(width):
            row_start = max(0, i - half_block)
            row_end = min(height, i + half_block + 1)
            col_start = max(0, j - half_block)
            col_end = min(width, j + half_block + 1)

            local_block = image[row_start:row_end, col_start:col_end]
            local_threshold = calculate_local_otsu_threshold(local_block)
            binary_output[i, j] = 255 if image[i, j] > local_threshold else 0

    return binary_output

def OtsuSegmentation(input: np.ndarray, method: str = 'global', block_size: int = 35) -> np.ndarray:
    if len(input.shape) == 3:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    
    if method == 'global':
        _, segmented = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'dynamic':
        segmented = apply_local_otsu_thresholding(input, block_size)
    else:
        raise ValueError("Method must be 'global' or 'dynamic'")
    
    return segmented

# ------------------- Image Handling -------------------
def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_image_pair(index: int):
    if not image_pairs or index >= len(image_pairs):
        return None, None, None
    
    input_path, global_path, dynamic_path = image_pairs[index]
    return (
        read_image(input_path),
        read_image(global_path),
        read_image(dynamic_path)
    )

def process_uploaded_image(uploaded_image: np.ndarray, method: str, block_size: int):
    if uploaded_image is None:
        return None
    
    if uploaded_image.shape[-1] == 4:
        uploaded_image = uploaded_image[..., :3]

    segmented = OtsuSegmentation(uploaded_image, method, block_size)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
    return segmented_rgb

def process_editor_image(edits, method, block_size):
    if not edits or not isinstance(edits, dict) or "composite" not in edits:
        return None
    composite = edits["composite"]
    return process_uploaded_image(composite, method, block_size)

# ------------------- Precomputed Loader -------------------
def load_image_pairs(input_dir: str, output_dir: str) -> List[Tuple[str, str, str]]:
    SUPPORTED_EXT = {'.webp', '.tiff', '.tif', '.png', '.jpg', '.jpeg'}
    
    input_files = [f for f in Path(input_dir).iterdir() 
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
    
    image_pairs = []
    for input_file in input_files:
        stem = input_file.stem
        suffix = input_file.suffix
        global_path = Path(output_dir) / f"{stem}_global{suffix}"
        dynamic_path = Path(output_dir) / f"{stem}_dynamic{suffix}"
        
        if global_path.exists() and dynamic_path.exists():
            image_pairs.append((
                str(input_file),
                str(global_path),
                str(dynamic_path)
            ))
    return sorted(image_pairs, key=lambda x: Path(x[0]).stem)

# ------------------- Gradio UI -------------------
image_pairs = load_image_pairs("input", "output")

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("Otsu Thresholding Demo")
        gr.Markdown("Compare *Global vs Dynamic* Otsu Thresholding")

        if image_pairs:
            with gr.Tab("Precomputed Examples"):
                gr.Markdown("### Gallery of Precomputed Results")
                index_slider = gr.Slider(0, len(image_pairs)-1, value=0, step=1, label="Image Index")
                with gr.Row():
                    input_display = gr.Image(label="Original Image", interactive=False)
                    global_display = gr.Image(label="Global Otsu", interactive=False)
                    dynamic_display = gr.Image(label="Dynamic Otsu", interactive=False)
                index_slider.change(
                    show_image_pair,
                    inputs=[index_slider],
                    outputs=[input_display, global_display, dynamic_display]
                )

        with gr.Tab("Live Image Thresholding"):
            gr.Markdown("### Upload and Edit an Image, then Click Process")
            with gr.Row():
                image_editor = gr.ImageEditor(label="Upload + Edit", type="numpy")
                segmented_output = gr.Image(label="Segmented Output")

            with gr.Row():
                method_dropdown = gr.Radio(choices=["global", "dynamic"], value="global", label="Thresholding Method")
                block_slider = gr.Slider(minimum=3, maximum=101, step=2, value=35, label="Block Size (for dynamic)")

            process_button = gr.Button("Process")
            process_button.click(
                process_editor_image,
                inputs=[image_editor, method_dropdown, block_slider],
                outputs=[segmented_output]
            )

    return demo

# ------------------- Launch -------------------
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()