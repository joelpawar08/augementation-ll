import os
import cv2
import numpy as np
import random
import math
import io
import requests
import uuid
import zipfile
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import streamlit as st
import albumentations as A
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear

def create_directories(save_dir):
    """Create directories if they don't exist"""
    os.makedirs(save_dir, exist_ok=True)

def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

def save_image(image, save_path):
    """Save an image to the specified path"""
    try:
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
                
        if hasattr(image, 'mode') and image.mode == "L":
            image.save(save_path)
        else:
            try:
                image.convert("RGB").save(save_path)
            except Exception:
                Image.fromarray(np.array(image)).save(save_path)
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def augment_with_albumentations(image, save_dir, prefix="alb", selected_augs=None):
    """Apply augmentations using Albumentations library"""
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    augmentations = {
        "rotate": A.Rotate(limit=45, p=1.0),
        "horizontal_flip": A.HorizontalFlip(p=1.0),
        "vertical_flip": A.VerticalFlip(p=1.0),
        "gauss_noise": A.GaussNoise(p=1.0),
        "random_gamma": A.RandomGamma(p=1.0),
        "blur": A.Blur(blur_limit=9, p=1.0),
        "random_scale": A.RandomScale(scale_limit=0.3, p=1.0),
        "shift_scale_rotate": A.ShiftScaleRotate(p=1.0),
        "random_rain": A.RandomRain(p=1.0),
        "random_fog": A.RandomFog(p=1.0),
        "random_sunflare": A.RandomSunFlare(p=1.0),
        "random_shadow": A.RandomShadow(p=1.0),
        "elastic_transform": A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20, p=1.0),
        "grid_distortion": A.GridDistortion(p=1.0),
        "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
    }
    
    augmented_images = []
    if selected_augs:
        for name in selected_augs:
            if name in augmentations:
                try:
                    transform = augmentations[name]
                    augmented = transform(image=image_cv)['image']
                    augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                    
                    unique_id = random.randint(10, 999999)
                    save_path = os.path.join(save_dir, f'{prefix}_{name}_{unique_id}.jpg')
                    
                    cv2.imwrite(save_path, cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR))
                    augmented_images.append((name, augmented_rgb, save_path))
                except Exception as e:
                    st.warning(f"Skipping albumentations {name}: {str(e)}")
    
    return augmented_images

def augment_with_keras(image, save_dir, prefix="keras", num_augs=5, selected_augs=None):
    """Apply augmentations using Keras ImageDataGenerator"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        if img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] == 1):
            st.warning("Skipping grayscale image for Keras augmentations")
            return []

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        x = np.expand_dims(img_array, axis=0)

        datagen_params = {}
        if selected_augs:
            if "keras_rotation" in selected_augs:
                datagen_params["rotation_range"] = 40
            if "keras_width_shift" in selected_augs:
                datagen_params["width_shift_range"] = 0.2
            if "keras_height_shift" in selected_augs:
                datagen_params["height_shift_range"] = 0.2
            if "keras_shear" in selected_augs:
                datagen_params["shear_range"] = 0.2
            if "keras_zoom" in selected_augs:
                datagen_params["zoom_range"] = 0.2
            if "keras_horizontal_flip" in selected_augs:
                datagen_params["horizontal_flip"] = True
            if "keras_brightness" in selected_augs:
                datagen_params["brightness_range"] = [0.8, 1.2]
            if "keras_channel_shift" in selected_augs:
                datagen_params["channel_shift_range"] = 20.0
            if "keras_vertical_flip" in selected_augs:
                datagen_params["vertical_flip"] = True

        datagen = ImageDataGenerator(
            fill_mode='nearest',
            validation_split=0.2,
            **datagen_params
        )

        augmented_images = []
        for i, batch in enumerate(datagen.flow(x, batch_size=1)):
            if i >= num_augs:
                break
                
            augmented_img = batch[0].astype(np.uint8)
            unique_id = random.randint(10, 999999)
            save_path = os.path.join(save_dir, f'{prefix}_aug_{i+1}_{unique_id}.jpg')
            
            Image.fromarray(augmented_img).save(save_path)
            augmented_images.append((f"keras_{i+1}", augmented_img, save_path))

        return augmented_images
    except Exception as e:
        st.error(f"Failed to apply Keras augmentations: {e}")
        return []

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an OpenCV image"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """Add salt and pepper noise to an OpenCV image"""
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image

def random_erase(image, erase_ratio=0.2):
    """Apply random erase to a numpy array image"""
    img_h, img_w = image.shape[:2]
    target_area = random.uniform(0.02, erase_ratio) * img_h * img_w
    aspect_ratio = random.uniform(0.3, 1/0.3)
    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))
    if h < img_h and w < img_w:
        x1 = random.randint(0, img_w - w)
        y1 = random.randint(0, img_h - h)
        if len(image.shape) == 3:
            image[y1:y1+h, x1:x1+w, :] = random.randint(0, 255)
        else:
            image[y1:y1+h, x1:x1+w] = random.randint(0, 255)
    return image

def cutout(image, n_holes=1, length=50):
    """Apply cutout to a numpy array image"""
    h, w = image.shape[:2]
    result = image.copy()
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        if len(image.shape) == 3:
            result[y1:y2, x1:x2, :] = 0
        else:
            result[y1:y2, x1:x2] = 0
    return result

def add_fog(image, fog_coeff=0.3):
    """Add fog effect to an OpenCV image"""
    fog = np.zeros_like(image, dtype=np.uint8)
    fog[:] = 255
    return cv2.addWeighted(image, 1 - fog_coeff, fog, fog_coeff, 0)

def add_rain(image, rain_drops=500, slant=20, drop_length=20, drop_width=2, drop_color=(200, 200, 200)):
    """Add rain effect to an OpenCV image"""
    result = image.copy()
    for _ in range(rain_drops):
        x = np.random.randint(0, image.shape[1] - slant)
        y = np.random.randint(0, image.shape[0] - drop_length)
        for j in range(drop_length):
            x_shift = int(j * slant / drop_length)
            if x + x_shift < image.shape[1] and y + j < image.shape[0]:
                cv2.line(result, (x + x_shift, y + j), (x + x_shift + drop_width, y + j), drop_color, 1)
    return result

def add_sunflare(image, num_flare_circles=8, threshold=200, flare_center=None):
    """Add sunflare effect to an RGB numpy array image"""
    result = image.copy()
    h, w = image.shape[:2]
    
    if flare_center is None:
        flare_center = (random.randint(0, w), random.randint(0, h // 2))
        
    for i in range(num_flare_circles):
        flare_size = random.randint(10, 30)
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        x = min(max(flare_center[0] + dx, 0), w - 1)
        y = min(max(flare_center[1] + dy, 0), h - 1)
        
        cv2.circle(
            result, 
            (x, y), 
            flare_size, 
            (random.randint(200, 255), random.randint(150, 255), random.randint(0, 150)), 
            -1
        )
    
    brightness_mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(brightness_mask, flare_center, min(h, w) // 2, 1.0, -1)
    brightness_mask = cv2.GaussianBlur(brightness_mask, (51, 51), 0)
    
    for i in range(3):
        result[:,:,i] = np.clip(result[:,:,i] + brightness_mask * 50, 0, 255)
    
    return result.astype(np.uint8)

def add_snow(image, snow_coeff=0.5):
    """Add snow effect to an OpenCV image"""
    result = image.copy()
    snow_layer = np.random.normal(0, 1, result.shape[:2])
    snow_layer = snow_layer / np.max(snow_layer)
    
    snow_threshold = 1 - snow_coeff
    snow_mask = snow_layer > snow_threshold
    
    result[snow_mask] = np.clip(result[snow_mask] + 50, 0, 255)
    
    return result

def compress_jpeg(image, quality_range=(10, 30)):
    """Compress a PIL image using JPEG compression"""
    buffer = io.BytesIO()
    quality = random.randint(quality_range[0], quality_range[1])
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def apply_custom_augmentations(image, save_dir, prefix="custom", selected_augs=None):
    """Apply custom augmentations"""
    augmented_images = []
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image)
        cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
    elif isinstance(image, Image.Image):
        pil_image = image
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        st.error("Unsupported image format for custom augmentations")
        return []
    
    augmentation_functions = {
        "saturation": lambda img: (ImageEnhance.Color(img).enhance(random.uniform(0.5, 2.5)), "pil"),
        "brightness": lambda img: (ImageEnhance.Brightness(img).enhance(random.uniform(1.5, 4.5)), "pil"),
        "rotation": lambda img: (img.rotate(random.choice([90, 180, 270])), "pil"),
        "resizing": lambda img: (img.resize((random.randint(50, 300), random.randint(50, 300))), "pil"),
        "black_white": lambda img: (img.convert("L"), "pil"),
        "flip_horizontal": lambda img: (ImageOps.mirror(img), "pil"),
        "flip_vertical": lambda img: (ImageOps.flip(img), "pil"),
        "perspective": lambda img: (img.transform(
            img.size,
            Image.PERSPECTIVE,
            (
                random.uniform(0, 0.1), random.uniform(0, 0.1),
                random.uniform(0, 0.1), random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0), random.uniform(0, 0.1),
                random.uniform(0.9, 1.0), random.uniform(0.9, 1.0)
            ),
            resample=Image.BICUBIC
        ), "pil"),
        "contrast": lambda img: (ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 2.0)), "pil"),
        "solarize": lambda img: (ImageOps.solarize(img, threshold=random.randint(0, 255)), "pil"),
        "posterize": lambda img: (ImageOps.posterize(img, bits=random.randint(1, 7)), "pil"),
        "equalize": lambda img: (ImageOps.equalize(img), "pil"),
        "edge_enhance": lambda img: (img.filter(ImageFilter.EDGE_ENHANCE), "pil"),
        "sharpen": lambda img: (img.filter(ImageFilter.SHARPEN), "pil"),
        "emboss": lambda img: (img.filter(ImageFilter.EMBOSS), "pil"),
        "jpeg_compression": lambda img: (compress_jpeg(img), "pil"),
        "pixelate": lambda img: (img.resize(
            (img.width // random.randint(5, 10), img.height // random.randint(5, 10)),
            Image.NEAREST
        ).resize((img.width, img.height), Image.NEAREST), "pil"),
        "random_crop": lambda img: (img.crop((
            random.randint(0, img.width // 4),
            random.randint(0, img.height // 4),
            random.randint(img.width * 3 // 4, img.width),
            random.randint(img.height * 3 // 4, img.height)
        )), "pil"),
        "blur": lambda img: (cv2.GaussianBlur(img, (35, 35), 0), "cv"),
        "median_blur": lambda img: (cv2.medianBlur(img, ksize=random.choice([9, 14, 18])), "cv"),
        "gaussian_noise": lambda img: (add_gaussian_noise(img), "cv"),
        "salt_pepper_noise": lambda img: (add_salt_pepper_noise(img), "cv"),
        "fog": lambda img: (add_fog(img), "cv"),
        "rain": lambda img: (add_rain(img), "cv"),
        "snow": lambda img: (add_snow(img), "cv"),
        "shear": lambda img: (Image.fromarray(cv2.cvtColor(
            np.array(random_shear(np.array(img), intensity=random.uniform(0.2, 0.5), 
                                row_axis=0, col_axis=1, channel_axis=2)), 
            cv2.COLOR_BGR2RGB)), "pil"),
        "random_erase": lambda img: (Image.fromarray(random_erase(np.array(img))), "pil"),
        "cutout": lambda img: (Image.fromarray(cutout(np.array(img))), "pil"),
        "sunflare": lambda img: (Image.fromarray(add_sunflare(np.array(img))), "pil")
    }

    augmented_images = []
    if selected_augs:
        for aug_name in selected_augs:
            if aug_name in augmentation_functions:
                try:
                    aug_func = augmentation_functions[aug_name]
                    img_to_use = pil_image if aug_func("dummy")[1] == "pil" else cv_image
                    aug_result, _ = aug_func(img_to_use if _ == "cv" else pil_image)
                    
                    if _ == "cv":
                        aug_result = Image.fromarray(cv2.cvtColor(aug_result, cv2.COLOR_BGR2RGB))
                        
                    unique_id = random.randint(100000, 999999)
                    save_path = os.path.join(save_dir, f"{prefix}_{aug_name}_{unique_id}.jpg")
                    
                    aug_result.save(save_path)
                    aug_result_np = np.array(aug_result)
                    augmented_images.append((aug_name, aug_result_np, save_path))
                    
                except Exception as e:
                    st.warning(f"Skipping {aug_name}: {str(e)}")
    
    return augmented_images

def create_zip_from_folder(folder_path, zip_name="augmented_images.zip"):
    """Create a ZIP file from a folder and return its binary content"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname=arcname)
    memory_file.seek(0)
    return memory_file.getvalue()

def main():
    st.set_page_config(page_title="Advanced Image Augmentation Tool", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        .stNumberInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        .stCheckbox {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2c3e50;
        }
        .stExpander {
            background-color: white;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🖼️ Advanced Image Augmentation Tool")
    st.markdown("Transform your images with a variety of augmentation techniques. Upload images or provide URLs, select your desired augmentations, and download the results in a convenient ZIP file.")

    # Input tabs
    tab1, tab2 = st.tabs(["📤 Upload Images", "🔗 Use Image URLs"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, help="Upload multiple images in JPG, JPEG, or PNG format.")

    with tab2:
        image_urls = st.text_area("Enter image URLs (one per line)", height=100, help="Enter URLs of images, one per line. Ensure the URLs are publicly accessible.")

    # Output directory and total augmentations
    save_dir = st.text_input("📂 Output Directory Name", value="augmented_images", 
                            help="This will be the name of the ZIP file containing augmented images")
    
    num_total_augs = st.number_input("🔢 Total Number of Augmented Images to Generate", 
                                   min_value=1, max_value=1000, value=100, 
                                   help="Total augmented images will be evenly distributed across all input images")

    # Augmentation Options
    st.markdown("### 🎨 Augmentation Options")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Albumentations")
        use_albumentations = st.checkbox("Enable Albumentations", value=False, help="Select to enable Albumentations library augmentations.")
    
    with col2:
        st.subheader("Keras")
        use_keras = st.checkbox("Enable Keras", value=False, help="Select to enable Keras ImageDataGenerator augmentations.")
    
    with col3:
        st.subheader("Custom")
        use_custom = st.checkbox("Enable Custom", value=False, help="Select to enable custom-designed augmentations.")
    
    with col4:
        st.subheader("Select Augmentations")
        all_augmentation_options = [
            "rotate", "horizontal_flip", "vertical_flip", "gauss_noise", "random_gamma",
            "blur", "random_scale", "shift_scale_rotate", "random_rain", "random_fog",
            "random_sunflare", "random_shadow", "elastic_transform", "grid_distortion", "clahe",
            "keras_rotation", "keras_width_shift", "keras_height_shift", "keras_shear",
            "keras_zoom", "keras_horizontal_flip", "keras_brightness", "keras_channel_shift",
            "keras_vertical_flip", "saturation", "brightness", "resizing", "black_white",
            "perspective", "contrast", "solarize", "posterize", "equalize", "edge_enhance",
            "sharpen", "emboss", "jpeg_compression", "pixelate", "random_crop", "median_blur",
            "salt_pepper_noise", "fog", "rain", "snow", "shear", "random_erase", "cutout", "sunflare"
        ]
        selected_augs = []
        with st.expander("Choose Augmentation Types", expanded=True):
            st.markdown("**Select specific augmentations to apply:**")
            for aug in all_augmentation_options:
                if st.checkbox(aug.replace("keras_", "").capitalize(), key=f"aug_{aug}"):
                    selected_augs.append(aug)

    # Advanced Settings
    with st.expander("🔍 Advanced Settings"):
        intensity = st.select_slider("Augmentation Intensity", options=["Low", "Medium", "High"], value="Medium", 
                                   help="Adjust the strength of applied augmentations.")

    # Generate Button
    if st.button("🚀 Generate Augmented Images"):
        temp_dir = save_dir
        create_directories(temp_dir)
        
        images_to_process = []
        
        if uploaded_files:
            for file in uploaded_files:
                try:
                    img = Image.open(file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image_name = file.name
                    images_to_process.append((img, image_name))
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
        
        if image_urls:
            for url in image_urls.strip().split('\n'):
                if url.strip():
                    img = load_image_from_url(url)
                    if img:
                        image_name = os.path.basename(url.split('?')[0]) or f"image_{uuid.uuid4().hex[:8]}.jpg"
                        images_to_process.append((img, image_name))
        
        if not images_to_process:
            st.error("No valid images found! Please upload images or provide valid URLs.")
            return
        
        if not selected_augs:
            st.error("Please select at least one augmentation type!")
            return

        # Distribute augmentations
        num_images = len(images_to_process)
        augs_per_image = max(1, num_total_augs // num_images) if num_images > 0 else num_total_augs
        
        # Categorize selected augmentations
        alb_augs = [aug for aug in selected_augs if aug in [
            "rotate", "horizontal_flip", "vertical_flip", "gauss_noise", "random_gamma",
            "blur", "random_scale", "shift_scale_rotate", "random_rain", "random_fog",
            "random_sunflare", "random_shadow", "elastic_transform", "grid_distortion", "clahe"
        ]]
        keras_augs = [aug for aug in selected_augs if aug in [
            "keras_rotation", "keras_width_shift", "keras_height_shift", "keras_shear",
            "keras_zoom", "keras_horizontal_flip", "keras_brightness", "keras_channel_shift",
            "keras_vertical_flip"
        ]]
        custom_augs = [aug for aug in selected_augs if aug in [
            "saturation", "brightness", "resizing", "black_white", "perspective",
            "contrast", "solarize", "posterize", "equalize", "edge_enhance",
            "sharpen", "emboss", "jpeg_compression", "pixelate", "random_crop",
            "median_blur", "salt_pepper_noise", "fog", "rain", "snow",
            "shear", "random_erase", "cutout", "sunflare"
        ]]
        
        total_methods = sum([
            (use_albumentations and alb_augs),
            (use_keras and keras_augs),
            (use_custom and custom_augs)
        ])
        
        if total_methods == 0:
            st.error("Please enable at least one augmentation method with selected types!")
            return
            
        augs_per_method = max(1, augs_per_image // total_methods) if total_methods > 0 else augs_per_image

        for idx, (image, image_name) in enumerate(images_to_process):
            image_folder = os.path.join(temp_dir, f"image_{idx+1}_{image_name.split('.')[0]}")
            create_directories(image_folder)
            
            st.markdown(f"### Processing Image: {image_name}")
            st.image(image, caption="Original Image", width=300)
            
            original_path = os.path.join(image_folder, f"original_{image_name}")
            image.save(original_path)
            
            all_augmented = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            current_method = 0
            
            if use_albumentations and alb_augs:
                status_text.text("Applying Albumentations...")
                alb_results = augment_with_albumentations(image, image_folder, selected_augs=alb_augs)
                all_augmented.extend(alb_results)
                current_method += 1
                progress_bar.progress(current_method / total_methods)
            
            if use_keras and keras_augs:
                status_text.text("Applying Keras augmentations...")
                keras_results = augment_with_keras(image, image_folder, num_augs=augs_per_method, selected_augs=keras_augs)
                all_augmented.extend(keras_results)
                current_method += 1
                progress_bar.progress(current_method / total_methods)
            
            if use_custom and custom_augs:
                status_text.text("Applying custom augmentations...")
                custom_results = apply_custom_augmentations(image, image_folder, selected_augs=custom_augs)
                all_augmented.extend(custom_results)
                current_method += 1
                progress_bar.progress(current_method / total_methods)
            
            progress_bar.progress(1.0)
            status_text.success(f"Generated {len(all_augmented)} augmented images for {image_name}!")
            
            if all_augmented:
                st.markdown("#### Sample of Augmented Images")
                sample_size = min(12, len(all_augmented))
                sample_augmentations = random.sample(all_augmented, sample_size)
                
                cols = st.columns(3)
                for i, (aug_name, aug_img, _) in enumerate(sample_augmentations):
                    cols[i % 3].image(aug_img, caption=aug_name.replace("keras_", "").capitalize(), width=200)
                
                st.success(f"Augmented images for {image_name} saved to: {image_folder}")
            else:
                st.warning("No augmentations generated for this image. Check your settings.")
        
        zip_content = create_zip_from_folder(temp_dir)
        st.balloons()
        st.success("✅ All images processed successfully!")
        st.download_button(
            label="📥 Download Augmented Images",
            data=zip_content,
            file_name=f"{save_dir}.zip",
            mime="application/zip",
            help="Download all augmented images as a ZIP file."
        )

if __name__ == "__main__":
    main()