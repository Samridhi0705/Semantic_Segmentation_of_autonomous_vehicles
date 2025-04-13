import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from numpy import expand_dims
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="Image Segmentation Viewer", layout="wide")

# Title and description
st.title("Image Segmentation Dataset Viewer")
st.markdown("""
This application helps you explore and visualize image segmentation datasets.
Upload your dataset or use the default path to view RGB images and their corresponding segmentation masks.
""")

# Parameters
class_colors = {
    0: [0, 0, 0],       # Background (black)
    1: [0, 0, 255],     # Person (blue)
    2: [255, 255, 0],   # Bicycle (yellow)
    3: [255, 0, 0],     # Car (red)
    4: [0, 255, 0],     # Truck (green)
    5: [255, 255, 255], # Motorbike (white)
    # Add more classes as needed
}

# Helper Functions
def load_image_pixels(filename, shape=(416, 416)):
    '''
    Function preprocess the images to 416x416, which is the standard input shape for YOLOv3, 
    and also keeps track of the original shape, which is later used to draw the boxes.
    '''
    # Load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    
    # Load the image with the required size
    image = load_img(filename, target_size=shape)
    
    # Convert to numpy array
    image = img_to_array(image)
    
    # Scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    
    # Add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

def create_segmentation_image(mask, class_colors=None):
    '''
    Convert a class mask to a color-coded segmentation image
    '''
    if class_colors is None:
        class_colors = {
            0: [0, 0, 0],       # Background (black)
            1: [0, 0, 255],     # Class 1 (blue)
            2: [255, 255, 0],   # Class 2 (yellow)
            3: [255, 0, 0],     # Class 3 (red)
            4: [0, 255, 0],     # Class 4 (green)
            5: [255, 255, 255], # Class 5 (white)
        }
    
    # Create empty RGB image
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Fill with colors according to class
    unique_classes = np.unique(mask)
    
    for class_idx in unique_classes:
        if class_idx in class_colors:
            colored_mask[mask == class_idx] = class_colors[class_idx]
        else:
            # Use random color for unknown classes
            colored_mask[mask == class_idx] = np.random.randint(0, 255, 3)
    
    return colored_mask

def check_dataset_structure(data_path):
    '''
    Debug function to check dataset structure and availability of images.
    '''
    result = []
    result.append(f"Checking dataset at: {data_path}")
    
    if not os.path.exists(data_path):
        result.append(f"ERROR: Dataset path does not exist: {data_path}")
        return result
    
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    result.append(f"Found {len(folders)} folders: {folders}")
    
    total_rgb = 0
    total_seg = 0
    
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            
            rgb_path = os.path.join(subfolder_path, "CameraRGB")
            seg_path = os.path.join(subfolder_path, "CameraSeg")
            
            rgb_exists = os.path.exists(rgb_path)
            seg_exists = os.path.exists(seg_path)
            
            rgb_files = []
            seg_files = []
            
            if rgb_exists:
                rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                total_rgb += len(rgb_files)
            
            if seg_exists:
                seg_files = [f for f in os.listdir(seg_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                total_seg += len(seg_files)
            
            result.append(f"  {folder}/{subfolder}:")
            result.append(f"    CameraRGB: {'✓' if rgb_exists else '✗'} ({len(rgb_files) if rgb_exists else 0} images)")
            result.append(f"    CameraSeg: {'✓' if seg_exists else '✗'} ({len(seg_files) if seg_exists else 0} images)")
    
    result.append(f"\nTotal: {total_rgb} RGB images, {total_seg} segmentation masks")
    return result

def get_dataset_images(data_path):
    '''
    Get paths to RGB images and corresponding segmentation masks from the dataset.
    '''
    # Check if the path exists
    if not os.path.exists(data_path):
        st.error(f"Dataset path does not exist: {data_path}")
        return [], []
    
    # Folder paths for datasets
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # Collect image paths and segmentation mask paths
    rgb_paths = []  # Paths for original images
    seg_paths = []  # Paths for segmentation masks
    
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        # Check if there's a subfolder with the same name (as in the example)
        if os.path.exists(os.path.join(folder_path, folder)):
            subfolder_path = os.path.join(folder_path, folder)
            rgb_folder = os.path.join(subfolder_path, "CameraRGB")
            seg_folder = os.path.join(subfolder_path, "CameraSeg")
        else:
            # Try direct folder structure
            rgb_folder = os.path.join(folder_path, "CameraRGB")
            seg_folder = os.path.join(folder_path, "CameraSeg")
        
        if os.path.exists(rgb_folder) and os.path.exists(seg_folder):
            # Get files in both folders
            rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            seg_files = sorted([f for f in os.listdir(seg_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Match files if possible
            for i, rgb_file in enumerate(rgb_files):
                if i < len(seg_files):
                    rgb_paths.append(os.path.join(rgb_folder, rgb_file))
                    seg_paths.append(os.path.join(seg_folder, seg_files[i]))
    
    return rgb_paths, seg_paths

def plot_images_and_masks(rgb_paths, seg_paths, num_images=5):
    '''
    Create and return a matplotlib figure with original images, segmentation masks, and overlays
    '''
    # Limit to specified number of images
    rgb_paths = rgb_paths[:num_images]
    seg_paths = seg_paths[:num_images]
    
    if not rgb_paths:
        return None
    
    # Define grid layout
    rows = 3  # Original, Segmentation Mask, Overlay
    cols = len(rgb_paths)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 10))
    
    # Adjust for the case of only one image
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (rgb_path, seg_path) in enumerate(zip(rgb_paths, seg_paths)):
        # Load original image
        original = cv2.imread(rgb_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Load segmentation mask
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        if seg_mask is None:
            st.warning(f"Could not read segmentation mask: {seg_path}")
            seg_mask = np.zeros_like(original[:,:,0])  # Create empty mask
        
        # Create colored segmentation mask
        colored_mask = create_segmentation_image(seg_mask, class_colors)
        
        # Create overlay (blend original and mask)
        overlay = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)
        
        # Display original image
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Display segmentation mask
        axes[1, i].imshow(colored_mask)
        axes[1, i].set_title(f"Segmentation {i+1}")
        axes[1, i].axis('off')
        
        # Display overlay
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f"Overlay {i+1}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig

# Sidebar for settings
st.sidebar.header("Dataset Settings")

# Dataset path input
default_path = r"C:\Users\Sristi\OneDrive\Desktop\cv\Datasets"
data_path = st.sidebar.text_input("Dataset Path", value=default_path)

# Option to browse for a directory
st.sidebar.markdown("### Or upload sample images")
uploaded_rgb = st.sidebar.file_uploader("Upload RGB Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_seg = st.sidebar.file_uploader("Upload Segmentation Mask", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Display options
st.sidebar.header("Display Settings")
num_images = st.sidebar.slider("Number of Images to Display", min_value=1, max_value=10, value=3)

# Visualization options
visualization_type = st.sidebar.radio(
    "Visualization Type",
    ["Grid View", "Interactive View"]
)

# Button to check dataset structure
if st.sidebar.button("Check Dataset Structure"):
    st.subheader("Dataset Structure")
    with st.spinner("Analyzing dataset structure..."):
        results = check_dataset_structure(data_path)
        for line in results:
            st.text(line)

# Button to visualize dataset
if st.sidebar.button("Visualize Dataset"):
    st.subheader("Dataset Visualization")
    
    # Get paths to images and segmentation masks
    with st.spinner("Loading dataset images..."):
        rgb_paths, seg_paths = get_dataset_images(data_path)
        
        if not rgb_paths:
            st.warning("No images found. Please check your dataset paths.")
        else:
            st.success(f"Found {len(rgb_paths)} images and {len(seg_paths)} segmentation masks.")
            
            # Display a sample of the dataset
            with st.spinner("Generating visualizations..."):
                fig = plot_images_and_masks(rgb_paths, seg_paths, num_images=num_images)
                if fig:
                    st.pyplot(fig)

# Handle uploaded files if any
if uploaded_rgb and uploaded_seg:
    st.subheader("Uploaded Image Visualization")
    
    # Limit number of displayed images
    uploaded_rgb = uploaded_rgb[:num_images]
    uploaded_seg = uploaded_seg[:min(num_images, len(uploaded_seg))]
    
    if len(uploaded_rgb) != len(uploaded_seg):
        st.warning(f"Number of RGB images ({len(uploaded_rgb)}) doesn't match number of segmentation masks ({len(uploaded_seg)}). Using the minimum number.")
    
    # Create temporary directory for uploaded files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    rgb_paths = []
    seg_paths = []
    
    # Save uploaded files to temporary directory
    for i, (rgb_file, seg_file) in enumerate(zip(uploaded_rgb, uploaded_seg)):
        rgb_path = os.path.join(temp_dir, f"rgb_{i}.png")
        seg_path = os.path.join(temp_dir, f"seg_{i}.png")
        
        with open(rgb_path, "wb") as f:
            f.write(rgb_file.getbuffer())
        
        with open(seg_path, "wb") as f:
            f.write(seg_file.getbuffer())
        
        rgb_paths.append(rgb_path)
        seg_paths.append(seg_path)
    
    # Display the images
    with st.spinner("Generating visualizations for uploaded images..."):
        fig = plot_images_and_masks(rgb_paths, seg_paths, num_images=len(rgb_paths))
        if fig:
            st.pyplot(fig)

# Interactive view section
if visualization_type == "Interactive View" and st.button("Show Interactive View"):
    st.subheader("Interactive Image Viewer")
    
    # Get paths to images and segmentation masks
    with st.spinner("Loading dataset for interactive view..."):
        rgb_paths, seg_paths = get_dataset_images(data_path)
        
        if not rgb_paths:
            st.warning("No images found. Please check your dataset paths.")
        else:
            # Select an image to display
            image_index = st.slider("Select Image", 0, len(rgb_paths)-1, 0)
            
            # Load and display the selected image
            rgb_path = rgb_paths[image_index]
            seg_path = seg_paths[image_index]
            
            # Load original image
            original = cv2.imread(rgb_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Load segmentation mask
            seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            
            if seg_mask is None:
                st.warning(f"Could not read segmentation mask: {seg_path}")
                seg_mask = np.zeros_like(original[:,:,0])  # Create empty mask
            
            # Create colored segmentation mask
            colored_mask = create_segmentation_image(seg_mask, class_colors)
            
            # Create overlay (blend original and mask)
            overlay = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)
            
            # Display images in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(original, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(colored_mask, caption="Segmentation Mask", use_column_width=True)
            
            with col3:
                st.image(overlay, caption="Overlay", use_column_width=True)
            
            # Display mask statistics
            unique_classes = np.unique(seg_mask)
            st.write("Classes in this image:", unique_classes)
            
            # Display class legend
            st.subheader("Class Legend")
            legend_cols = st.columns(len(class_colors))
            for i, (class_id, color) in enumerate(class_colors.items()):
                with legend_cols[i % len(legend_cols)]:
                    # Create a small color swatch
                    swatch = np.ones((20, 40, 3), dtype=np.uint8)
                    swatch[:] = color
                    st.image(swatch, caption=f"Class {class_id}")

# Add a section to explain class colors
st.sidebar.header("Class Colors")
for class_id, color in class_colors.items():
    # Create a color swatch
    swatch = np.ones((20, 20, 3), dtype=np.uint8)
    swatch[:] = color
    # Convert to PIL Image
    swatch_pil = Image.fromarray(swatch)
    # Display in sidebar with class name
    col1, col2 = st.sidebar.columns([1, 3])
    with col1:
        st.image(swatch_pil)
    with col2:
        if class_id == 0:
            st.write("Background")
        elif class_id == 1:
            st.write("Person")
        elif class_id == 2:
            st.write("Bicycle")
        elif class_id == 3:
            st.write("Car")
        elif class_id == 4:
            st.write("Truck")
        elif class_id == 5:
            st.write("Motorbike")
        else:
            st.write(f"Class {class_id}")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use this application:
1. Set the path to your dataset in the sidebar
2. Click "Check Dataset Structure" to analyze your dataset
3. Adjust the number of images to display
4. Click "Visualize Dataset" to see images and their segmentation masks
5. Try "Interactive View" for a more detailed look at individual images

### Dataset format:
The application expects a dataset organized as follows:""")