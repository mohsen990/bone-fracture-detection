import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.models.ensemble_model import EnsembleModel
from src.data.preprocessing import get_val_transforms
from src.utils.config import Config

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try to import GradCAM (optional)
try:
    from src.evaluation.gradcam import GradCAM, get_target_layer
    GRADCAM_AVAILABLE = True
except:
    GRADCAM_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fractured {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .info-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .detection-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def get_available_models():
    """Scan for available trained models"""
    config = Config()
    model_dir = Path(config.MODEL_DIR)

    available_models = {}

    # Check for individual classifier models
    model_files = {
        'ResNet-50': 'best_model.pth',
        'ResNet-101': 'resnet101_best_model.pth',
        'EfficientNet-B0': 'efficientnet_b0_best_model.pth',
        'EfficientNet-B3': 'efficientnet_b3_best_model.pth',
    }

    for display_name, filename in model_files.items():
        model_path = model_dir / filename
        if model_path.exists():
            available_models[display_name] = str(model_path)

    # Check for ensemble
    ensemble_path = model_dir / 'ensemble_weighted_best.pth'
    if ensemble_path.exists():
        available_models['Ensemble (Best)'] = str(ensemble_path)

    return available_models


def get_available_yolo_models():
    """Scan for available YOLO detection models"""
    if not YOLO_AVAILABLE:
        return {}

    yolo_models = {}

    # Check common YOLO model locations (improved model first)
    yolo_paths = [
        Path(parent_dir) / 'runs' / 'detect' / 'fracture_detection_improved' / 'weights' / 'best.pt',
        Path(parent_dir) / 'runs' / 'detect' / 'fracture_detection' / 'weights' / 'best.pt',
        Path(parent_dir) / 'runs' / 'detect' / 'runs' / 'detect' / 'fracture_detection' / 'weights' / 'best.pt',
        Path(parent_dir) / 'runs' / 'detect' / 'runs' / 'detect' / 'fracture_detection_improved' / 'weights' / 'best.pt',
        Path(parent_dir) / 'fracture_detection' / 'yolov8n_fracture' / 'weights' / 'best.pt',
        Path(parent_dir) / 'fracture_detection' / 'yolov8s_fracture' / 'weights' / 'best.pt',
        Path(parent_dir) / 'fracture_detection' / 'yolov8m_fracture' / 'weights' / 'best.pt',
        Path(parent_dir) / 'yolo_fracture_best.pt',
        Path(parent_dir) / 'models' / 'yolo_best.pt',
    ]

    for path in yolo_paths:
        if path.exists():
            model_name = f"YOLO ({path.parent.parent.name})"
            yolo_models[model_name] = str(path)

    # Also check for any .pt files in runs/detect
    runs_detect = Path(parent_dir) / 'runs' / 'detect'
    if runs_detect.exists():
        for folder in runs_detect.iterdir():
            if folder.is_dir():
                best_pt = folder / 'weights' / 'best.pt'
                if best_pt.exists():
                    yolo_models[f"YOLO ({folder.name})"] = str(best_pt)

    return yolo_models


@st.cache_resource
def load_classifier_model(model_name, model_path):
    """Load a trained classifier model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # Handle ensemble model separately
    if 'ensemble' in model_name.lower():
        return load_ensemble_model(model_path, device, config)

    # Map display name to model type
    model_type_map = {
        'ResNet-50': 'resnet50',
        'ResNet-101': 'resnet101',
        'EfficientNet-B0': 'efficientnet_b0',
        'EfficientNet-B3': 'efficientnet_b3',
    }

    model_type = model_type_map.get(model_name, 'resnet50')

    # Create model
    if 'resnet' in model_type.lower():
        model = create_resnet_model(model_type, num_classes=config.NUM_CLASSES)
    elif 'efficientnet' in model_type.lower():
        model = create_efficientnet_model(model_type, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get model info
    model_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
    }

    return model, device, model_info


def load_ensemble_model(ensemble_path, device, config):
    """Load ensemble model by loading individual models first"""
    model_dir = Path(config.MODEL_DIR)

    # Define individual model paths
    individual_models_config = [
        {'name': 'resnet50', 'path': model_dir / 'best_model.pth'},
        {'name': 'resnet101', 'path': model_dir / 'resnet101_best_model.pth'},
        {'name': 'efficientnet_b3', 'path': model_dir / 'efficientnet_b3_best_model.pth'},
    ]

    # Load individual models
    models = []
    for model_config in individual_models_config:
        model_name = model_config['name']
        model_path = model_config['path']

        if not model_path.exists():
            raise FileNotFoundError(f"Individual model not found: {model_path}")

        # Create model architecture
        if 'resnet' in model_name:
            model = create_resnet_model(model_name, num_classes=config.NUM_CLASSES)
        else:
            model = create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)

    # Load ensemble checkpoint to get optimal weights
    ensemble_checkpoint = torch.load(ensemble_path, map_location=device)
    weights = ensemble_checkpoint.get('weights', [1/len(models)] * len(models))

    # Create ensemble model
    ensemble = EnsembleModel(models, weights=weights)
    ensemble = ensemble.to(device)
    ensemble.eval()

    model_info = {
        'epoch': 'N/A',
        'best_val_acc': 'Ensemble',
        'weights': weights,
        'num_models': len(models),
    }

    return ensemble, device, model_info


@st.cache_resource
def load_yolo_model(model_path):
    """Load a trained YOLO model"""
    if not YOLO_AVAILABLE:
        return None
    model = YOLO(model_path)
    return model


def preprocess_image(image, img_size=224):
    """Preprocess uploaded image for classification"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)

    # Apply transforms
    transform = get_val_transforms(img_size)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)

    return image_tensor, image_np


def predict_classification(model, image_tensor, device):
    """Make classification prediction"""
    config = Config()

    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    return pred_class, confidence, probs[0].cpu().numpy()


def predict_detection(yolo_model, image_np, conf_threshold=0.25):
    """Make YOLO detection prediction and return annotated image"""
    if yolo_model is None:
        return None, []

    # Run detection
    results = yolo_model.predict(
        source=image_np,
        conf=conf_threshold,
        save=False,
        verbose=False
    )

    detections = []
    annotated_image = image_np.copy()

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Get class name
            class_name = results[0].names.get(cls, f'Class {cls}')

            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

            # Draw bounding box - RED for fracture
            color = (255, 0, 0)  # Red in RGB
            thickness = 3

            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 2)

            cv2.rectangle(annotated_image,
                         (x1, y1 - label_h - 10),
                         (x1 + label_w + 5, y1),
                         color, -1)

            # Draw label text
            cv2.putText(annotated_image, label,
                       (x1 + 2, y1 - 5),
                       font, font_scale, (255, 255, 255), 2)

    return annotated_image, detections


def generate_gradcam(model, image_tensor, original_image, pred_class, device, model_name):
    """Generate Grad-CAM visualization"""
    if not GRADCAM_AVAILABLE:
        return None, None

    try:
        model_type = model_name.lower().replace('-', '').replace(' ', '')
        target_layer = get_target_layer(model, model_type)
        gradcam = GradCAM(model, target_layer)

        superimposed, heatmap = gradcam.visualize(
            image_tensor.to(device),
            original_image,
            target_class=pred_class,
            alpha=0.4
        )

        return superimposed, heatmap
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None, None


def extract_boxes_from_gradcam(model, image_tensor, original_image, pred_class, device, model_name, threshold=0.5):
    """
    Extract bounding boxes from Grad-CAM heatmap for classification models.
    This provides approximate localization of the fracture area.
    """
    if not GRADCAM_AVAILABLE:
        return None, []

    try:
        model_type = model_name.lower().replace('-', '').replace(' ', '')
        target_layer = get_target_layer(model, model_type)
        gradcam = GradCAM(model, target_layer)

        # Get the raw CAM
        cam = gradcam.generate_cam(image_tensor.to(device), target_class=pred_class)

        # Resize CAM to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Normalize to 0-255
        cam_normalized = ((cam_resized - cam_resized.min()) /
                         (cam_resized.max() - cam_resized.min() + 1e-8) * 255).astype(np.uint8)

        # Apply threshold to find high activation regions
        thresh_value = int(threshold * 255)
        _, binary = cv2.threshold(cam_normalized, thresh_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw boxes on image
        annotated_image = original_image.copy()
        boxes = []

        # Filter small contours and get bounding boxes
        min_area = (h * w) * 0.01  # Minimum 1% of image area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, box_w, box_h = cv2.boundingRect(contour)
                boxes.append({
                    'bbox': (x, y, x + box_w, y + box_h),
                    'area': area,
                    'confidence': float(cam_resized[y:y+box_h, x:x+box_w].mean())
                })

                # Draw bounding box - Orange for CAM-based detection
                color = (255, 165, 0)  # Orange in RGB
                thickness = 3
                cv2.rectangle(annotated_image, (x, y), (x + box_w, y + box_h), color, thickness)

                # Draw label
                label = f"Region (CAM)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 2)

                cv2.rectangle(annotated_image,
                             (x, y - label_h - 10),
                             (x + label_w + 5, y),
                             color, -1)

                cv2.putText(annotated_image, label,
                           (x + 2, y - 5),
                           font, font_scale, (255, 255, 255), 2)

        return annotated_image, boxes

    except Exception as e:
        st.warning(f"Could not extract boxes from Grad-CAM: {e}")
        return None, []


def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🦴 Bone Fracture Detection System")
        st.markdown(
            "<p style='text-align: center; color: gray;'>AI-Powered X-ray Analysis with Classification & Detection</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Get available models
    available_classifiers = get_available_models()
    available_yolo = get_available_yolo_models()

    # Mode selection
    st.sidebar.subheader("🎯 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose analysis type",
        ["Classification Only", "Detection Only", "Both (Classification + Detection)"],
        help="Classification: Determines if fracture exists\nDetection: Locates fracture areas with bounding boxes"
    )

    # Classifier selection
    selected_classifier = None
    if analysis_mode in ["Classification Only", "Both (Classification + Detection)"]:
        st.sidebar.subheader("🤖 Classification Model")
        if available_classifiers:
            selected_classifier = st.sidebar.selectbox(
                "Choose classifier",
                list(available_classifiers.keys()),
                help="Select which trained model to use for classification"
            )
        else:
            st.sidebar.warning("No classifier models found")

    # YOLO selection
    selected_yolo = None
    if analysis_mode in ["Detection Only", "Both (Classification + Detection)"]:
        st.sidebar.subheader("🔍 Detection Model (YOLO)")
        if available_yolo:
            selected_yolo = st.sidebar.selectbox(
                "Choose YOLO model",
                list(available_yolo.keys()),
                help="Select YOLO model for fracture localization"
            )
        elif YOLO_AVAILABLE:
            st.sidebar.info("No trained YOLO models found. Using pre-trained YOLOv8.")
            selected_yolo = "YOLOv8n (Pre-trained)"
        else:
            st.sidebar.error("YOLO not available. Install ultralytics package.")

    # Detection settings
    if analysis_mode in ["Detection Only", "Both (Classification + Detection)"]:
        st.sidebar.subheader("🎚️ Detection Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.05, 1.0, 0.15, 0.05,
            help="Lower values detect more fractures (may include false positives). Recommended: 0.10-0.20"
        )
    else:
        confidence_threshold = 0.5

    # Grad-CAM option
    show_gradcam = False
    show_cam_boxes = False
    cam_threshold = 0.5
    if GRADCAM_AVAILABLE and analysis_mode in ["Classification Only", "Both (Classification + Detection)"]:
        show_gradcam = st.sidebar.checkbox(
            "Show Grad-CAM Heatmap",
            value=False,
            help="Visualize attention regions"
        )
        show_cam_boxes = st.sidebar.checkbox(
            "Show CAM-based Bounding Boxes",
            value=False,
            help="Extract approximate fracture location from classification model using Grad-CAM"
        )
        if show_cam_boxes:
            cam_threshold = st.sidebar.slider(
                "CAM Threshold",
                0.3, 0.9, 0.5, 0.05,
                help="Higher = smaller/more precise boxes, Lower = larger boxes"
            )

    # About
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "**Classification:** Determines if an X-ray shows a fracture (Yes/No)\n\n"
        "**Detection:** Draws bounding boxes around detected fracture areas"
    )

    st.sidebar.warning(
        "⚠️ **Medical Disclaimer:** For educational purposes only. "
        "Consult medical professionals for diagnosis."
    )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📈 Model Info", "❓ Help"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📤 Upload X-ray Image")

            uploaded_file = st.file_uploader(
                "Choose an X-ray image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Supported formats: JPG, PNG, BMP"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

                # Image info
                st.markdown(f"""
                <div class="info-box">
                    <b>📝 Image Details:</b><br>
                    • Size: {image.size[0]} × {image.size[1]} px<br>
                    • Format: {image.format or 'N/A'}<br>
                    • Mode: {image.mode}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Upload an X-ray image to begin analysis")

        with col2:
            st.header("🔍 Analysis Results")

            if uploaded_file is not None:
                # Analysis button
                if st.button("🚀 Analyze X-ray", key="analyze_btn"):
                    # Convert image
                    if image.mode != 'RGB':
                        image_rgb = image.convert('RGB')
                    else:
                        image_rgb = image
                    image_np = np.array(image_rgb)

                    # Store image_np in session state for visualization section
                    st.session_state['image_np'] = image_np

                    # =====================
                    # CLASSIFICATION
                    # =====================
                    if analysis_mode in ["Classification Only", "Both (Classification + Detection)"]:
                        if selected_classifier and available_classifiers:
                            with st.spinner("🔄 Running classification..."):
                                try:
                                    model_path = available_classifiers[selected_classifier]
                                    model, device, model_info = load_classifier_model(
                                        selected_classifier, model_path
                                    )

                                    image_tensor, _ = preprocess_image(image_rgb)
                                    pred_class, confidence, all_probs = predict_classification(
                                        model, image_tensor, device
                                    )

                                    config = Config()
                                    prediction = config.CLASS_NAMES[pred_class]

                                    # Display classification results
                                    st.markdown("### 📊 Classification Result")

                                    if prediction == "Fractured":
                                        st.markdown(f"""
                                        <div class="result-box fractured">
                                            <h2 style="color: #721c24; margin: 0;">⚠️ FRACTURE DETECTED</h2>
                                            <h3 style="color: #721c24; margin-top: 10px;">Confidence: {confidence*100:.2f}%</h3>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="result-box normal">
                                            <h2 style="color: #155724; margin: 0;">✅ NO FRACTURE DETECTED</h2>
                                            <h3 style="color: #155724; margin-top: 10px;">Confidence: {confidence*100:.2f}%</h3>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    # Store for Grad-CAM
                                    st.session_state['classification_done'] = True
                                    st.session_state['model'] = model
                                    st.session_state['device'] = device
                                    st.session_state['image_tensor'] = image_tensor
                                    st.session_state['image_np'] = image_np
                                    st.session_state['pred_class'] = pred_class
                                    st.session_state['selected_model'] = selected_classifier

                                except Exception as e:
                                    st.error(f"❌ Classification error: {str(e)}")

                    # =====================
                    # DETECTION (YOLO)
                    # =====================
                    if analysis_mode in ["Detection Only", "Both (Classification + Detection)"]:
                        st.markdown("### 🎯 Detection Result")

                        with st.spinner("🔄 Running fracture detection..."):
                            try:
                                # Load YOLO model
                                if selected_yolo and selected_yolo in available_yolo:
                                    yolo_model = load_yolo_model(available_yolo[selected_yolo])
                                elif YOLO_AVAILABLE:
                                    # Use pre-trained model
                                    yolo_model = YOLO('yolov8n.pt')
                                    st.info("Using pre-trained YOLOv8. Train on your fracture data for better results.")
                                else:
                                    yolo_model = None

                                if yolo_model:
                                    annotated_image, detections = predict_detection(
                                        yolo_model, image_np, confidence_threshold
                                    )

                                    if detections:
                                        st.markdown(f"""
                                        <div class="detection-box">
                                            <h3 style="color: #856404; margin: 0;">🔍 Found {len(detections)} Fracture Region(s)</h3>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        # Show detection details
                                        for i, det in enumerate(detections, 1):
                                            st.write(f"**Region {i}:** {det['class']} - Confidence: {det['confidence']*100:.1f}%")

                                        # Store annotated image
                                        st.session_state['annotated_image'] = annotated_image
                                        st.session_state['detections'] = detections
                                    else:
                                        st.success("✅ No fracture regions detected in the image")
                                        st.session_state['annotated_image'] = image_np
                                        st.session_state['detections'] = []
                                else:
                                    st.warning("YOLO model not available")

                            except Exception as e:
                                st.error(f"❌ Detection error: {str(e)}")
                                import traceback
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())

            else:
                st.info("📤 Please upload an X-ray image first")

        # =====================
        # VISUALIZATION SECTION
        # =====================
        if uploaded_file is not None:
            # Detection visualization
            if 'annotated_image' in st.session_state and analysis_mode in ["Detection Only", "Both (Classification + Detection)"]:
                st.markdown("---")
                st.header("🖼️ Detection Visualization")
                st.markdown("*Red boxes indicate detected fracture regions*")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Use session state for image_np since it's saved during analysis
                    original_img = st.session_state.get('image_np', None)
                    if original_img is not None:
                        st.image(original_img, caption="Original X-ray", use_container_width=True)

                with viz_col2:
                    st.image(
                        st.session_state['annotated_image'],
                        caption=f"Detected Fractures ({len(st.session_state.get('detections', []))} regions)",
                        use_container_width=True
                    )

            # Grad-CAM visualization
            if (show_gradcam and st.session_state.get('classification_done', False) and
                analysis_mode in ["Classification Only", "Both (Classification + Detection)"]):

                st.markdown("---")
                st.header("🔥 Grad-CAM Visualization")
                st.markdown("*Highlighted regions show where the classifier focused*")

                with st.spinner("🎨 Generating Grad-CAM..."):
                    superimposed, heatmap = generate_gradcam(
                        st.session_state['model'],
                        st.session_state['image_tensor'],
                        st.session_state['image_np'],
                        st.session_state['pred_class'],
                        st.session_state['device'],
                        st.session_state['selected_model']
                    )

                    if superimposed is not None and heatmap is not None:
                        gc_col1, gc_col2, gc_col3 = st.columns(3)

                        with gc_col1:
                            st.image(st.session_state['image_np'], caption="Original", use_container_width=True)
                        with gc_col2:
                            st.image(heatmap, caption="Heatmap", use_container_width=True)
                        with gc_col3:
                            st.image(superimposed, caption="Overlay", use_container_width=True)

            # CAM-based Bounding Box visualization (for Classification)
            if (show_cam_boxes and st.session_state.get('classification_done', False) and
                st.session_state.get('pred_class', 0) == 1 and  # Only show boxes if fracture detected
                analysis_mode in ["Classification Only", "Both (Classification + Detection)"]):

                st.markdown("---")
                st.header("📦 CAM-based Fracture Localization")
                st.markdown("*Orange boxes show approximate fracture regions extracted from classifier attention (Grad-CAM)*")

                with st.spinner("🔍 Extracting fracture regions..."):
                    cam_annotated, cam_boxes = extract_boxes_from_gradcam(
                        st.session_state['model'],
                        st.session_state['image_tensor'],
                        st.session_state['image_np'],
                        st.session_state['pred_class'],
                        st.session_state['device'],
                        st.session_state['selected_model'],
                        threshold=cam_threshold
                    )

                    if cam_annotated is not None and len(cam_boxes) > 0:
                        cam_col1, cam_col2 = st.columns(2)

                        with cam_col1:
                            st.image(st.session_state['image_np'], caption="Original X-ray", use_container_width=True)

                        with cam_col2:
                            st.image(cam_annotated, caption=f"Detected Regions ({len(cam_boxes)} areas)", use_container_width=True)

                        st.info(f"Found **{len(cam_boxes)}** region(s) of interest based on classifier attention. "
                               "These boxes are approximate and based on where the classification model focused.")
                    elif cam_annotated is not None:
                        st.warning("No distinct regions found. Try lowering the CAM threshold.")
                    else:
                        st.error("Could not generate CAM-based boxes.")

    with tab2:
        st.header("📊 Model Information")

        st.markdown("### 🤖 Classification Models")
        if available_classifiers:
            import pandas as pd
            clf_df = pd.DataFrame([
                {'Model': name, 'Status': '✅ Ready', 'Path': path}
                for name, path in available_classifiers.items()
            ])
            st.dataframe(clf_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No classification models found. Train models first.")

        st.markdown("### 🔍 Detection Models (YOLO)")
        if available_yolo:
            import pandas as pd
            yolo_df = pd.DataFrame([
                {'Model': name, 'Status': '✅ Ready', 'Path': path}
                for name, path in available_yolo.items()
            ])
            st.dataframe(yolo_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trained YOLO models found. Train using `run_yolo_training.py`")

        st.markdown("### 📈 Training Commands")
        st.code("""
# Train classification models
python examples/train_single_model.py --model resnet50 --epochs 50
python examples/train_single_model.py --model efficientnet_b3 --epochs 50

# Train YOLO detection model
python run_yolo_training.py

# Train all classifiers
python examples/train_all_models.py
        """, language="bash")

    with tab3:
        st.header("❓ How to Use")

        st.markdown("""
        ### 🎯 Analysis Modes

        **1. Classification Only**
        - Uses ResNet/EfficientNet models
        - Tells you if a fracture exists (Yes/No)
        - Shows confidence percentage

        **2. Detection Only**
        - Uses YOLO model
        - Draws bounding boxes around fracture areas
        - Shows location and confidence of each detection

        **3. Both (Classification + Detection)**
        - Combines both approaches
        - Best for comprehensive analysis
        - Shows overall prediction + precise locations

        ### 🔴 Understanding the Results

        **Bounding Boxes (Red)**
        - Red rectangles indicate detected fracture regions
        - Label shows class name and confidence
        - Multiple boxes = multiple fracture areas

        **Confidence Scores**
        - Higher = more certain about the prediction
        - >80% = High confidence
        - 50-80% = Moderate confidence
        - <50% = Low confidence (review recommended)

        ### 💡 Tips

        1. Use high-quality X-ray images
        2. Ensure good lighting and contrast
        3. Train YOLO on your specific fracture data for best detection results
        4. Use "Both" mode for comprehensive analysis
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p style='font-size: 14px;'><b>⚠️ Medical Disclaimer</b></p>
            <p style='font-size: 12px;'>
                This tool is for educational and research purposes only.
                Always consult qualified healthcare professionals for diagnosis.
            </p>
            <p style='font-size: 12px; margin-top: 20px;'>
                🦴 Bone Fracture Detection System | Classification + Detection
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
