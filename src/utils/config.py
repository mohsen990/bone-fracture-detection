import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'validation')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Image parameters
    IMG_SIZE = 224  # 224 for ResNet/EfficientNet, can be 512 for detection
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Classification
    NUM_CLASSES = 2  # Binary: fracture vs normal
    CLASS_NAMES = ['Normal', 'Fractured']
    
    # Model parameters
    MODELS = {
        'resnet50': {'img_size': 224, 'weights': 'imagenet'},
        'resnet101': {'img_size': 224, 'weights': 'imagenet'},
        'efficientnet_b0': {'img_size': 224, 'weights': 'imagenet'},
        'efficientnet_b3': {'img_size': 300, 'weights': 'imagenet'},
        'efficientnet_b7': {'img_size': 600, 'weights': 'imagenet'},
    }
    
    # Detection parameters (for YOLO/RetinaNet)
    DETECTION_IMG_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45