import os
import random
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

BLUR_PROB = 0.35       # Probability of applying Gaussian blur per image
GAMMA_PROB = 0.5       # Probability of applying random gamma correction per image
BRIGHTNESS_PROB = 0.4  # Probability of applying brightness/contrast jitter per image

def _gaussian_kernel(kernel_size: int, sigma: float, device) -> torch.Tensor:
    """Build a 2D Gaussian blur kernel as a (1, 1, k, k) tensor."""
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    return kernel.view(1, 1, kernel_size, kernel_size)

class MotionBlurTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass that applies on-GPU augmentations each batch:
      - Random Gaussian blur  (simulates camera/head vibration)
      - Random gamma correction (simulates non-linear lighting shifts)
      - Random brightness/contrast jitter (simulates exposure differences)
    All augmentations are applied independently per image in the batch.
    """
    BLUR_PROB = 0.35
    GAMMA_PROB = 0.5
    BRIGHTNESS_PROB = 0.4

    def preprocess_batch(self, batch):
        # Run standard preprocessing first (normalize, move to device, etc.)
        batch = super().preprocess_batch(batch)

        imgs = batch.get("img", None)
        if imgs is None or not isinstance(imgs, torch.Tensor):
            return batch

        # imgs is float32 in [0, 1] after standard preprocessing
        device = imgs.device
        out = imgs.clone()

        for i in range(out.shape[0]):
            img = out[i]  # (C, H, W), values in [0, 1]

            # --- Gaussian Blur ---
            if random.random() < self.BLUR_PROB:
                k = random.choice([3, 5, 7])
                sigma = random.uniform(0.8, 2.5)
                kernel = _gaussian_kernel(k, sigma, device)  # (1,1,k,k)
                C = img.shape[0]
                kernel_c = kernel.expand(C, 1, k, k)
                img = F.conv2d(img.unsqueeze(0), kernel_c, padding=k // 2, groups=C).squeeze(0)

            # --- Gamma Correction ---
            # gamma < 1 → brighter (over-exposed); gamma > 1 → darker (under-exposed)
            if random.random() < self.GAMMA_PROB:
                gamma = random.uniform(0.4, 2.2)
                img = img.clamp(1e-6, 1.0).pow(gamma)

            # --- Brightness / Contrast Jitter ---
            if random.random() < self.BRIGHTNESS_PROB:
                # Brightness shift: additive offset in [-0.3, +0.3]
                brightness_delta = random.uniform(-0.3, 0.3)
                # Contrast scale: multiplicative factor in [0.6, 1.6]
                contrast_factor = random.uniform(0.6, 1.6)
                mean = img.mean()
                img = (img - mean) * contrast_factor + mean + brightness_delta
                img = img.clamp(0.0, 1.0)

            out[i] = img

        batch["img"] = out
        return batch

def main():
    # Verify CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    if device == 'cpu':
        # Limit CPU threads to reduce memory footprint and context switching overhead
        torch.set_num_threads(2)
        print("Configured PyTorch to use 2 CPU threads for memory saving.")
    
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Path to the dataset configuration yaml
    dataset_yaml = os.path.join(ws_root, "datasets/dataset_merged/dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset config not found at {dataset_yaml}")
        print("Make sure you run merge_datasets.py first!")
        return

    # Load standard pre-trained yolov8n model
    # print("Loading pre-trained YOLOv8n weights...")
        # Load standard pre-trained yolo26n model
    print("Loading pre-trained YOLO26n weights...")
    # model_path = os.path.join(script_dir, "yolov8n.pt")
    model = YOLO("yolo26n.pt")  # Or yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
    # model = YOLO(model_path)
    print(f"Motion blur enabled via MotionBlurTrainer (p={MotionBlurTrainer.BLUR_PROB}, Gaussian kernel 3-7px).")



    # Train the model
    print("Starting YOLO training on merged dataset (single-class)...")
    results = model.train(
        trainer=MotionBlurTrainer,
        data=dataset_yaml,
        epochs=500,
        imgsz=640,
        batch=16,           
        workers=0,          
        device=device,
        project=os.path.join(ws_root, "runs/detect"),
        name="tiago_single_class_yolo26",
        save=True,
        
        # --- Custom Augmentation Tweaks ---
        degrees=30.0,       # Add rotational robustness
        mosaic=0.5,         # Reduce mosaic frequency (too disruptive for clean scenes)
        erasing=0.15,       # Reduce erasing so small object features aren't completely lost
        # Motion blur: applied via on_train_batch_start callback above (no albumentations needed)
        #################################
        scale=0.5,          # Random scaling +/- 50%
        translate=0.15,     # Random translation +/- 15%
        perspective=0.0005, # Random 3D perspective distortion
        hsv_h=0.02,         # Random Hue +/- 2%
        hsv_s=0.9,          # Random Saturation +/- 90%
        hsv_v=0.6,          # Random Value/Brightness +/- 60%
        auto_augment="augmix",  # AugMix: corruption/lighting robust policy (replaces randaugment)
    )

    print("Training complete!")
    print(f"Best model weights saved to: {os.path.join(ws_root, 'runs/detect/tiago_single_class_yolo26/weights/best.pt')}")

if __name__ == '__main__':
    main()
