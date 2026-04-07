
        
import argparse
import wandb
from ultralytics import YOLO, settings
import os

def calculate_f1_callback(trainer):
    metrics = trainer.metrics

    p = metrics.get("metrics/precision(B)", 0.0)
    r = metrics.get("metrics/recall(B)", 0.0)

    eps = 1e-16
    f1 = 2 * p * r / (p + r + eps)

    if wandb.run:
        wandb.log({"metrics/f1(B)": f1}, commit=False)
        
def main():
    parser = argparse.ArgumentParser(description="YOLO Training with WandB and F1 Logging")
    
    # Dataset and Model
    parser.add_argument("--data", default="dataset.yaml", help="Path to dataset YAML file")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (yolov8n.pt, yolov11n.pt, etc.)")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    # Directories and WandB
    parser.add_argument("--project", default="runs/train", help="Local directory for results")
    parser.add_argument("--name", default="fine_tune", help="Experiment name")
    parser.add_argument("--wandb-project", default="YOLO_Project", help="WandB project name")
    
    # Training Controls
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--freeze", type=int, default=None, help="Number of layers to freeze")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze first 10 layers (backbone)")
    
    args = parser.parse_args()

    settings.update({"wandb": True})

    if os.getenv("WANDB_API_KEY") or wandb.api.api_key:
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            config=vars(args),
            job_type="training",
            reinit=True
        )
    else:
        print("WARNING: WandB API Key not found. Please run 'wandb login' or set WANDB_API_KEY environment variable.")

    model = YOLO(args.model)

    # This function triggers every time a validation epoch finishes to compute f1 score
    model.add_callback("on_fit_epoch_end", calculate_f1_callback)

    # Handle Freeze layers logic
    freeze_list = None
    if args.freeze is not None:
        freeze_list = list(range(args.freeze))
    elif args.freeze_backbone:
        freeze_list = list(range(10))

    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            resume=args.resume,
            freeze=freeze_list,
            plots=True,   
            val=True,     
            exist_ok=True 
        )
        print(f"Training successfully completed. Results: {args.project}/{args.name}")
    
    except Exception as e:
        print(f"Error during training: {e}")
    
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()