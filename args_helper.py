import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Maximum number of training steps (default: 20)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate for optimizer (default: 0.02)')
    parser.add_argument('--lora_num', type=str, default=20,
                        help='number of LoRA module to use (default: 20)')
    parser.add_argument('--log', action='store_true',
                        help='Whether to log the experiment (default: False)')
    parser.add_argument('--es', action='store_true',
                        help='Enable early stopping (default: False)')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit precision (default: False)')

    return parser.parse_args()
