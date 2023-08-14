import argparse

def argparse_config():
    parser = argparse.ArgumentParser(description="DDP Trainer")

    parser.add_argument("--epochs", help="훈련 횟수", type=int, default=80)
    parser.add_argument("--fp16", help="float16 데이터 형식으로 학습", action="store_true")
    parser.add_argument("--batch_size", help="한번에 학습시킬 데이터 개수", type=int, default=32)
    parser.add_argument("--wandb", help="Wandb 사용 여부", action="store_true")

    args = parser.parse_args()

    return args