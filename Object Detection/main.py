import os
import time
from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from trainer import build_trainer
from utils import not_change_test_dataset
from pprint import pprint
from datasets import load_dataset, DatasetDict

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """
    Main function to execute model training and evaluation.
    """

    # 下载并加载数据集到本地目录
    raw_datasets = load_dataset("cppe-5")

    # 打印数据集的键，检查是否包含 train, validation, test
    print("Loaded dataset splits:", raw_datasets.keys())

    # 确保 test 数据集不变
    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # 如果 "validation" 缺失，从 "train" 划分 20% 作为 "validation"
    if "validation" not in raw_datasets and "train" in raw_datasets:
        split_data = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
        raw_datasets = DatasetDict({
            "train": split_data["train"],
            "validation": split_data["test"],  # 20% 作为 validation
            "test": raw_datasets["test"]  # 保持 test 不变
        })
        print("After splitting, dataset splits:", raw_datasets.keys())

    # 初始化图像处理器
    processor = initialize_processor()

    # 对数据集进行预处理
    datasets = add_preprocessing(raw_datasets, processor)

    # 构建目标检测模型
    model = initialize_model()

    # 构建 Trainer 并进行训练
    trainer = build_trainer(
        model=model,
        image_processor=processor,
        datasets=datasets,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    # 评估模型在测试集上的表现
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    pprint(test_metrics)


if __name__ == "__main__":
    main()
