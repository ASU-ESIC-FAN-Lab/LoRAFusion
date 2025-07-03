
# LoRAFusion: A Crossbar-aware Multi-task Adaption Framework via Efficient Fusion of Pretrained LoRA Modules

This repository provides the implementation for **LoRAFusion**, a parameter-efficient multi-task on-device learning framework for ReRAM crossbar accelerators via fusion of pre-trained LoRA modules.

LoRAFusion achieves significant reduction in trainable parameters during on-device fine-tuning, while maintaining near full fine-tuning performance. For full details, please refer to our [GLSVLSI 2025 paper](https://doi.org/10.1145/3716368.3735213).

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@inproceedings{10.1145/3716368.3735213,
  author    = {Guo, Jingkai and Ali, Asmer and Yang, Li and Fan, Deliang},
  title     = {LoRAFusion: A Crossbar-aware Multi-task Adaption Framework via Efficient Fusion of Pretrained LoRA Modules},
  booktitle = {Proceedings of the Great Lakes Symposium on VLSI 2025},
  series    = {GLSVLSI '25'},
  pages     = {777--783},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3716368.3735213},
  url       = {https://doi.org/10.1145/3716368.3735213}
}
```

---

## 🚀 Getting Started

### Environment

We recommend creating a conda environment:

```bash
conda create -n lorafusion python=3.11
conda activate lorafusion
pip install -r requirements.txt
```

---

### Training

You can train a model using:

```bash
python train.py [options]
```

#### Key arguments

| Argument            | Type    | Default | Description |
|---------------------|---------|---------|-------------|
| `--epoch`            | int     | 20      | Maximum number of training steps |
| `--batch_size`       | int     | 10      | Batch size for training |
| `--lr`               | float   | 0.02    | Learning rate for optimizer |
| `--lora_num`         | str     | 20      | Number of LoRA modules to use |
| `--log`              | flag    | False   | Log experiment to Wandb |
| `--load_in_4bit`     | flag    | False   | Load model in 4-bit precision |

---

### Example

```bash
python train.py --epoch 30 --batch_size 16 --lr 0.01 --lora_num 10 --log --load_in_4bit
```

---

## 📊 Results

LoRAFusion uses only **3%** of the trainable parameters compared to LoRA (148K vs. 4700K), with just a **0.19% accuracy drop**.

---

## 🔗 Links

- 📂 **Code:** [https://github.com/ASU-ESIC-FAN-Lab/LoRAFusion](https://github.com/ASU-ESIC-FAN-Lab/LoRAFusion)
- 📄 **Paper:** [https://doi.org/10.1145/3716368.3735213](https://doi.org/10.1145/3716368.3735213)

---

## 💡 Acknowledgments

This project includes code from: [lorahub](https://github.com/sail-sg/lorahub)

This work was presented at **GLSVLSI 2025** and supported by ASU ESIC FAN Lab.


