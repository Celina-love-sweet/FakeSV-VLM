
## 📦 Dataset Preparation

Due to copyright reasons, we are unable to provide the original datasets.  
You can download them from the following links:

### FakeSV

- **Description**: A multimodal benchmark for fake news detection on short video platforms.
- **Access**: [ICTMCG/FakeSV](https://github.com/ICTMCG/FakeSV)  
  📄 *FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms*, AAAI 2023.

### FakeTT

- **Description**: A dataset for fake news detection from the perspective of creative manipulation.
- **Access**: [ICTMCG/FakingRecipe](https://github.com/ICTMCG/FakingRecipe)  
  📄 *FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process*, ACM MM 2024.

After downloading the datasets, please organize them according to the format described in the paper and required by the ms-swift framework. Please refer to the official manual for specific formatting and placement instructions.

---

## ⚙️ Environment Setup

We recommend using a Python virtual environment to avoid conflicts.

### 1. Create and activate a virtual environment (e.g., with conda):

```bash
conda create -n fakesv-vlm python=3.11.8 -y
conda activate fakesv-vlm
```

### 2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure `torch`, `transformers`, and other dependencies are installed properly as specified.

---

## 🛠️ Other Preparation

### 🔧 Install `ms-swift` (v3.2.0)

Please install the `ms-swift` library with the specific version:

```bash
pip install ms-swift==3.2.0
```

After installation, replace the following file with our customized version:

```text
[ms-swift-root]/llm/template/internvl.py → replace with utils/internvl.py
```

### 📥 Download InternVL2.5

Download the **InternVL2.5** model from the [official repository](https://github.com/OpenGVLab/InternVL).  
Then, replace the model definition file:

```text
[InternVL2.5-root]/modeling_internlm2.py → replace with our customized modeling_internlm2.py
```

---

## ✅ Running

After environment and data are ready, you can start training or inference as follows:

### 🔧 Train

```bash
bash train.sh
```

### 🔍 Inference

```bash
bash inference.sh
```

> 📌 Please make sure all dataset paths and model checkpoints are correctly configured in the script and config files.

---

## 🙏 Acknowledgements

- We sincerely thank the developers of the [**ms-swift**](https://github.com/modelscope/ms-swift) framework, which provides a powerful and modular infrastructure for large-scale multimodal experiments.
- We also gratefully acknowledge the [**InternVL**](https://github.com/OpenGVLab/InternVL) team for their release of the **InternVL2.5** model, which serves as the backbone of our visual-language encoding.

---

## 📬 Contact

If you have any questions or encounter any issues, feel free to open an issue or contact me directly.
