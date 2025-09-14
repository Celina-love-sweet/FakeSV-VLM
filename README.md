

Some parts in the README may not be very detailed. I’ll create a simpler and easier-to-understand version after I finish my current work. About 1 month.

## 📦 1. Dataset Preparation

Due to copyright reasons, we are unable to provide the original datasets. You can download them from the following links:

### 1.1. FakeSV

- **Description**: A multimodal benchmark for fake news detection on short video platforms.
- **Access**: [ICTMCG/FakeSV](https://github.com/ICTMCG/FakeSV)  
  📄 *FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms*, AAAI 2023.

### 1.2. FakeTT

- **Description**: A dataset for fake news detection from the perspective of creative manipulation.
- **Access**: [ICTMCG/FakingRecipe](https://github.com/ICTMCG/FakingRecipe)  
  📄 *FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process*, ACM MM 2024.

### 1.3. JSONL data processing
After downloading the datasets, please organize them according to the format described in the paper and required by the ms-swift framework. Please refer to the official manual for specific formatting and placement instructions. 

An example is as follows: we assume that the storage path of one video is `FakeSV_VLM/FakeTT/video`. Based on the description and event in the original dataset, we can obtain the corresponding JSONL file:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given the news video description, news event and key frames, you need to predict the authenticity of the news video. If the video is more likely to be fake news, return fake; otherwise, return real. Please avoid providing ambiguous evaluations such as undetermined. News video description: Jimmy Fallon ripped off Donald Trump's toupee, News event: Trump toupee video, News video key frames: <video>, Your prediction (no need to give your analysis, return real or fake only):"
    },
    {
      "role": "assistant",
      "content": "fake"
    }
  ],
  "videos": [
    "FakeSV_VLM/FakeTT/video/6687587479509273861.mp4"
  ]
}
```

---

## ⚙️ 2. Environment Setup

We recommend using a Python virtual environment to avoid conflicts.

### 2.1. Create and activate a virtual environment (e.g., with conda):

```bash
conda create -n fakesv-vlm python=3.11.8 -y
conda activate fakesv-vlm
```

### 2.2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure `torch`, `transformers`, and `other dependencies` are installed properly as specified.

---

## 🛠️ 3. Other Preparation

### 3.1. Install `ms-swift` (v3.2.0)

Please install the `ms-swift` library with the specific version:

```bash
pip install ms-swift==3.2.0
```

After installation, replace the following file with our customized version:

```text
[ms-swift-root]/llm/template/internvl.py → replace with utils/internvl.py
```

### 3.2. Download InternVL2.5

Download the **InternVL2.5** model from the [official repository](https://github.com/OpenGVLab/InternVL). Then, replace the model definition file:

```text
[InternVL2.5-root]/modeling_internlm2.py → replace with our customized modeling_internlm2.py
```

---

## ✅ 4. Running

After environment and data are ready, you can start training or inference as follows:

### 4.1. Train

```bash
bash train.sh
```

### 4.2. Inference

```bash
bash inference.sh
```

> 📌 Please make sure all dataset paths and model checkpoints are correctly configured in the script and config files.

---

## 🙏 5. Acknowledgements

- We sincerely thank the developers of the [**ms-swift**](https://github.com/modelscope/ms-swift) framework, which provides a powerful and modular infrastructure for large-scale multimodal experiments.
- We also gratefully acknowledge the [**InternVL**](https://github.com/OpenGVLab/InternVL) team for their release of the **InternVL2.5** model, which serves as the backbone of our visual-language encoding.

---

## 📬 6. Contact

If you have any questions or encounter any issues, feel free to open an issue or contact me directly.
