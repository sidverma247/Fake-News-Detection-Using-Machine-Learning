# 🔍 Fake News Detection Using Machine Learning & Transformers

A production-ready NLP pipeline that detects fake news using state-of-the-art transformer models (BERT, RoBERTa) with a FastAPI backend and interactive demo.

---

## 📁 Project Structure

```
fake-news-detection/
├── src/
│   ├── dataset.py          # Dataset loading & preprocessing
│   ├── model.py            # Transformer model wrapper
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation & metrics
│   └── predict.py          # Inference utilities
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py          # Pydantic models
├── notebooks/
│   └── EDA_and_Training.ipynb  # Full exploratory notebook
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── data/                   # Place datasets here
├── models/                 # Saved model checkpoints
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

### 2. Download Dataset
Place the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) or [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) into `data/`.

### 3. Train
```bash
python src/train.py --model roberta-base --epochs 4 --batch_size 16
```

### 4. Evaluate
```bash
python src/evaluate.py --model_path models/best_model/
```

### 5. Run API
```bash
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### 6. Docker
```bash
docker-compose up --build
```

---

## 🧠 Models Supported
| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| BERT-base     | 92.3%    | 0.921    |
| RoBERTa-base  | **94.1%**| **0.939**|
| DistilBERT    | 91.7%    | 0.915    |

---

## 📊 Datasets
- **LIAR**: 12.8K manually labelled statements from PolitiFact
- **FakeNewsNet**: News articles with social context
- **ISOT**: 44K real + fake news articles

---

## 🔬 Approach
1. **Preprocessing**: Tokenization, cleaning, truncation to 512 tokens
2. **Model**: Pre-trained transformer + classification head
3. **Training**: AdamW optimizer, linear warmup scheduler, cross-entropy loss
4. **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

---

## 📄 License
MIT License
