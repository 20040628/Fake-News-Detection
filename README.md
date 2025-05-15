

# Fake-News-Detection-BERT

This project trains a BERT-based classifier to detect fake news using HuggingFace Transformers and PyTorch.

## Files

- `Fake.csv` – Dataset of fake news articles  
- `True.csv` – Dataset of real news articles  
- `main.py` – Main training and evaluation script  
- `data_analysis.ipynb` – Notebook for dataset exploration and visualization  
- `bert-base-uncased/` – (Optional) Local BERT model directory (or use HuggingFace download)

## Requirements

Make sure you have Python 3.7+ and install the following packages:

```bash
pip install transformers datasets scikit-learn pandas numpy torch nltk
```

You also need to download NLTK stopwords:

```
import nltk
nltk.download('stopwords')
```

## Data Analysis

Run `data_analysis.ipynb` to explore and visualize the dataset. It performs the following:

- Loads and merges `Fake.csv` and `True.csv`
- Assigns labels (`0` = Fake, `1` = Real)
- Samples 3000 articles for faster experimentation
- Visualizes class distribution:
  - Fake vs Real label balance (relatively balanced)
  - Distribution of `subject` categories by label (unbalanced, not used as feature)
- Observes that the `date` field contains some noisy or invalid strings (e.g., URLs), so `date` is excluded as a feature

> The analysis helps confirm that only the `text` field is suitable as a classification input.

## Fine-tuning BERT

1. Make sure `Fake.csv` and `True.csv` are in the same folder as `main.py`.
2. (Optional) If using a local model, ensure `bert-base-uncased/` is in the same directory and modify this line in the code:

```
bert_name = "./bert-base-uncased" # path to your local model 
```

​	Otherwise, the model will be downloaded automatically from HuggingFace.

3. Run training:

```python main.py
python main.py
```

After training, the best model is saved in the `./results` directory and evaluation metrics will be printed.

## Output

- Training progress and validation metrics printed during training
- Final test accuracy, precision, recall, and F1 score