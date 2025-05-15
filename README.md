

# Fake-News-Detection-BERT

This project trains a BERT-based classifier to detect fake news using HuggingFace Transformers and PyTorch.

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

## How to Run

Put `Fake.csv` and `True.csv` in the same directory as `preprocessing_training.py`.

If you have a local BERT model, modify this line in the code:

```
bert_name = "./bert-base-uncased" # path to your local model 
```

Run the script:

```python main.py
python preprocessing_training.py
```

After training, the best model is saved in the `./results` directory and evaluation metrics will be printed.

## Output

- Training progress and validation metrics printed during training
- Final test accuracy, precision, recall, and F1 score