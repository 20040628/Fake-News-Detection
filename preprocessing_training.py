from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter

# Dataset management
CLASS_NAMES = ["Fake", "Real"]
MAPPING_DICT = {
    "Fake": 0,
    "Real": 1
}

# Load fake and real news datasets
fake_news_filepath = "Fake.csv"
real_news_filepath = "True.csv"
fake_df = pd.read_csv(fake_news_filepath)
real_df = pd.read_csv(real_news_filepath)

# Labeling the datasets with 'Fake' and 'Real' labels
real_df["Label"] = "Real"
fake_df["Label"] = "Fake"
df = pd.concat([fake_df, real_df])

# Sample 3000 rows for training and preprocessing
data = df.sample(3000).drop(columns=["title", "subject", "date"])
data.Label = data.Label.map(MAPPING_DICT)

# Preprocess the 'date' column and filter invalid entries
df = df[df.date.map(lambda x: len(str(x))) <= 20]
df.date = pd.to_datetime(df.date, format="mixed")

# ===================== Data Preprocessing =====================
stop_words = set(stopwords.words('english'))

# Function to process text by removing stopwords and punctuation
def text_processing(text):
    words = text.lower().split()  # Convert text to lowercase and split by spaces
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    clean_text = ' '.join(filtered_words)  # Join words back into a string
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation)).strip()  # Remove punctuation
    return clean_text

# Apply preprocessing to the 'text' column
X = data.text.apply(text_processing).tolist()
Y = data.Label.astype('float32').to_numpy()  # Convert labels to numpy array of float32

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, stratify=Y, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, train_size=0.8, stratify=y_train, random_state=42)

# ===================== Tokenizer =====================
bert_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_name)  # Load the tokenizer for BERT

# ===================== Custom Dataset Class =====================
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Tokenize the texts and prepare them for model input
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels  # Labels for each example

    def __len__(self):
        return len(self.labels)  # Return the size of the dataset

    def __getitem__(self, idx):
        # Convert the encodings to tensors and return the item
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)  # Labels as float for BCE
        return item

# Create datasets for training, validation, and testing
train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
valid_dataset = FakeNewsDataset(X_valid, y_valid, tokenizer)
test_dataset  = FakeNewsDataset(X_test,  y_test,  tokenizer)

# Confirm whether the category balance has been maintained
print("===== Dataset Distribution =====")
print(f"Training Set: {len(train_dataset)} samples")
print("  Class distribution:", Counter(y_train))
print(f"Validation Set: {len(valid_dataset)} samples")
print("  Class distribution:", Counter(y_valid))
print(f"Test Set: {len(test_dataset)} samples")
print("  Class distribution:", Counter(y_test))

# ===================== Load PyTorch Model =====================
model = AutoModelForSequenceClassification.from_pretrained(
    bert_name, num_labels=1  # Using 1 label for binary classification with BCE loss
)

# ===================== Define Evaluation Metrics =====================
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()  # Apply sigmoid to logits for probabilities
    preds = (probs > 0.8).astype(int)  # Convert probabilities to binary predictions (0 or 1)

    # Calculate accuracy, precision, and recall
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
         "f1": f1_score(labels, preds)
    }

# ===================== Training Arguments =====================
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    evaluation_strategy="epoch",  # Evaluate the model every epoch
    save_strategy="epoch",  # Save model every epoch
    num_train_epochs=50,  # Number of epochs to train the model
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    save_total_limit=1,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
    report_to="none",  # Disable reporting to external systems
    learning_rate=2e-5,  # Learning rate for optimizer
    weight_decay=0.01,  # Weight decay for regularization
)

# ===================== Trainer =====================
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # The training arguments
    train_dataset=train_dataset,  # The training dataset
    eval_dataset=valid_dataset,  # The validation dataset
    compute_metrics=compute_metrics,  # The function to compute metrics
)

# ===================== Start Training =====================
trainer.train()

# ===================== Model Evaluation =====================
eval_results = trainer.evaluate(test_dataset)  # Evaluate the model on the test dataset
print("Test results:", eval_results)  # Print the test evaluation results
