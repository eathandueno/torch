from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader
import torch
from datasets import Dataset, load_dataset
import pandas as pd

# Define the directory where the model is saved
output_dir = "./saved_model"

# Load the saved model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

# Function to dynamically map column names
def map_columns(dataset, input_col, target_col, context_col="qtype"):
    """
    Maps dataset columns to 'question', 'answer', and ensures 'qtype' exists.
    Fills missing or blank 'qtype' values with 'general'. Converts all answers to strings.
    """
    # Rename input and target columns if necessary
    if input_col != "question":
        dataset = dataset.rename_column(input_col, "question")
    if target_col != "answer":
        dataset = dataset.rename_column(target_col, "answer")

    # Ensure all answers are strings
    dataset = dataset.map(lambda x: {"answer": [str(val) if val is not None else "" for val in x["answer"]]}, batched=True)

    # Handle the qtype column
    if context_col in dataset.column_names:
        # Fill blank or missing values in 'qtype' with 'general'
        dataset = dataset.map(lambda x: {"qtype": [str(val) if val else "general" for val in x[context_col]]}, batched=True)
    else:
        # Add a new 'qtype' column with default value 'general'
        dataset = dataset.map(lambda x: {"qtype": ["general"] * len(x["question"])}, batched=True)

    return dataset









# Define a function to load data
def load_data(data_source, data_type, input_col=None, target_col=None, context_col="qtype"):
    if data_type == "csv":
        data = pd.read_csv(data_source)
        dataset = Dataset.from_pandas(data)
    elif data_type == "json":
        dataset = Dataset.from_json(data_source)
    elif data_type == "txt":
        # Assumes tab-delimited text files
        data = pd.read_csv(data_source, delimiter="\t", header=None, names=["question", "answer"])
        dataset = Dataset.from_pandas(data)
    elif data_type == "huggingface":
        dataset = load_dataset(data_source, split="train")
    else:
        raise ValueError("Unsupported data type. Supported types: csv, json, txt, huggingface.")
    
    # Map and fill columns
    dataset = map_columns(dataset, input_col, target_col, context_col)
    return dataset


# Define a function to tokenize the dataset
def prepare_dataset(dataset):
    def tokenize_function(batch):
    # Debugging: Print batch contents
        print("Batch questions:", batch["question"][:5])
        print("Batch answers:", batch["answer"][:5])
        if "qtype" in batch:
            print("Batch qtypes:", batch["qtype"][:5])

        # Validate input types
        if not all(isinstance(q, str) for q in batch["question"]):
            raise ValueError("All questions must be strings")
        if not all(isinstance(a, str) for a in batch["answer"]):
            raise ValueError("All answers must be strings")
        if "qtype" in batch and not all(isinstance(qt, str) for qt in batch["qtype"]):
            raise ValueError("All qtypes must be strings")

        # Process input text
        if "qtype" in batch and batch["qtype"]:
            input_text = [f"{qtype} | {question}" for qtype, question in zip(batch["qtype"], batch["question"])]
        else:
            input_text = batch["question"]

        target_text = batch["answer"]

        # Tokenize input and output text with padding
        inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        targets = tokenizer(target_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }



    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

# Prompt user for input data source details
data_source = input("Enter the path to your training data file or Hugging Face dataset name: ")
data_type = input("Enter the data type (csv, json, txt, huggingface): ")
input_col = input("Enter the input column name (e.g., 'act', 'question'): ")
target_col = input("Enter the target column name (e.g., 'prompt', 'answer'): ")
context_col = input("Enter the context column name (e.g., 'category') or leave blank if not applicable: ")

# Load and process the dataset
try:
    raw_dataset = load_data(data_source, data_type, input_col, target_col, context_col)
    print("Data loaded and columns mapped successfully!")
    processed_dataset = prepare_dataset(raw_dataset)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Convert dataset to PyTorch DataLoader
train_loader = DataLoader(processed_dataset, batch_size=8, shuffle=True)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Continue training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save the updated model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
