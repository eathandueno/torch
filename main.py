from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader
import torch

# Load a Hugging Face dataset
# Replace with your desired dataset name and split
dataset_name = "keivalya/MedQuad-MedicalQnADataset"
dataset = load_dataset(dataset_name, split="train")

# Split the dataset into 80% train and 20% validation
train_test_split = dataset.train_test_split(test_size=0.2)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

# Initialize tokenizer for T5 model
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define a tokenization function
def tokenize_function(batch):
    input_text = batch["Question"]  # Replace with your dataset's input column name
    target_text = batch["Answer"]   # Replace with your dataset's target column name

    # Tokenize input and target text
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    targets = tokenizer(target_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze()
    }

# Tokenize the training and validation datasets
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Convert datasets to PyTorch tensors and DataLoader
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# Load T5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # Move batch data to GPU if available
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss for monitoring
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save the model and tokenizer
output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
