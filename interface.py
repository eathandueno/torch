import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the directory where the model is saved
output_dir = "./saved_model"

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(output_dir)
model = T5ForConditionalGeneration.from_pretrained(output_dir)

# Prepare your input data
input_text = "What are cold like symptoms?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Print input tensor
print("Input IDs:", input_ids)

# Generate output
model.eval()
with torch.no_grad():
    # Forward pass through the model to get hidden states
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
    hidden_states = outputs.encoder_last_hidden_state


    # Generate output tokens
    output_ids = model.generate(input_ids, max_new_tokens=200)

# Decode the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print output tensor and text
print("Output IDs:", output_ids)
print("Output Text:", output_text)