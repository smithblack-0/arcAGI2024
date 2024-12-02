from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SplitInverse(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x_t, u_t, y):
		ctx.save_for_backward(x_t, u_t, y)
        high_inverse = (y-x_t*u_t)/()


# Specify the model name; for example, 'RWKV/rwkv-4-169m-pile'
model_name = 'RWKV/rwkv-4-169m-pile'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained RWKV model
model = AutoModelForCausalLM.from_pretrained(model_name)
base_model = model.base_model
block = base_model.blocks[0]
attn = block.attention
for name, tensor in attn.named_parameters():
    print(name, tensor - 1)


# Define your input prompt
prompt = "In a shocking finding, scientists discovered a herd of dragons"

# Encode the input prompt
inputs = tokenizer(prompt, return_tensors='pt')

# Generate text continuation
outputs = model.generate(inputs['input_ids'], max_new_tokens=50)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

