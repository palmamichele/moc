# from transformers import AutoTokenizer
# from datasets import load_dataset

# dataset = load_dataset("HuggingFaceFW/fineweb")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokens = tokenizer.tokenize("The cat sat on the mat.")
# print(tokens)

# #what the model really sees 
# tokenizer.encode("The cat sat on the mat.")

#launch python data/shakespeare_char/prepare.py from nanoGPT


from model import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)


#train the model 
#python train.py config/train_shakespeare_char.py