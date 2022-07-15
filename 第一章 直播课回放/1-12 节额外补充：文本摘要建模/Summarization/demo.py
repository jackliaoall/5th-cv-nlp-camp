from transformers import pipeline

summarizer = pipeline("summarization", model='./mt5-small-finetuned-amazon-en-es/checkpoint-4000')

print(summarizer('垃圾东西，一用就坏了，根本没法使用退钱'))