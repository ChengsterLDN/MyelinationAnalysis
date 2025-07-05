from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I am getting laid off tonight!")

print(res)