import random
import json

import torch
import torch.nn as nn

from utils import tokenize, embedding
from model import ChatBotModel

with open('data.json', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

information = torch.load("information_stored.pth")

state_dict = information["model_state_dict"]
input_size = information["input_size"]
output_size = information["output_size"]
all_words = information["all_words"]
tags = information["tags"]

model = ChatBotModel(input_size, output_size)
model.load_state_dict(state_dict)
model.eval()

print("شروع چت!")
print("\nبرای تمام کردن چت کلمه پایان را تایپ کنید")
print("\n")
while True:
    user = input("You: ")
    if user == 'پایان':
        break

    tokenized = tokenize(user)
    embed = embedding(tokenized, all_words)
    embed = embed.reshape(1, len(embed))
    X = torch.from_numpy(embed).to(dtype=torch.float)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    pred_cls = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob > 0.5:
        for instance in data["data"]:
            if instance["tag"] == pred_cls:
                sentence = random.choice(instance["responses"])
                print(f"Ai: {sentence}")
    else:
        print(f"Ai: I don't understand.")

print("\n")
print("چت به اتمام رسید!")