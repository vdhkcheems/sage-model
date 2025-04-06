from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from typing import List

app = FastAPI()

model_path = "vdhkcheems/SAGE"  
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class QuestionData(BaseModel):
    questionNumber: str
    question: str
    referenceAnswer: str
    answer: str

@app.post("/predict/")
async def predict(data: List[QuestionData]):
    results = []

    for item in data:
        text = f"{item.question} [SEP] {item.referenceAnswer} [SEP] {item.answer}"
        
        encoding = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        results.append({"questionNumber": item.questionNumber, "label": prediction})

    return results
