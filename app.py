import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "RajMohanReddy/bert-toxic-jigsaw"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

def predict_toxicity(text: str):
    # Basic input validation (error handling requirement)
    if text is None or not isinstance(text, str) or len(text.strip()) == 0:
        return "Please enter a non-empty comment.", 0.0

    try:
        inputs = tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()

        # label mapping: 0 = Not Toxic, 1 = Toxic
        toxic_prob = float(probs[1].item())
        label = "Toxic" if toxic_prob >= 0.5 else "Not Toxic"
        return label, toxic_prob

    except Exception as e:
        # Prevent demo from crashing; return a readable error
        return f"Error: {type(e).__name__}", 0.0

demo = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(lines=4, placeholder="Paste a comment here..."),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Toxic probability")
    ],
    title="Toxic Comment Detection (BERT)",
    description="Enter a comment to classify it as Toxic or Not Toxic using a fine-tuned BERT model.",
    examples=[
        ["You are amazing, thanks for helping!"],
        ["Shut up, you idiot."],
        ["I will hurt you."],
        ["This is a normal discussion about the topic."]
    ]
)

if __name__ == "__main__":
    demo.launch()
