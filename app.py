import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Change this to the folder path where your saved model is stored in the repo OR Hugging Face
MODEL_PATH = "./bert_toxic_model"  # if you upload model folder into repo

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_toxicity(text: str):
    if text is None or len(text.strip()) == 0:
        return "Please enter a comment.", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()

    # label mapping: 0 = Not Toxic, 1 = Toxic (as used in our project)
    toxic_prob = float(probs[1].item())
    label = "Toxic" if toxic_prob >= 0.5 else "Not Toxic"
    return label, toxic_prob

demo = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(lines=4, placeholder="Paste a comment here..."),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Toxic probability")
    ],
    title="Toxic Comment Detection (BERT)",
    description="Enter a comment to classify it as Toxic or Not Toxic using a fine-tuned BERT model."
)

if __name__ == "__main__":
    demo.launch()
