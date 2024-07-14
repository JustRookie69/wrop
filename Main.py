import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import whisper as wp
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
import numpy as np

model = wp.load_model("small")

audio_path = r"C:\projects\Wrop\test3.mp3"
result = model.transcribe(audio_path)
extracted_text = result.get('text', 'Error during STT conversion')

identified_language = result.get("language", None)
print(f"Language identified: {identified_language}")

if identified_language.lower() != "en":
    translated_result = model.transcribe(audio_path, task="translate")
    transcribed_text = translated_result.get('text', 'Error during translation')
else:
    transcribed_text = extracted_text

with open("englishTranslation.txt", "w", encoding="utf-8") as f:
    f.write(transcribed_text)

conversational_data = {
    "title": ["Greeting 1", "Greeting 2", "Small Talk 1"],
    "text": [
        "Hi! I'm doing well, thank you. How can I assist you today?",
        "Hello! I'm great, thanks for asking. What do you need help with?",
        "I'm fine, thank you. How can I help you today?"
    ],
    "embeddings": [
        np.random.rand(768).tolist(),  # Replace with actual embeddings for Greeting 1
        np.random.rand(768).tolist(),  # Replace with actual embeddings for Greeting 2
        np.random.rand(768).tolist()   # Replace with actual embeddings for Small Talk 1
    ]
}
conversational_dataset = Dataset.from_dict(conversational_data)

conversational_dataset = conversational_dataset.add_faiss_index(column='embeddings')

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", indexed_dataset=conversational_dataset)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)
inputs = tokenizer(transcribed_text, return_tensors='pt')
inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
outputs = rag_model.generate(**inputs)
generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
with open("RAGoutput.txt", "w", encoding="utf-8") as fo:
    fo.write(generated_output)

print("Process completed successfully.")
