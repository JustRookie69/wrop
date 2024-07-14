import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import whisper as wp
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
import numpy as np

# Load Whisper model
model = wp.load_model("small")

# Transcribe the audio
audio_path = r"C:\projects\Wrop\test3.mp3"
result = model.transcribe(audio_path)
extracted_text = result.get('text', 'Error during STT conversion')

# Identify the language
identified_language = result.get("language", None)
print(f"Language identified: {identified_language}")

# Translate if not English
if identified_language.lower() != "en":
    translated_result = model.transcribe(audio_path, task="translate")
    transcribed_text = translated_result.get('text', 'Error during translation')
else:
    transcribed_text = extracted_text

# Save transcribed/translated text to file
with open("englishTranslation.txt", "w", encoding="utf-8") as f:
    f.write(transcribed_text)

# Dummy dataset with precomputed embeddings
dummy_data = {
    "title": ["Document 1", "Document 2", "Document 3"],
    "text": ["Translate to French", "Explain me the objective", "Give me ideal reply as assistant"],
    "embeddings": [
        np.random.rand(768).tolist(),  # Replace with actual embeddings for Document 1
        np.random.rand(768).tolist(),  # Replace with actual embeddings for Document 2
        np.random.rand(768).tolist()   # Replace with actual embeddings for Document 3
    ]
}
dummy_dataset = Dataset.from_dict(dummy_data)

# Add FAISS index
dummy_dataset = dummy_dataset.add_faiss_index(column='embeddings')

# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", indexed_dataset=dummy_dataset)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Tokenize the transcribed text
inputs = tokenizer(transcribed_text, return_tensors='pt')

# Remove unused keys
inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

# Generate output based on the transcribed/translated text
outputs = rag_model.generate(**inputs)
generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Save generated output to file
with open("RAGoutput.txt", "w", encoding="utf-8") as fo:
    fo.write(generated_output)

print("Process completed successfully.")
