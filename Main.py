import whisper as wp
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset

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
with open("englishTranslation.txt", "w", encoding="utf-8") as f: f.write(transcribed_text)

# Dummy dataset for RAG
dummy_data = {
    "title": ["Sample Title 1", "Sample Title 2"],
    "context": ["This is the context of the first document.", "This is the context of the second document."]
}
dummy_dataset = Dataset.from_dict(dummy_data)

# Initialize RAG components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", indexed_dataset=dummy_dataset)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Tokenize the transcribed text
inputs = tokenizer(transcribed_text, return_tensors='pt')

# Generate output based on the transcribed/translated text
outputs = rag_model.generate(**inputs)
generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Save generated output to file
with open("RAGoutput.txt", "w", encoding="utf-8") as fo: fo.write(generated_output)




