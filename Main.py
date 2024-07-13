import whisper as wp
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = wp.load_model("small")
audio_path = r"C:\projects\Wrop\test3.mp3"
result = model.transcribe(audio_path)
extracted_text = result.get('text', 'Error during STT conversion')
with open("originalLangTranscript.txt", "w", encoding="utf-8" ) as f: f.write(extracted_text)

def lang_identifier(result):
    identified_Spoken_Language = result.get("language", None)
    return identified_Spoken_Language

identified_language = lang_identifier(result)
print(f"Language identified: {identified_language}")

if identified_language.lower() != "en":
    new_result = model.transcribe(audio_path, task="translate")
    new_text = new_result.get('text', "error durring stt conversion")
    with open("EnglishTranslation.txt", "w", encoding="utf-8") as fe: fe.write(new_text)

tokenize = RagTokenizer.from_pretrained("facebook/rag-token-base")
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train", trust_remote_code=True)
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", indexed_dataset=dataset)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

with open("EnglishTranslation.txt", "r", encoding="utf-8") as fe: prompt = fe.read()

input = tokenize(prompt, return_tensors='pt')
output = rag_model.generate(**input)
generated_output = tokenize.batch_decode(output, skip_special_tokens=True)[0]
with open("RAGoutput.txt", "w", encoding="utf-8") as fo: fo.write(generated_output)
exit()
