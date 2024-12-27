from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

stop_words = set(stopwords.words('english'))
file_path = "/h/emzed/data/qa_discharge.csv"


def remove_related_sentences(entry):
    context = entry["text"]
    answer = str(entry["a"]) + " " + str(entry["q"])

    # Split context into sentences
    context_sentences = sent_tokenize(context)
    answer_sentences = sent_tokenize(answer)

    # Filter sentences based on keywords
    keyword_threshold = 0.1
    relevant_sentences = []
    for ans_sent in answer_sentences:
        keywords = set(word for word in ans_sent.split(" ") if word.lower() not in stop_words)
        relevant_sentences.extend([s for s in context_sentences if sum(word in s for word in keywords) >= len(keywords) * keyword_threshold])
    
    # Reconstruct the context with excluded sentences, preserving original order
    removed_context = " ".join([s for s in context_sentences if s not in relevant_sentences])
    return removed_context

df = pd.read_csv(file_path)
df = df.head(10_000)
df['masked_text'] = df.progress_apply(remove_related_sentences, axis=1)
df = df[['masked_text', 'q', 'a']]
df.to_csv("/h/emzed/data/qa_discharge_masked.csv", index=False)

