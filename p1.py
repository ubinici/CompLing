from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import re

# Load datasets with proper paths and setup
corpora = {
    "King James Bible": load_dataset("text", data_files={"train": "kingjamesbible_tokenized.txt"})['train'],
    "The Jungle Book": load_dataset("text", data_files={"train": "junglebook.txt"})['train'],
    "SETIMES": load_dataset("setimes", "bg-tr")['train']
}

# Function to calculate word frequency
def calc_freq(text):
    tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize to words
    count_tokens = Counter(tokens)
    freqs = [count for _, count in count_tokens.most_common()]
    return freqs

# Calculate frequencies for each corpus
frequencies = {}
for name, ds in corpora.items():
    if name == "SETIMES":
        # Separate Bulgarian and Turkish data
        bulgarian_text = " ".join(item['translation']['bg'] for item in ds)
        turkish_text = " ".join(item['translation']['tr'] for item in ds)
        frequencies["SETIMES Bulgarian"] = calc_freq(bulgarian_text)
        frequencies["SETIMES Turkish"] = calc_freq(turkish_text)
    else:
        # Concatenate all text for single-language corpora
        text_data = " ".join(item['text'] for item in ds)
        frequencies[name] = calc_freq(text_data)

# Plotting function with an option to limit points for clarity
def plot_freq(freq, title, limit=5000):
    plt.figure(figsize=(12, 5))
    
    # Linear plot
    plt.subplot(1, 2, 1)
    plt.plot(freq[:limit], marker="o")
    plt.title(f'{title} - Linear Scale')
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    
    # Log-log plot
    plt.subplot(1, 2, 2)
    plt.loglog(freq[:limit], marker="o")
    plt.title(f'{title} - Log-Log Scale')
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    
    plt.tight_layout()
    plt.show()

# Plot for each corpus
for name, freq in frequencies.items():
    plot_freq(freq, name)