import math
from collections import Counter
from datasets import load_dataset

# Load and tokenize the SETIMES Turkish dataset
dataset = load_dataset("setimes", "bg-tr")['train']
text_data = " ".join(item['translation']['tr'] for item in dataset)
words = text_data.split()  # Basic tokenization by whitespace

# Filter out words that occur fewer than 10 times
word_counts = Counter(words)
filtered_words = [word for word in words if word_counts[word] >= 10]

# Calculate bigram frequencies, excluding bigrams with low-frequency words
filtered_bigrams = [
    (w1, w2) for w1, w2 in zip(filtered_words, filtered_words[1:])
    if word_counts[w1] >= 10 and word_counts[w2] >= 10
]
bigram_counts = Counter(filtered_bigrams)

# Calculate total counts
total_bigrams = sum(bigram_counts.values())
total_words = len(filtered_words)

def calculate_pmi(bigram, bigram_counts, word_counts, total_bigrams, total_words):
    """
    Calculate the Pointwise Mutual Information (PMI) for a given bigram.
    
    Parameters:
        bigram (tuple of str): The word pair for which PMI is calculated.
        bigram_counts (Counter): Counts of all bigrams.
        word_counts (Counter): Counts of individual words.
        total_bigrams (int): Total count of bigrams.
        total_words (int): Total count of words in the filtered corpus.
    
    Returns:
        float: The calculated PMI for the given bigram.
    """
    w1, w2 = bigram
    p_w1_w2 = bigram_counts[bigram] / total_bigrams
    p_w1 = word_counts[w1] / total_words
    p_w2 = word_counts[w2] / total_words
    
    if p_w1 > 0 and p_w2 > 0:
        pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
        return pmi
    else:
        return float('-inf')  # Avoid pairs where either word frequency is zero

# Calculate PMI values for each bigram
pmi_values = {
    bigram: calculate_pmi(bigram, bigram_counts, word_counts, total_bigrams, total_words)
    for bigram in bigram_counts
}

# Sort bigrams by PMI values and select top and bottom 20 pairs
sorted_pmi = sorted(pmi_values.items(), key=lambda item: item[1])
top_20_pmi = sorted_pmi[-20:]   # Top 20 highest PMI values
bottom_20_pmi = sorted_pmi[:20] # Top 20 lowest PMI values

# Display results
print("Top 20 Word Pairs by PMI:")
for bigram, pmi in top_20_pmi:
    print(f"{bigram}: {pmi}")

print("\nBottom 20 Word Pairs by PMI:")
for bigram, pmi in bottom_20_pmi:
    print(f"{bigram}: {pmi}")
