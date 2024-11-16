from datasets import load_dataset
from nltk.probability import MLEProbDist
from ngram import BasicNgram

# Load the SETIMES Turkish dataset and extract text
dataset = load_dataset("setimes", "bg-tr")['train']
text_data = " ".join(item['translation']['tr'] for item in dataset)
words = text_data.split()  # Basic tokenization by whitespace

def train_ngram_model(n, corpus, estimator=MLEProbDist):
    """
    Train an n-gram model on the given corpus.
    
    Parameters:
        n (int): The size of the n-gram (context size + 1).
        corpus (list of str): List of words in the corpus.
        estimator (function): Smoothing method (default: MLE).
    
    Returns:
        BasicNgram: Trained BasicNgram model instance.
    """
    return BasicNgram(n, corpus, estimator=estimator)

def generate_text(ngram, n, num_words=100):
    """
    Generate text from the trained n-gram model.
    
    Parameters:
        ngram (BasicNgram): Trained BasicNgram model instance.
        n (int): The n-gram size used for context (context size + 1).
        num_words (int): Number of words to generate in the text.
    
    Returns:
        str: Generated text as a single string.
    """
    context = (ngram._start_symbol,) * (n - 1)  # Initialize context with start symbols
    generated_words = list(context)  # List to store generated words
    
    for _ in range(num_words):
        # Get the probability distribution for the current context
        prob_dist = ngram[context]
        
        # Generate the next word
        next_word = prob_dist.generate()
        
        # Append the word and update context
        generated_words.append(next_word)
        context = (*context[1:], next_word)
        
        # Reset context if end symbol is reached
        if next_word == ngram._end_symbol:
            context = (ngram._start_symbol,) * (n - 1)
    
    # Join and return generated words, ignoring padding symbols
    return ' '.join(word for word in generated_words if word not in {ngram._start_symbol, ngram._end_symbol})

# Train and generate samples with varying n-gram sizes
for n in range(2, 5):
    print(f"\nTraining {n}-gram model and generating text...")
    ngram = train_ngram_model(n, words)
    print(f"Generated Text (n={n}):\n")
    print(generate_text(ngram, n, num_words=100))
    print("\n" + "-" * 40)