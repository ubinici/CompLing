# Findings and Discussion Report

## Problem 1 - Zipf's Law

The empirical analysis of Zipf’s Law across different textual corpora reveals key insights about word frequency distributions and the consistency of Zipfian behavior in natural language. 

In the linear plots, we observe a steep, rapid drop in frequency from the most common words to the least common ones. This decline demonstrates that only a few words appear frequently, while most words are used sparingly or only once. This pattern is consistent across all corpora, indicating a high concentration of frequency in a small subset of the vocabulary.

The log-log plots provide a clearer view of Zipf’s Law. Each plot displays an approximately straight, downward-sloping line, which aligns with Zipf’s prediction that word frequency is inversely proportional to its rank. This straight line on a log-log scale further confirms that the distribution of word frequencies across ranks is not random but follows a predictable pattern across languages and genres.

Despite being different genres (religious text vs. fiction), both King James Bible and The Jungle Book exhibit the same general pattern of frequency decay. This suggests that Zipf's Law applies universally across different forms of written language, likely due to shared linguistic structures in English.

Both languages of SETIMES corpora also follow a Zipfian distribution, though the specific slope and positioning may vary slightly due to linguistic differences in vocabulary density and morphological structure. For instance, Bulgarian and Turkish might have distinct high-frequency words due to their unique syntactic and grammatical structures, yet the overall distribution remains consistent with Zipf’s Law.

## Problem 2 - Random Text Generation

### Dependencies

The code relies on the following libraries:

- **Hugging Face Datasets**: For loading the SETIMES corpus. Install with `pip install datasets`.
- **NLTK**: For MLE probability distributions. Install with `pip install nltk`.
- **ngram**: For creating the n-gram language model. Courtesy of the course instructors.

The dataset used to train the model is **SETIMES Bulgarian-Turkish parallel corpus**. Load using `load_dataset("setimes", "bg-tr")` from Hugging Face. Only the Turkish side of this dataset was utilized for our purposes.

### Analysis

**2-gram model**

With the 2-gram model, the text generated is disjointed and lacks continuity. The model can only consider the previous word, leading to abrupt topic shifts and incoherent sentence structures. The creativity here is limited, as the phrases often make little sense beyond two-word pairings.

For instance, the number agreement between the recursive pronoun "kendilerini" (themselves) and the subject of the sentence "hırvat" (the Croatian) is non-existent. Both sentences are semantically distinct from one another by a considerable margin.

```
Hırvat şarap içerken kendilerini giderek uzaklaşmaktadır. SETimes: Vize Muafiyet Programına katıldı ve euro-dizel gibi faklı malzemelerle ve kesin olan kuzey Kıbrıs’ı da ısrar ediyor...
```

**3-gram model**

The 3-gram model introduces greater continuity, creating phrases that often make grammatical sense. However, thematic coherence is still limited. While creativity is slightly enhanced, allowing for more realistic word combinations, some sentences still feel random or lack a clear topic.

We see a slight improvement in the semantic consistency between two sentences as well. Despite some grammatical mistakes, we observe that the Croatian local elections is the main theme for this passage.

```
Hırvat yerel seçimleri için asker sayısını Fransız, Alman ve İngiliz makamlarını bulguları hakkında daha endişeli. "Umarım sorun çıkmaz; partiler için bir fırsat olacağını söyledi...
```

**4-gram model**

With the 4-gram model, the text generated is notably more cohesive, with sentences that resemble actual Turkish syntax and structure. Contextual awareness allows for more specific topic continuity, and while there may be some repetition, the model generates more readable and coherent text. The creativity is balanced with improved realism in sentence formation.

Grammatically and semantically, there is a huge improvement from both other examples, too. The text generator went for the Croatian local elections once again, and there is virtually no grammatical mistakes being observed. 

```
Hırvat yerel seçimleri ikinci tura gidiyor. Yerel halk ilk tur seçimler için Pazar günü sandık başına gidecek. Sanader ve iktidardaki HDZ partisi, 2003 yılında iktidara gelmesinden bu yana...
```

## Problem 3 - Statistical Dependence

### Analysis

The PMI analysis on the Turkish SETIMES corpus reveals clear patterns:

1. Top PMI Word Pairs:

High PMI values indicate strong associations, often due to proper nouns and named entities (such as `('Fredrik', 'Reinfeldt')` and `('Jerzy', 'Buzek')`) which occur together due to contextual relevance, or because of specialized terms and citation styles (such as `('rahim', 'ağzı')` for "cervix", `['Linda', 'Karadaku/SETimes]`) which appear in specific, recurring formats. These pairs show that the presence of one word strongly predicts the other, violating the independence assumption of unigram models.

```
('[Dünya', 'Bankası]'): 12.267856125023457
('Fredrik', 'Reinfeldt'): 12.267856125023457
('Ta', 'Nea'): 12.267856125023457
('Molotof', 'kokteylleri'): 12.290328980875517
('rahim', 'ağzı'): 12.290328980875517
('Kaderini', 'Tayin)'): 12.305596453006304
('[Misko', 'Taleski/SETimes]'): 12.330914260990594
('[Andy', 'Dabilis/SETimes]'): 12.336848996510408
('[Linda', 'Karadaku/SETimes]'): 12.336848996510408
('[Aleksandar', 'Pavlevski/SETimes]'): 12.336848996510408
('Jerzy', 'Buzek'): 12.354867502013088
('Katma', 'Değer'): 12.38563916067984
('Andrus', 'Ansip'): 12.38563916067984
('[Klaudija', 'Lutovska/SETimes]'): 12.41095696866413
('Nasya', 'Nenova,'): 12.490999676337667
('Seremb', "Gjergji'nin"): 12.578011053327296
('paha', 'biçilmez'): 12.578011053327296
('[Ana', 'Pekmezi/SETimes]'): 12.578011053327296
('hatırı', 'sayılır'): 12.578011053327296
('Erl', "Murati'nin"): 12.673321233131622
```

2. Bottom PMI Word Pairs:

Low PMI values are seen with function words (such as `('bir', 'arasında')` and `('ve', 'haberi')`) and grammatically common, non-specific terms. These words pair frequently with various others, supporting the independence assumption of unigram models.

```
('European', 've'): -4.985408340934488
('ve', 'haberi'): -4.812015981657611
('en', 'bir'): -4.7448453914117055
('European', 'bir'): -4.718831757557643
('olmak', 'bir'): -4.556000220598508
('haberi', 'bir'): -4.545439398280766
('ve', 'belirtti.'): -4.521633941315218
('bir', 'etti.'): -4.456956215303804
('bir', 'arasında'): -4.454052251429399
('bir', 'şöyle'): -4.431220529213434
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('European', 'bir'): -4.718831757557643
('olmak', 'bir'): -4.556000220598508
('haberi', 'bir'): -4.545439398280766
('ve', 'belirtti.'): -4.521633941315218
('bir', 'etti.'): -4.456956215303804
('bir', 'arasında'): -4.454052251429399
('bir', 'şöyle'): -4.431220529213434
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('haberi', 'bir'): -4.545439398280766
('ve', 'belirtti.'): -4.521633941315218
('bir', 'etti.'): -4.456956215303804
('bir', 'arasında'): -4.454052251429399
('bir', 'şöyle'): -4.431220529213434
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('bir', 'etti.'): -4.456956215303804
('bir', 'arasında'): -4.454052251429399
('bir', 'şöyle'): -4.431220529213434
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('bir', 'arasında'): -4.454052251429399
('bir', 'şöyle'): -4.431220529213434
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('bu', 'olan'): -4.4175242323991055
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('yanı', 've'): -4.383796472414141
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('ve', 'konuşan'): -4.332870592453781
('bir', 'bin'): -4.378867778332368
('günü', 'da'): -4.341071698038863
('ve', 'konuşan'): -4.332870592453781
('günü', 'da'): -4.341071698038863
('ve', 'konuşan'): -4.332870592453781
('ve', 'alıyor.'): -4.3295999435856585
('ve', 'konuşan'): -4.332870592453781
('ve', 'alıyor.'): -4.3295999435856585
('bir', 'Dışişleri'): -4.299347478223233
('bir', 'Çarşamba'): -4.285019652526418
('bir', 'Dışişleri'): -4.299347478223233
('bir', 'Çarşamba'): -4.285019652526418
('ve', 'verdi.'): -4.272296989744013
('ve', 'verdi.'): -4.272296989744013
```

### Discussion

While unigram models are effective for capturing frequency-based statistics of independent, high-frequency terms, they are limited by the independence assumption and fail to account for meaningful contextual associations. The PMI analysis highlights that real-world language contains dependencies, and these dependencies are crucial for accurate language modeling. This suggests that more complex models, such as bigrams or neural models, which account for word context and association, are better suited for tasks requiring nuanced language understanding.
