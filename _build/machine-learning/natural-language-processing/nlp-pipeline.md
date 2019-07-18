---
interact_link: content/machine-learning/natural-language-processing/nlp-pipeline.ipynb
kernel_name: python3
has_widgets: false
title: 'Natural Language Processing Pipeline'
prev_page:
  url: /machine-learning/natural-language-processing/what-is-natural-language-processing
  title: 'Natural Language Processing'
next_page:
  url: /machine-learning/natural-language-processing/word2vec-cbow
  title: 'Word2Vec Part I - Continuous Bag of Words'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# NLP Pipeline

In this notebook, we'll go over a general pipeline used in solving natural language processing
problem. The workflow is as follows:

1. Text Processing
2. Feature Extraction 
3. Modeling



---
# Text Processing

Take raw input text, clean it, normalize it, and convert it into a form that is suitable for feature extraction.

1. Cleaning:
    - **Cleaning** to remove irrelevant items, such as HTML tags
2. Normalization:
    - **Normalizing** by converting to all lowercase and removing punctuation
3. Tokenization
    - Splitting text into words or **tokens**
4. Stop Word Removal
    - Removing words that are too common, also known as **stop words**
5. Part of Speech Tagging (POS Tagging) and Named Entity Recognition (NER):
    - Identifying different **parts of speech** and **named entities**
6. Stemming and Lemmatization
    - Converting words into their dictionary forms, using **stemming and lemmatization**

Extracting plain text: 
- Textual data can come from a wide variety of sources: the web, PDFs, word documents, speech recognition systems, book scans, etc. Your goal is to extract plain text that is free of any source specific markup or constructs that are not relevant to your task.

Reducing complexity: 
- Some features of our language like capitalization, punctuation, and common words such as a, of, and the, often help provide structure, but don't add much meaning. Sometimes it's best to remove them if that helps reduce the complexity of the procedures you want to apply later.



## 1. Cleaning

We'll first get recent news about S&P 500 from MarketWatch, remove all the unecessary html tags and download the news text from each of the recent news links. This will become our text corpus.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Library imports
import re
from tqdm import tqdm
import numpy as np
import requests
from bs4 import BeautifulSoup
import string

# Fetch web page
response = requests.get('https://www.marketwatch.com/investing/index/spx')

# dict to store articles
articles = {}

# Remove HTML Tags
if response.ok:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get all news articles
    for news in soup.findAll(name='mw-scrollable-news'):
        news_type = news.findChild(name='div', attrs={'class': 'collection__list j-scrollElement'})['data-type']
        news_url_list = [article.findChildren(name='a')[0]['href'] for article in news.findChildren(
            name='h3', 
            attrs={'class': 'article__headline'}
        )]
        articles[news_type] = news_url_list 

articles['MarketWatch']

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['https://www.marketwatch.com/story/asian-markets-rally-on-encouraging-trade-developments-2019-06-09?mod=mw_quote_news',
 'https://www.marketwatch.com/story/time-to-panic-on-economy-no-but-ongoing-trade-wars-give-a-taste-of-unpleasant-future-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/stock-market-investors-discover-they-cant-ignore-politicians-anymore-2019-06-08?mod=mw_quote_news',
 'https://www.marketwatch.com/articles/retirement-in-a-bear-market-51559342111?mod=mw_quote_news',
 'https://www.marketwatch.com/story/some-baby-boomers-say-doctors-arent-giving-them-enough-information-about-cannabis-2019-06-03?mod=mw_quote_news',
 'https://www.marketwatch.com/story/stitch-fix-is-on-a-growth-trajectory----here-are-2-reasons-why-2019-06-06?mod=mw_quote_news',
 'https://www.marketwatch.com/story/value-stocks-are-trading-at-the-steepest-discount-in-history-2019-06-06?mod=mw_quote_news',
 'https://www.marketwatch.com/story/heres-one-big-wrinkle-if-the-feds-hit-big-tech-with-antitrust-cases-2019-06-06?mod=mw_quote_news',
 'https://www.marketwatch.com/story/pagerduty-ipo-pays-off-as-customer-additions-crush-expectations-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/why-the-jobs-report-could-give-the-junk-bond-market-a-second-wind-2019-06-06?mod=mw_quote_news',
 'https://www.marketwatch.com/story/another-bad-sign-in-jobs-reportbreadth-of-companies-hiring-at-two-year-low-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/uber-stock-hit-by-executive-shake-up-a-month-after-the-ipo-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/revolve-closes-its-first-trading-day-up-94-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/fedex-opts-out-of-express-service-contract-with-amazon-to-focus-on-broader-e-commerce-market-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/jeff-bezos-is-proud-of-ex-wifes-pledge-to-give-away-over-half-of-her-35-billion-fortune-go-get-em-mackenzie-2019-05-28?mod=mw_quote_news',
 'https://www.marketwatch.com/story/forget-elon-musks-marijuana-smoking-security-clearance-troubles-just-investing-in-cannabis-stocks-can-cause-you-problems-2019-03-11?mod=mw_quote_news',
 'https://www.marketwatch.com/story/how-summer-fridays-could-let-employers-off-the-hook-2019-06-07?mod=mw_quote_news',
 'https://www.marketwatch.com/story/1-in-3-us-adults-is-interested-in-using-legalized-cannabis-but-not-for-the-reason-youd-think-2019-06-05?mod=mw_quote_news',
 'https://www.marketwatch.com/articles/jpmorgan-chase-stock-is-a-solid-bet-with-ceo-jamie-dimon-51559956262?mod=mw_quote_news']
```


</div>
</div>
</div>



Let's get all the article text from each url we've gathered.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Text corpus
corpus = []

# Flatten list function
flatten = lambda l: [item for sublist in l for item in sublist]

# Extract the text from each article link from MarketWatch
# WSJ requires an account and SeekingAlpha checks for robots
for article in tqdm(articles['MarketWatch']):
    
    # Fetch web page
    response = requests.get(article)
    
    if response.ok:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = str()
        
        # Remove last 4 paragraphs because
        # they are unecessary:
        # Reporter name, Copyright stuff...
        for paragraph in s.findAll('p')[:-4]:
            article_text += ' ' + ' '.join(paragraph.get_text().strip().split())
            
    corpus.append(article_text)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
corpus[0]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
' Published: June 9, 2019 6:35 p.m. ET Slowdown in hiring and wage growth reflect weak spots in economy By The economy looked like it was perking up a few months ago, but now gray clouds are moving in. The latest shadow was cast by a dismal May employment report that showed a meager 75,000 increase in new jobs along with slowing wage growth. Thankfully the unemployment rate held fast at a 49-year low of 3.6%. Read: U.S. adds a meager 75,000 jobs in May in warning sign for economy Also: The worst part of a crummy jobs report might be ebbing pay gains for workers One poor jobs report is usually nothing to worry about, but the slowdown in hiring in May is part of a broader trend. The U.S. has added an average of 151,000 jobs in the past three months, down from a recent high of 238,000 at the start of 2019. Economists place a large share of the blame squarely on festering trade tensions between the U.S. and China. The dispute has hurt the global economy, crimped U.S. exports, damaged American manufacturers and rattled corporate executives and small-business owners alike. “The May U.S. jobs report gave us a taste of what’s ahead if trade war threats continue to escalate and tariffs continue to go higher,” said chief economist Scott Anderson of Bank of the West. “This this is not just a one-off hiccup in the data, but part of a broader more prolonged pattern of labor market softening.” The U.S. economy, to be sure, doesn’t appear in danger of imminent recession. The strongest labor market in decades has elevated consumer confidence and stoked steady household spending, for one thing. And the Federal Reserve earlier this year put a halt to further interest-rate increases, a sharp turnabout that’s led to sharply lower borrowing costs for businesses and consumers. “Is it time to hit the panic button? Probably not,” said Ward McCarthy, chief financial economist at Jefferies LLC. That doesn’t mean investors, ordinary Americans and the Fed shouldn’t worry. “A low jobless rate, elevated consumer confidence, and firmer wage growth suggest the broader economy is still on firm footing,” said chief economist Richard Moody of Regions Financial, “but a similarly weak June employment report would be an ominous sign.” Read: Here’s another bad sign in the jobs report The Fed is unlikely to cut interest rates in June, economists say, so the pressure is likely to grow on the White House to hasten negotiations with China and to resolve a conflict with Mexico over illegal immigration that spurred President Trump to threaten to apply tariffs to all Mexican imports. “The sooner the U.S. can steer out of choppy water, the faster our economy will expand,” said Michael D. Farren, an economist and research fellow at the Mercatus Center at George Mason University. Some temporary relief could come next week if retail sales rebound in May as expected. Economists forecast a solid 0.6% increase in sales, which would support the idea that households are still spending at a steady pace. A pair of inflation barometers, meanwhile, are likely to show that price pressures remain muted. Low inflation gives the Fed further ammunition to cut interest rates if the central bank thinks the economy needs support. The rising chances of a Fed rate cut actually spurred a rally in the stock market last week despite the dispute with Mexico and the weak employment gains in May. The Dow Jones Industrial Average DJIA, +1.02% and S&P 500 SPX, +1.05% ripped off fourth straight daily gains and the yield on the 10-year Treasury yield TMUBMUSD10Y, +1.70% fell to a 21-month low of 2.06%.'
```


</div>
</div>
</div>



## 2. Normalization

Let's change everything to lower case and remove punctuation.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
norm_corpus = [' '.join(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()) for text in corpus]
norm_corpus[0]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'published june 9 2019 6 35 p m et slowdown in hiring and wage growth reflect weak spots in economy by the economy looked like it was perking up a few months ago but now gray clouds are moving in the latest shadow was cast by a dismal may employment report that showed a meager 75 000 increase in new jobs along with slowing wage growth thankfully the unemployment rate held fast at a 49 year low of 3 6 read u s adds a meager 75 000 jobs in may in warning sign for economy also the worst part of a crummy jobs report might be ebbing pay gains for workers one poor jobs report is usually nothing to worry about but the slowdown in hiring in may is part of a broader trend the u s has added an average of 151 000 jobs in the past three months down from a recent high of 238 000 at the start of 2019 economists place a large share of the blame squarely on festering trade tensions between the u s and china the dispute has hurt the global economy crimped u s exports damaged american manufacturers and rattled corporate executives and small business owners alike the may u s jobs report gave us a taste of what s ahead if trade war threats continue to escalate and tariffs continue to go higher said chief economist scott anderson of bank of the west this this is not just a one off hiccup in the data but part of a broader more prolonged pattern of labor market softening the u s economy to be sure doesn t appear in danger of imminent recession the strongest labor market in decades has elevated consumer confidence and stoked steady household spending for one thing and the federal reserve earlier this year put a halt to further interest rate increases a sharp turnabout that s led to sharply lower borrowing costs for businesses and consumers is it time to hit the panic button probably not said ward mccarthy chief financial economist at jefferies llc that doesn t mean investors ordinary americans and the fed shouldn t worry a low jobless rate elevated consumer confidence and firmer wage growth suggest the broader economy is still on firm footing said chief economist richard moody of regions financial but a similarly weak june employment report would be an ominous sign read here s another bad sign in the jobs report the fed is unlikely to cut interest rates in june economists say so the pressure is likely to grow on the white house to hasten negotiations with china and to resolve a conflict with mexico over illegal immigration that spurred president trump to threaten to apply tariffs to all mexican imports the sooner the u s can steer out of choppy water the faster our economy will expand said michael d farren an economist and research fellow at the mercatus center at george mason university some temporary relief could come next week if retail sales rebound in may as expected economists forecast a solid 0 6 increase in sales which would support the idea that households are still spending at a steady pace a pair of inflation barometers meanwhile are likely to show that price pressures remain muted low inflation gives the fed further ammunition to cut interest rates if the central bank thinks the economy needs support the rising chances of a fed rate cut actually spurred a rally in the stock market last week despite the dispute with mexico and the weak employment gains in may the dow jones industrial average djia 1 02 and s p 500 spx 1 05 ripped off fourth straight daily gains and the yield on the 10 year treasury yield tmubmusd10y 1 70 fell to a 21 month low of 2 06'
```


</div>
</div>
</div>



## 3. Tokenization

Let's use the Natural Language Toolkit (NLTK) to tokenize our corpus.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import nltk

# Punkt sentence tokenizer models 
# that help to detect sentence boundaries
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package punkt to
[nltk_data]     /Users/jeffchenchengyi/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
```
</div>
</div>
</div>



Let's tokenize our corpus by sentence first, then by words.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
tokenized_corpus = [[word_tokenize(sentence) for sentence in sent_tokenize(text)] for text in corpus]
tokenized_corpus[0][2]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['The',
 'latest',
 'shadow',
 'was',
 'cast',
 'by',
 'a',
 'dismal',
 'May',
 'employment',
 'report',
 'that',
 'showed',
 'a',
 'meager',
 '75,000',
 'increase',
 'in',
 'new',
 'jobs',
 'along',
 'with',
 'slowing',
 'wage',
 'growth',
 '.']
```


</div>
</div>
</div>



## 4. Stop Word Removal

We will remove all the stop words ('the', 'at', 'it') in each tokenized sentence in our corpus.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove stop words
stopwords_removed_corpus = [[[token for token in sentence_tokens if token not in stopwords.words("english")] for sentence_tokens in text] for text in tokenized_corpus]
stopwords_removed_corpus[0][2]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/jeffchenchengyi/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['The',
 'latest',
 'shadow',
 'cast',
 'dismal',
 'May',
 'employment',
 'report',
 'showed',
 'meager',
 '75,000',
 'increase',
 'new',
 'jobs',
 'along',
 'slowing',
 'wage',
 'growth',
 '.']
```


</div>
</div>
</div>



## 5a. POS Tagging

The identification of the type of word used in a sentence - which words are nouns, pronouns, verbs, adverbs.

- Rule-Based POS Tagging:
    - Defines a set of rules, e.g. if the preceding word is an article, then the word in question must be a noun. This information is coded in the form of rules.
        - EngCG Tagger
        - Brill's Tagger
            - Goes through the training data and finds out the set of tagging rules that best define the data and minimize POS tagging errors. The most important point to note here about Brill’s tagger is that the rules are not hand-crafted, but are instead found out using the corpus provided. The only feature engineering required is a set of rule templates that the model can use to come up with new features.
- Stochastic POS Tagging:
    - Any model which somehow incorporates frequency or probability may be properly labelled stochastic.
        - Word frequency measurements Methods:
        - Tag sequence Probability Methods:
        - Both (Sequence):
            - Hidden Markov Model (HMM)
            - MEMM
            - Conditional Random Field (CRF)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Averaged Perceptron POS tagging model
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

pos_tagged_corpus = [[pos_tag(sentence_tokens) for sentence_tokens in text] for text in tokenized_corpus]
pos_tagged_corpus[0][2]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /Users/jeffchenchengyi/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[('The', 'DT'),
 ('latest', 'JJS'),
 ('shadow', 'NN'),
 ('was', 'VBD'),
 ('cast', 'VBN'),
 ('by', 'IN'),
 ('a', 'DT'),
 ('dismal', 'NN'),
 ('May', 'NNP'),
 ('employment', 'NN'),
 ('report', 'NN'),
 ('that', 'WDT'),
 ('showed', 'VBD'),
 ('a', 'DT'),
 ('meager', 'NN'),
 ('75,000', 'CD'),
 ('increase', 'NN'),
 ('in', 'IN'),
 ('new', 'JJ'),
 ('jobs', 'NNS'),
 ('along', 'IN'),
 ('with', 'IN'),
 ('slowing', 'NN'),
 ('wage', 'NN'),
 ('growth', 'NN'),
 ('.', '.')]
```


</div>
</div>
</div>



## 5b. NER



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# NER model
nltk.download('maxent_ne_chunker')
from nltk import ne_chunk

# Recognize named entities in a pos tagged sentence from corpus
tree = ne_chunk(pos_tagged_corpus[0][6])
print(tree)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package maxent_ne_chunker to
[nltk_data]     /Users/jeffchenchengyi/nltk_data...
[nltk_data]   Package maxent_ne_chunker is already up-to-date!
(S
  Economists/NNS
  place/VBP
  a/DT
  large/JJ
  share/NN
  of/IN
  the/DT
  blame/NN
  squarely/RB
  on/IN
  festering/VBG
  trade/NN
  tensions/NNS
  between/IN
  the/DT
  (GPE U.S./NNP)
  and/CC
  (GPE China/NNP)
  ./.)
```
</div>
</div>
</div>



## 6. Stemming and Lemmatization

Stemming:
- Removing the "ed", "ing", "es" from "changed", "changing", and "changes" to become "chang" (Not a real word)

Lemmatization:
- Removing the "ed", "ing", "es" from "changed", "changing", and "changes" to become "change" (A real word)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Model for Stemming and Lemmatization
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stemmed_corpus = [[[PorterStemmer().stem(token) for token in sentence_tokens] for sentence_tokens in text] for text in stopwords_removed_corpus]
stemmed_corpus[0][6]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[nltk_data] Downloading package wordnet to
[nltk_data]     /Users/jeffchenchengyi/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
```
</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['economist',
 'place',
 'larg',
 'share',
 'blame',
 'squar',
 'fester',
 'trade',
 'tension',
 'u.s.',
 'china',
 '.']
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
lemmed_corpus = [[[WordNetLemmatizer().lemmatize(token, pos='v') for token in sentence_tokens] for sentence_tokens in text] for text in stopwords_removed_corpus]
lemmed_corpus[0][6]

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['Economists',
 'place',
 'large',
 'share',
 'blame',
 'squarely',
 'fester',
 'trade',
 'tensions',
 'U.S.',
 'China',
 '.']
```


</div>
</div>
</div>



---
# Feature Extraction

Extract and produce context / feature representations that are appropriate for the type of NLP task you are trying to accomplish and the type of model you are planning to use. The following approaches are the most well-known for converting text into vectors / matrices that an ML model can understand:

Creates a [document-term matrix](https://en.wikipedia.org/wiki/Document-term_matrix) with terms - words / tokens / n-grams - as the columns and documents as the rows of the matrix:
- Bag of Words (BoW)
- Term Frequency - Inverse Document Frequency (TF-IDF)

Creates vector representations for each word:
- Word Embeddings



## BoW

A general method that's called a “bag” of words because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. Hence, to create any BoW model, we need to do 2 things:
1. Define a **vocabulary** of known words / tokens / terms by going through our entire text corpus and finding all the unique word tokens. 
2. Define a **score measure of the presence** of the words in our vocabulary for each document and subsequently, each document will be represented by a vector containing the **score measure of the presence** of the vocabulary word as the entry in each position.

All variations of BoW will differ in complexity by how we carry out the 2 steps above.



### 1. Binary BoW
- With the simplest implementation of bag of words, we're trying to build a one-hot encoded vector for each document, where the classes are the words in our vocabulary (aggregated from our text corpus).

**vocabulary**: Set of all the unique words in our text corpus after text processing.

**score measure of the presence**: Because it's a "one-hot" encoding, all values of the vector will take on either a 1 (if the vocabulary word exists in the document) or a 0 (if the vocabulary word does not exist in the document).



### 2. Count Occurence BoW

Similar to the Binary BoW, but instead of having either 1 or 0 for each entry in each document vector, we have the count of the vocabulary word in each document.

**vocabulary**: Set of all the unique words in our text corpus after text processing.

**score measure of the presence**: Count of the number of occurence of the vocabulary word



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(tokenizer=lambda text: [WordNetLemmatizer().lemmatize(token.strip(), pos='v') for token in word_tokenize(text.lower()) if token not in stopwords.words("english") and token not in list(string.punctuation)])
X = vect.fit_transform(corpus) # get counts of each token (word) in text data
X.toarray() # convert sparse matrix to numpy array to view

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[1, 1, 1, ..., 6, 6, 6],
       [1, 1, 1, ..., 6, 6, 6],
       [1, 1, 1, ..., 6, 6, 6],
       ...,
       [1, 1, 1, ..., 6, 6, 6],
       [1, 1, 1, ..., 6, 6, 6],
       [1, 1, 1, ..., 6, 6, 6]], dtype=int64)
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
vect.vocabulary_

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
{'publish': 189,
 'june': 133,
 '9': 16,
 '2019': 7,
 '6:35': 14,
 'p.m.': 173,
 'et': 79,
 'slowdown': 223,
 'hire': 111,
 'wage': 267,
 'growth': 105,
 'reflect': 199,
 'weak': 272,
 'spot': 229,
 'economy': 75,
 'look': 142,
 'like': 139,
 'perk': 181,
 'months': 158,
 'ago': 19,
 'gray': 103,
 'cloud': 47,
 'move': 160,
 'latest': 137,
 'shadow': 214,
 'cast': 40,
 'dismal': 67,
 'may': 148,
 'employment': 77,
 'report': 203,
 'show': 218,
 'meager': 150,
 '75,000': 15,
 'increase': 123,
 'new': 164,
 'job': 130,
 'along': 22,
 'slow': 222,
 'thankfully': 248,
 'unemployment': 262,
 'rate': 193,
 'hold': 113,
 'fast': 85,
 '49-year': 12,
 'low': 143,
 '3.6': 11,
 'read': 195,
 'u.s.': 261,
 'add': 18,
 'warn': 270,
 'sign': 219,
 'also': 23,
 'worst': 278,
 'part': 177,
 'crummy': 58,
 'might': 157,
 'ebb': 72,
 'pay': 180,
 'gain': 98,
 'workers': 276,
 'one': 168,
 'poor': 183,
 'usually': 266,
 'nothing': 166,
 'worry': 277,
 'broader': 37,
 'trend': 258,
 'average': 31,
 '151,000': 5,
 'past': 178,
 'three': 253,
 'recent': 197,
 'high': 109,
 '238,000': 10,
 'start': 233,
 '2019.': 8,
 'economists': 74,
 'place': 182,
 'large': 135,
 'share': 215,
 'blame': 35,
 'squarely': 232,
 'fester': 91,
 'trade': 256,
 'tensions': 247,
 'china': 45,
 'dispute': 68,
 'hurt': 117,
 'global': 101,
 'crimp': 57,
 'export': 83,
 'damage': 62,
 'american': 24,
 'manufacturers': 145,
 'rattle': 194,
 'corporate': 54,
 'executives': 80,
 'small-business': 224,
 'owners': 171,
 'alike': 21,
 '“': 283,
 'give': 100,
 'us': 265,
 'taste': 245,
 '’': 282,
 'ahead': 20,
 'war': 268,
 'threats': 252,
 'continue': 53,
 'escalate': 78,
 'tariff': 244,
 'go': 102,
 'higher': 110,
 '”': 284,
 'say': 212,
 'chief': 44,
 'economist': 73,
 'scott': 213,
 'anderson': 27,
 'bank': 33,
 'west': 274,
 'one-off': 169,
 'hiccup': 108,
 'data': 64,
 'prolong': 188,
 'pattern': 179,
 'labor': 134,
 'market': 146,
 'softening.': 225,
 'sure': 243,
 'appear': 29,
 'danger': 63,
 'imminent': 121,
 'recession': 198,
 'strongest': 240,
 'decades': 65,
 'elevate': 76,
 'consumer': 51,
 'confidence': 49,
 'stoke': 238,
 'steady': 234,
 'household': 115,
 'spend': 228,
 'thing': 249,
 'federal': 87,
 'reserve': 205,
 'earlier': 71,
 'year': 280,
 'put': 190,
 'halt': 106,
 'interest-rate': 127,
 'sharp': 216,
 'turnabout': 260,
 'lead': 138,
 'sharply': 217,
 'lower': 144,
 'borrow': 36,
 'cost': 55,
 'businesses': 38,
 'consumers': 52,
 'time': 254,
 'hit': 112,
 'panic': 176,
 'button': 39,
 'probably': 187,
 'ward': 269,
 'mccarthy': 149,
 'financial': 92,
 'jefferies': 129,
 'llc': 141,
 'mean': 151,
 'investors': 128,
 'ordinary': 170,
 'americans': 25,
 'feed': 88,
 'jobless': 131,
 'firmer': 94,
 'suggest': 241,
 'still': 236,
 'firm': 93,
 'foot': 95,
 'richard': 208,
 'moody': 159,
 'regions': 200,
 'similarly': 221,
 'would': 279,
 'ominous': 167,
 'sign.': 220,
 'another': 28,
 'bad': 32,
 'unlikely': 264,
 'cut': 59,
 'interest': 126,
 'rat': 192,
 'pressure': 185,
 'likely': 140,
 'grow': 104,
 'white': 275,
 'house': 114,
 'hasten': 107,
 'negotiations': 163,
 'resolve': 206,
 'conflict': 50,
 'mexico': 155,
 'illegal': 119,
 'immigration': 120,
 'spur': 230,
 'president': 184,
 'trump': 259,
 'threaten': 251,
 'apply': 30,
 'mexican': 154,
 'import': 122,
 'sooner': 227,
 'steer': 235,
 'choppy': 46,
 'water': 271,
 'faster': 86,
 'expand': 81,
 'michael': 156,
 'd.': 60,
 'farren': 84,
 'research': 204,
 'fellow': 90,
 'mercatus': 153,
 'center': 41,
 'george': 99,
 'mason': 147,
 'university': 263,
 'temporary': 246,
 'relief': 201,
 'could': 56,
 'come': 48,
 'next': 165,
 'week': 273,
 'retail': 207,
 'sales': 211,
 'rebound': 196,
 'expect': 82,
 'forecast': 96,
 'solid': 226,
 '0.6': 3,
 'support': 242,
 'idea': 118,
 'households': 116,
 'pace': 174,
 'pair': 175,
 'inflation': 125,
 'barometers': 34,
 'meanwhile': 152,
 'price': 186,
 'remain': 202,
 'mute': 161,
 'ammunition': 26,
 'central': 42,
 'think': 250,
 'need': 162,
 'rise': 210,
 'chance': 43,
 'actually': 17,
 'rally': 191,
 'stock': 237,
 'last': 136,
 'despite': 66,
 'dow': 70,
 'jones': 132,
 'industrial': 124,
 'djia': 69,
 '+1.02': 0,
 'p': 172,
 '500': 13,
 'spx': 231,
 '+1.05': 1,
 'rip': 209,
 'fourth': 97,
 'straight': 239,
 'daily': 61,
 'yield': 281,
 '10-year': 4,
 'treasury': 257,
 'tmubmusd10y': 255,
 '+1.70': 2,
 'fell': 89,
 '21-month': 9,
 '2.06': 6}
```


</div>
</div>
</div>



### 3. Normalized Count Occurence BoW with TF Weights

When there are tokens with extremely high occurence in our document vectors, it might cause model bias by making the model unecessarily sensitive to the scale. To correct this, we convert the raw counts of token occurences by the total number of words in each document (L1 norm) or euclidean distance (L2 norm) of each document vector.

**vocabulary**: Set of all the unique words in our text corpus after text processing.

**score measure of the presence**: Term Frequency (Count of the number of occurence of the vocabulary word divided by total number of vocabulary words in each document (L1 norm) or euclidean distance (L2 norm) of each document vector).



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False, norm='l2') # only uses tf weights (euclidean normalization used)
tf = tf_transformer.fit_transform(X)
tf.toarray() 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       ...,
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746]])
```


</div>
</div>
</div>



### 4. Bag of k-Skip, $n$-grams / w-shingling

We define a "gram" as a word / token / term in our text corpus after processing. In contrast to the Binary / Count Occurence BoW above, we alter the **vocabulary** in this approach by changing a vocabulary of single word tokens to a vocabulary of n-grams. This means that instead of having a bag of single words, AKA a unigram model, we can instead have a bag of contiguous word-pairs in the case of a bigram ($n=2$) model, bag of all contiguous sequences of 3 tokens in the case of a trigram ($n=2$) model... Furthermore, we can also include skips as well. For a $k=1$-skip bigram model, we'll connect every alternate token to form a word pair instead of consecutive word pairs. E.g. if the tokens were 'jeff', 'very', 'smart', 'jeff smart' will be our 1-skip bigram instead of 'jeff very' and 'very smart' for a 0-skip bigram (the normal bigram).

**vocabulary**: Set of all the unique $k$-skip $n$-grams in our text corpus after text processing.

**score measure of the presence**: Count of the number of occurence of each $k$-skip $n$-gram in the vocabulary / 1 or 0 if $k$-skip $n$-gram is present in document or not



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
X_2

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[1, 1, 4, ..., 2, 1, 1],
       [1, 1, 4, ..., 2, 1, 1],
       [1, 1, 4, ..., 2, 1, 1],
       ...,
       [1, 1, 4, ..., 2, 1, 1],
       [1, 1, 4, ..., 2, 1, 1],
       [1, 1, 4, ..., 2, 1, 1]], dtype=int64)
```


</div>
</div>
</div>



### 5. Word Hashing BoW

As our vocabulary size increases, the size of each document vector can explode. 

Furthermore, training something like a spam classifier on a fixed vocabulary size is also not great. E.g. if we train our spam classifier on a fixed vocabulary, spam like "*ii mayke are you th0usands of free for a \\$\$\$s surf1ing teh webz meeting early next week*" will not seem any different from "*are you free for a meeting early next week?*". Feature Hashing, AKA "the Hashing trick" can be used instead. Using a hash function, we can map our tokens / $k$-skip $n$-grams to index positions of a vector, incrementing the entry. This way, any new token can also be accounted for. 

| Dictionary | Hashing Trick
| :---: | :---: |
| No Collisions | Collisions
| Need to store dictionary for <br> learning and in production, <br> slow for large dictionaries  - $O$(log$\|D\|$) | No dictionary, <br> calculations are on the fly - $O$(1)
| Feature vector size = <br> Unique words and k-skip, n-grams count <br> $\therefore$ Variable memory footprint | Feature vector size = <br> Size of hashtable initialized <br> at the start <br> $\therefore$ Fixed memory footprint



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=10)
D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
f = h.transform(D)
f.toarray()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
       [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer()
X_3 = hv.transform(corpus).toarray()
X_3

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
```


</div>
</div>
</div>



## TF-IDF

$${w}_{x, y}={tf}_{x, y} \times log(\frac{N + 1}{{df}_{x} + 1}) + 1$$

- ${w}_{x, y}$ = TF-IDF weight of token $x$ within document $y$
- ${tf}_{x, y}$ = Frequency of token $x$ in document $y$
- ${df}_{x}$ = Number of documents containing $x$
- $N$ = Total number of documents (Number of rows in document-term matrix)
- $+1$ = Smoothing of idf term to prevent zero divisions

With tf-idf, if your vocabulary word / token / $k$-skip $n$-gram is in __high frequency__ in a __select few__ documents, it'll have a very high weight. If it is in __low frequency__ in __a lot__ of documents, it'll have a very low weightage.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer = CountVectorizer + TfidfTransformer
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda text: [WordNetLemmatizer().lemmatize(token.strip(), pos='v') for token in word_tokenize(text.lower()) if token not in stopwords.words("english") and token not in list(string.punctuation)])
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf.toarray() 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       ...,
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746],
       [0.03302291, 0.03302291, 0.03302291, ..., 0.19813746, 0.19813746,
        0.19813746]])
```


</div>
</div>
</div>



## Word Embeddings

The goal of word embeddings is to represent words in the text as vectors for processing by machines instead. There are several methods used to achieve this like the following:

1. Singular Value Decomposition (SVD)
2. Tomas Mikolov's Word2Vec
3. Stanford University's GloVe (Global Vectors for Word Representation)
4. FAIR (Facebook AI Research Lab)'s fastText
5. AllenNLP's Embeddings from Language Models (ELMo)



### Word2Vec

Implementations:
1. Continuous Bag of Words (CBOW)
2. Skip-Ngram

Training Methods:
1. Negative Sampling
2. Hierarchical Softmax



### GloVe



### fastText



### ELMo



---
# Modeling

Design a statistical or machine learning model, fit its parameters to training data, use an optimization procedure, and then use it to make predictions about unseen data.

Similarity Analysis:
- After creating vector representations of documents and words, we could use specific similarity metrics such as Cosine Similarity, Jaccard Coefficient (Intersection over Union), etc... in order to determine how similar two documents / words are.

Topic Modeling
- Latent Dirichlet Allocation (LDA)
- Latent Semantic Indexing (LSI)
- Non-negative Matrix Factorization (NMF)

Sentiment Analysis

Sequence models:
- Recurrent Neural Networks
    - Long Short Term Memory
    - Gated Recurrent Unit
- Transformers
    - BERT
    - GPT/GPT2
- Attention



---
## Resources:
- [POS Tagging and Hidden Markov Models](https://www.freecodecamp.org/news/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24/)
- [Intuition to POS Tagging with HMMs](https://www.youtube.com/watch?v=1O0qnNye6IQ&list=PLC0PzjY99Q_U5bba7gYJicCxIufrFmlTa&index=4)
- [Gentle Intro to BOW](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
- [BoW > Word Embeddings](https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016)
- [sklearn tfidf weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Hashing Trick](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f)
- [Shogun Hashing Trick](http://www.shogun-toolbox.org/static/notebook/current/HashedDocDotFeatures.html)
- [Dict VS Hashing Trick](https://www.coursera.org/lecture/machine-learning-applications-big-data/hashing-trick-GswXH)
- [sklearn Feature Hashing](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing)
- [Word2Vec Training Math explained](https://arxiv.org/pdf/1411.2738.pdf)
- [CBOW from scratch with Keras](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
- [Embedding Layers in Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
- [LDA for Topic Modeling](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)

