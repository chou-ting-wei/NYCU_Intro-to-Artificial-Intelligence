import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

stop_word_list = stopwords.words('english')

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    preprocessed_text = preprocessed_text.lower()
    preprocessed_text = preprocessed_text.replace('<br / >', ' ')
    preprocessed_text = ''.join([char for char in preprocessed_text if char.isalpha() or char.isspace()])
    
    words = nltk.word_tokenize(preprocessed_text)
    
    # PorterStemmer() SnowballStemmer() LancasterStemmer()
    # wordsPorter = [PorterStemmer().stem(word) for word in words]
    # wordsSnowball = [SnowballStemmer(language='english').stem(word) for word in words]
    # wordsLancaster = [LancasterStemmer().stem(word) for word in words]
    
    # print(f'Original:\n{text}\n'
    #       f'Remove stopwords/symbols:\n{" ".join(words)}\n'
    #       f'PorterStemmer:\n{" ".join(wordsPorter)}\n'
    #       f'SnowballStemmer:\n{" ".join(wordsSnowball)}\n'
    #       f'LancasterStemmer:\n{" ".join(wordsLancaster)}')
    
    stemmed_words = [SnowballStemmer(language='english').stem(word) for word in words]
    preprocessed_text = ' '.join(stemmed_words)
    # End your code

    return preprocessed_text