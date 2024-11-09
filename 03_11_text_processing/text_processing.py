from nltk import wordpunct_tokenize
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List
import spacy

class TextProcessor:
    """
    A class for processing text with various tokenization and stemming methods.

    Attributes:
        text (str): The input text to be processed.
    """

    def __init__(self):
        """
        Initializes the TextProcessor with the provided text.

        Args:
            text (str): The input text to be processed.
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.wordpunct_tokenizer = WordPunctTokenizer()
        self.treebank_tokenizer = TreebankWordTokenizer()

    def whitespace_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text using whitespace as a delimiter.

        Args:
            text (str): The input text to be processed.

        Returns:
            List[str]: A list of tokens.
        """
        return self.whitespace_tokenizer.tokenize(text)

    def wordpunct_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text using word punctuations.

        Args:
            text (str): The input text to be processed.

        Returns:
            List[str]: A list of tokens.
        """
        return self.wordpunct_tokenizer.tokenize(text)

    def treebank_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text using the Treebank tokenizer.

        Args:
            text (str): The input text to be processed.

        Returns:
            List[str]: A list of tokens.
        """
        return self.treebank_tokenizer.tokenize(text)

    def word_stem(self, text: str, mode: str = "whitespace") -> List[str]:
        """
        Applies stemming to each token in the text based on the specified tokenization mode.

        Args:
            mode (str): The tokenization mode to use for stemming.
                        Options are "whitespace", "wordpunct", and "treebank".
                        Default is "whitespace".
            text (str): The input text to be processed.

        Returns:
            List[str]: A list of stemmed tokens.
        """
        if mode == "whitespace":
            tokens = self.whitespace_tokenize(text)
        elif mode == "wordpunct":
            tokens = self.wordpunct_tokenize(text)
        elif mode == "treebank":
            tokens = self.treebank_tokenize(text)
        else:
            raise ValueError("Invalid mode. Choose 'whitespace', 'wordpunct', or 'treebank'.")

        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_text_with_spacy(self, text: str) -> List[str]:
        """
        Lemmatizes each token in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            List[str]: A list of lemmatized tokens.
        """
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
