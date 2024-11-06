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

    def __init__(self, text: str):
        """
        Initializes the TextProcessor with the provided text.

        Args:
            text (str): The input text to be processed.
        """
        self.text = text
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")

    def whitespace_tokenize(self) -> List[str]:
        """
        Tokenizes the text using whitespace as a delimiter.

        Returns:
            List[str]: A list of tokens.
        """
        return WhitespaceTokenizer().tokenize(self.text)

    def wordpunct_tokenize(self) -> List[str]:
        """
        Tokenizes the text using word punctuations.

        Returns:
            List[str]: A list of tokens.
        """
        return WordPunctTokenizer().tokenize(self.text)

    def treebank_tokenize(self) -> List[str]:
        """
        Tokenizes the text using the Treebank tokenizer.

        Returns:
            List[str]: A list of tokens.
        """
        return TreebankWordTokenizer().tokenize(self.text)

    def word_stem(self, mode: str = "whitespace") -> List[str]:
        """
        Applies stemming to each token in the text based on the specified tokenization mode.

        Args:
            mode (str): The tokenization mode to use for stemming.
                        Options are "whitespace", "wordpunct", and "treebank".
                        Default is "whitespace".

        Returns:
            List[str]: A list of stemmed tokens.
        """
        if mode == "whitespace":
            tokens = self.whitespace_tokenize()
        elif mode == "wordpunct":
            tokens = self.wordpunct_tokenize()
        elif mode == "treebank":
            tokens = self.treebank_tokenize()
        else:
            raise ValueError("Invalid mode. Choose 'whitespace', 'wordpunct', or 'treebank'.")

        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_text_with_spacy(self) -> List[str]:
        """
        Lemmatizes each token in the text.

        Returns:
            List[str]: A list of lemmatized tokens.
        """
        doc = self.nlp(self.text)
        return [token.lemma_ for token in doc]
