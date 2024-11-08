from feature.text_processing import TextProcessor

def main():
    # Sample text for demonstration
    text = "London was the old capital of United Kingdom which has the most population in the world and I'll be go there someday with Hung's girlfriend who is in the U.S now."

    # Initialize the TextProcessor with the sample text
    processor = TextProcessor(text)

    # Tokenization outputs
    print("Whitespace Tokenization:", processor.whitespace_tokenize())
    print("WordPunct Tokenization:", processor.wordpunct_tokenize())
    print("Treebank Tokenization:", processor.treebank_tokenize())

    # Stemming outputs
    print("Stemmed (Whitespace):", processor.word_stem(mode="whitespace"))
    print("Stemmed (WordPunct):", processor.word_stem(mode="wordpunct"))
    print("Stemmed (Treebank):", processor.word_stem(mode="treebank"))

    # Lemmatization output
    print("Lemmatized Text with spacy:", processor.lemmatize_text_with_spacy())

if __name__ == "__main__":
    main()
