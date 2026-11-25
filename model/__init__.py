"""Sindhi line recognizer and the language layers that sit outside it.

The network is model.crnn.CRNN. Character n-grams and the word list live in
model.ngram and model.lexicon and are not part of the CTC loss.
"""
