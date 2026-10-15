"""Text preprocessing module for Spam Message Classification.

Provides deterministic, leakage-free text normalization for SMS messages:
- Lowercase normalization
- URL, currency, and special token standardization
- Whitespace normalization
- Safe punctuation handling that preserves spam indicators (e.g., currency, numbers)
- Optional stopword filtering with negation preservation ('not', 'no', 'never')
- Self-contained (does not mandate external NLTK corpus downloads to run)
"""

import re
import string
from typing import Optional, Set

# Essential negation words that change sentence semantics and must NOT be removed as stopwords
NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "neither",
    "never",
    "none",
    "cannot",
    "cant",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "won't",
    "wont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "ain't",
    "aint",
    "couldn't",
    "couldnt",
    "shouldn't",
    "shouldnt",
    "wouldn't",
    "wouldnt",
}

# Self-contained standard English stopwords for lightweight zero-download reliability
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him",
    "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in",
    "into", "is", "it", "its", "itself", "lets", "me", "more", "most", "my",
    "myself", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes",
    "should", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs",
    "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll",
    "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "wed", "well", "were", "weve", "what", "whats",
    "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom",
    "why", "whys", "with", "would", "you", "youd", "youll", "youre", "youve",
    "your", "yours", "yourself", "yourselves"
} - NEGATION_WORDS


def clean_text(
    text: Optional[str],
    remove_stopwords: bool = False,
    preserve_negations: bool = True
) -> str:
    """Clean and normalize an SMS text message.

    Parameters
    ----------
    text : str or None
        Raw input string.
    remove_stopwords : bool, default=False
        Whether to filter out common English stopwords. Default is False to
        preserve message conversational structure, which often distinguishes
        informal ham from formulaic spam.
    preserve_negations : bool, default=True
        When remove_stopwords=True, if True, explicitly retains negation tokens
        such as 'not', 'no', and 'never'.

    Returns
    -------
    str
        Normalized text string ready for TF-IDF feature extraction.
    """
    if text is None:
        return ""

    # Ensure text is a string
    cleaned = str(text)

    # 1. Lowercase normalization
    cleaned = cleaned.lower()

    # 2. Standardize URLs to 'httpaddr' token (a strong spam feature)
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    cleaned = url_pattern.sub(" httpaddr ", cleaned)

    # 3. Standardize email addresses
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    cleaned = email_pattern.sub(" emailaddr ", cleaned)

    # 4. Standardize currency symbols ($, £, €) to 'currencysymb' token
    # Promotional spam frequently includes £1000, $500, etc.
    currency_pattern = re.compile(r"[\$£€¥₹]")
    cleaned = currency_pattern.sub(" currencysymb ", cleaned)

    # 5. Standardize phone numbers / long numeric strings (e.g. 5+ digits)
    # Often used in spam for claim hotlines or premium rate numbers
    phone_pattern = re.compile(r"\b\d{5,12}\b")
    cleaned = phone_pattern.sub(" phonenbr ", cleaned)

    # 6. Remove remaining punctuation while keeping alphanumeric tokens
    # Replace punctuation characters with a space so words don't accidentally merge
    punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    cleaned = punctuation_pattern.sub(" ", cleaned)

    # 7. Normalize whitespace
    tokens = cleaned.split()

    # 8. Optional stopword removal
    if remove_stopwords:
        if preserve_negations:
            stop_set = DEFAULT_STOPWORDS - NEGATION_WORDS
        else:
            stop_set = DEFAULT_STOPWORDS
        tokens = [token for token in tokens if token not in stop_set]

    return " ".join(tokens)


if __name__ == "__main__":
    # Test cases demonstrating spam indicator preservation
    sample_ham = "Hey, are you free tomorrow? Let's catch up at the cafe."
    sample_spam = "CONGRATS! You won £1000 cash! Call 08452810075 or visit http://win.com now!"
    sample_negation = "I do not want to go there, no way."

    print("Sample Ham:")
    print("  Original:", sample_ham)
    print("  Cleaned :", clean_text(sample_ham))
    print("\nSample Spam:")
    print("  Original:", sample_spam)
    print("  Cleaned :", clean_text(sample_spam))
    print("\nSample with Negations (with remove_stopwords=True):")
    print("  Original:", sample_negation)
    print("  Cleaned :", clean_text(sample_negation, remove_stopwords=True))
