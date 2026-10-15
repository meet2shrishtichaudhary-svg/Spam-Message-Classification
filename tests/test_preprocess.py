"""Unit tests for text preprocessing module (src/preprocess.py)."""

from src.preprocess import clean_text, NEGATION_WORDS


def test_lowercase_normalization():
    """Verify text is converted to lowercase."""
    text = "HELLO World! This IS a TEST."
    result = clean_text(text)
    assert result == "hello world this is a test"


def test_whitespace_normalization():
    """Verify leading, trailing, and multiple internal spaces/newlines are collapsed."""
    text = "   Lots   of    \t\n  whitespace    here   "
    result = clean_text(text)
    assert result == "lots of whitespace here"


def test_empty_and_none_handling():
    """Verify empty strings and None return empty string safely without error."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_url_tokenization():
    """Verify URLs are standardized into httpaddr tokens as spam signals."""
    text = "Click here: https://reward-claim.com/win?id=123 or www.prizecenter.co.uk"
    result = clean_text(text)
    assert "httpaddr" in result
    assert "https" not in result


def test_currency_symbol_tokenization():
    """Verify currency symbols ($, £, €) are replaced with currencysymb token."""
    text = "Win £1000 or $500 or €250 prize today!"
    result = clean_text(text)
    assert "currencysymb" in result
    # Ensure there are 3 currency tokens
    assert result.count("currencysymb") == 3


def test_phone_number_tokenization():
    """Verify phone numbers and long digit strings are replaced with phonenbr token."""
    text = "Call 08452810075 or text 87121 to enter."
    result = clean_text(text)
    assert "phonenbr" in result


def test_punctuation_removal_without_merging():
    """Verify punctuation is stripped without merging adjacent words."""
    text = "hello,world!one/two;three:four-five."
    result = clean_text(text)
    assert "hello world one two three four five" == result


def test_negation_preservation_in_stopwords():
    """Verify important negation words ('not', 'no', 'never') are retained when stopwords are removed."""
    text = "I do not want this offer and no one should never click it"
    result = clean_text(text, remove_stopwords=True, preserve_negations=True)

    # Negations must be kept
    assert "not" in result.split()
    assert "no" in result.split()
    assert "never" in result.split()

    # Common non-negation stopwords like 'do', 'this', 'and', 'should', 'it' should be removed
    tokens = result.split()
    assert "do" not in tokens
    assert "and" not in tokens
    assert "this" not in tokens
