"""
Created on Sun Jun 15 20:46:40 2025

@author: recruta42
"""

###############################################################################
# Libraries
###############################################################################

import unicodedata
import regex
from ftfy import fix_encoding

from sacremoses import MosesPunctNormalizer

###############################################################################
# Unicode Normalization
###############################################################################

def normalize_unicode(text, form="NFC"):
    """
    Normalize Unicode characters to NFC/NFKC form.
    """
    return unicodedata.normalize(form, text)

def fix_text_encoding(text):
    """
    Fixes mojibake, broken accents, and line breaks.
    """
    return fix_encoding(text, normalization='NFKC', fix_line_breaks=True)

def normalize_sentence_punctuation(text, lang):
    """
    Normalizes punctuation in a language-aware way based on Moses tokenizer rules
    """
    return MosesPunctNormalizer(lang).normalize(text)

###############################################################################
# Control and Invisible Characters
###############################################################################

def remove_control_characters(text):
    """
    Removes ASCII control characters (U+0000–001F and U+007F).
    """
    return ''.join(ch for ch in text if ch >= ' ' and ch != '\x7f')

def remove_zero_width_spaces(text):
    """
    Removes zero-width spaces from the text, cleaning hidden formatting characters.

    Example:
    >>> remove_zero_width_spaces("Hello\u200bWorld")
    'HelloWorld'
    """
    pattern = regex.compile(r'[\u200B\u200C\u200D\uFEFF]')
    
    return pattern.sub('', text)

###############################################################################
# Punctuation and Spacing
###############################################################################

def normalize_punctuation_spacing(text):
    """
    Ensures there is a space after punctuation followed directly by a letter, improving readability.

    Example:
    >>> normalize_punctuation_spacing("Hello,world!Today is sunny.")
    'Hello, world! Today is sunny.'
    
    """
    pattern = regex.compile(r"(\w+|\"|')([!,:;?])([a-zA-Z]\w)")
    
    return pattern.sub(r'\1\2 \3', text)

def remove_space_before_punctuation(text):
    """
    Removes unnecessary spaces before punctuation marks to clean up text formatting.

    Example:
    >>> remove_space_before_punctuation("Hello , world !")
    'Hello, world!'
    """
    return regex.sub(r"(\s)([!',:;?.])", r'\2', text)

def add_space_around_opening_quotes(text):
    """
    Adds space before opening quotes if directly attached to a word.
    """
    return regex.sub(r"(\w)([«“'\"])", r"\1 \2", text)

def add_space_after_closing_quotes(text):
    """
    Adds space after closing quotes if attached to next word.
    """
    return regex.sub(r"([”’'\"])(\w)", r"\1 \2", text)

###############################################################################
# HTML Tag Removal
###############################################################################

def remove_html_tags(text):
    """
    Replaces <p> with newline and removes all other HTML tags.
    """
    text = regex.sub(r'(\s*)(<p>)+', '\n', text)
    text = regex.sub(r' *(<.*?> ?)+ *', ' ', text)
    return text

def replace_urls(text, placeholder=""):
    """
    Replaces all URLs in the provided text with a placeholder string.
    Supports URLs with or without parentheses and markdown formats.

    Example:
    >>> replace_urls("Visit http://example.com and (https://domain.org)")
    'Visit <URL> and <URL>'
    """
    url_pattern = regex.compile(
        r"""(?xi)
        \b                                  # Start at a word boundary
        (                                   # Begin capture group for full URL
            (https?:\/\/)?                  # Optional http/https scheme
            (www\.)?                        # Optional www
            [\p{L}0-9\-._~%]+               # Domain name
            \.                              # Dot
            [\p{L}]{2,}                     # Top-level domain (e.g., com, org)
            (\/[\p{L}0-9\-._~%!$&'()*+,;=:@/]*)?  # Optional path
            (\?[^\s]*)?                     # Optional query string
        )
        """
    )

    # Replace detected URLs with the placeholder
    return url_pattern.sub(placeholder, text)

def replace_email(text, placeholder=" <EMAIL> "):
    
    """
    Replaces email addresses in the text with ' <EMAIL> ', which can be useful for anonymizing 
    email addresses or simplifying text for further processing.

    Example:
    >>> replace_email("Contact me at example@domain.com for more info.")
    'Contact me at  <EMAIL>  for more info.'
    """
    
    emails_pattern = regex.compile(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-z0-9-.]+',
        flags=regex.IGNORECASE
    )
        
    text, subs = emails_pattern.subn(placeholder, text)
    
    return text

def replace_hashtags_and_mentions(text, placeholder=' <HASHTAG> '):
    
    """
    Replaces hashtags and mentions in the text with ' <HASHTAG> ', useful for reducing noise in text 
    analysis or anonymizing data in content meant for public sharing.

    Example:
    >>> remove_hashtags("Hello @user, check out #Python and #coding.")
    'Hello  <HASHTAG> , check out  <HASHTAG>  and  <HASHTAG> .'
    """
    
    hashtags_pattern = regex.compile('(@[A-Za-z0-9_]+)|(#[\w_]+)')
    
    text, subs = hashtags_pattern.subn(placeholder, text)
    
    return text

def remove_citations(text):
    
    """
    Removes citation markers from the text. Citation markers are typically numerical and enclosed in brackets,
    such as [1], [2], [99], etc. This is useful for cleaning up academic or technical documents before processing.

    Example:
    >>> text = "According to Smith [1], the findings are inconclusive. See also [2], [3]."
    >>> remove_citations(text)
    'According to Smith , the findings are inconclusive. See also .'
    """
    
    # Regex to match numerical citations in brackets
    remove_citations_pattern = regex.compile(r'[\s,;:]*\[\d{1,3}\]')

    # Remove the citations from the text
    text = remove_citations_pattern.sub('', text)

    return text

###############################################################################
# Sentence Segmentation
###############################################################################

def segment_sentences_basic(text):
    """
    Segments sentences by inserting newlines at missing punctuation between sentences.

    Example:
    >>> segment_sentences_basic("This is great.Stop there.")
    'This is great.\nStop there.'
    """
    
    pattern = regex.compile(r"(\s)(\p{Ll}+)([.!?:]*)(\p{Lu})(\p{Ll}+)([\s.,;:?!])")
    
    return pattern.sub(r"\1\2\3\n\4\5\6", text)

def segment_sentences_with_quotes(text):
    """
    Inserts newlines in text where new sentences within quotes follow punctuation.

    Example:
    >>> segment_sentences_with_quotes("He said:'Watch out!'Next, he moved.")
    "He said:'Watch out!'\nNext, he moved."
    """
    
    pattern = regex.compile(r"(\s)(\p{Ll}+)([.!?:]+)('|\")(\p{Lu})(\p{Ll}+)([\s.,;:?!])")
    
    return pattern.sub(r"\1\2\3\n\4\5\6\7", text)


def clean_html(text):
    
    # Correct html
    text = remove_html_tags(text)
    text = replace_urls(text)
    text = replace_email(text)
    text = replace_hashtags_and_mentions(text)
    text = remove_citations(text)
    
    return text
    
def clean_text(text):
    
    # Normalize text
    text = normalize_unicode(text)
    text = fix_text_encoding(text)
    
    # Remove control chars
    text = remove_control_characters(text)
    text = remove_zero_width_spaces(text)
    
    # Correct spacing
    text = normalize_punctuation_spacing(text)
    text = remove_space_before_punctuation(text)
    text = add_space_around_opening_quotes(text)
    text = add_space_after_closing_quotes(text)
    
    # Sentence segmentation
    text = segment_sentences_basic(text)
    text = segment_sentences_with_quotes(text)
    
    return text
    