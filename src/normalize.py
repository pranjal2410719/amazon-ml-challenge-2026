"""
Text normalization utilities for business names and addresses.

Design principle: normalization must be GENERIC (regex/rule based on token
patterns), never tied to specific country values, so it generalizes to
unseen countries (e.g. France in the test set) without modification.
"""
import re

# Legal-suffix / abbreviation equivalence classes.
# Keys map to a single canonical token so "Corp" and "Corporation" collapse
# to the same normalized form. This list is generic across countries; it is
# NOT filtered or branched by country value.
_SUFFIX_MAP = {
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "company": "co", "co": "co",
    "llc": "llc", "llp": "llp",
    "sarl": "sarl", "sas": "sas", "sa": "sa",  # French legal forms
    "gmbh": "gmbh",
    "and": "&",
}

_ADDR_ABBR_MAP = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg",
    "floor": "fl", "fl": "fl",
    "near": "near", "nr": "near",
}

_PUNCT_RE = re.compile(r"[^\w\s&]")
_WS_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    # Handles both Python None and pandas/numpy NaN (a float) -- str(nan)
    # would otherwise silently become the literal word "nan" in the
    # normalized text, polluting similarity features. ~3.3% of this
    # dataset's business_address values are NaN, so this matters at scale.
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN check without pandas dependency
        return ""
    text = str(text).lower().strip()
    if text == "nan":  # belt-and-suspenders in case a literal "nan" string slips through
        return ""
    text = text.replace("&", " and ")  # unify before token mapping re-collapses it
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse legal-suffix synonyms."""
    text = _basic_clean(name)
    tokens = [_SUFFIX_MAP.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def normalize_address(addr: str) -> str:
    """Lowercase, strip punctuation, collapse common address abbreviations."""
    text = _basic_clean(addr)
    tokens = [_ADDR_ABBR_MAP.get(tok, tok) for tok in text.split()]
    return " ".join(tokens)


def name_tokens(name: str) -> set:
    return {_strip_leading_zero(t) for t in normalize_name(name).split()}


def address_tokens(addr: str) -> set:
    return {_strip_leading_zero(t) for t in normalize_address(addr).split()}


def _strip_leading_zero(token: str) -> str:
    """'053' -> '53', but leaves non-numeric tokens untouched. Fixes
    real cases like 'Flat No.-53' vs 'FLAT NO.-053' failing to match as
    the same token."""
    if token.isdigit():
        stripped = token.lstrip("0")
        return stripped if stripped else "0"
    return token


def sorted_token_key(tokens: set) -> str:
    """Alphabetically-sorted, space-joined tokens -- a blocking sort key
    that is INVARIANT to word order. Real data shows frequent name and
    address component transposition (e.g. "Gangaya Techinfra Limited" vs
    "Techinfra Gangaya Limited"; address components fully reordered across
    sources) -- sorting by raw normalized string order misses these because
    the strings start with different letters and never land near each
    other in a sorted-neighborhood window. Sorting the TOKENS first makes
    both variants produce the identical key."""
    return " ".join(sorted(tokens))


def char_ngrams(text: str, n: int = 3) -> set:
    """Character n-gram shingles, used for MinHash / LSH blocking.
    Works on any script/language since it needs no tokenizer."""
    text = _basic_clean(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
