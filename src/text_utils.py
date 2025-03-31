# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

def get_nq_tokens(simplified_nq_example):
    """Returns list of blank-separated tokens from document_text."""
    if "document_text" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on an example with `document_text` field.")
    
    return simplified_nq_example["document_text"].split(" ")

def simplify_nq_example(nq_example):
    """Returns dictionary with blank-separated tokens in `document_text` field.

    This version handles the simplified NQ format, which does not contain
    `document_tokens`. Instead, it processes `document_text` directly.

    Args:
        nq_example: Dictionary containing original NQ example fields.

    Returns:
        Dictionary containing `document_text`, without `document_tokens` or `document_html`.
    """

    # Use `document_text` directly since `document_tokens` is missing
    text = nq_example["document_text"]
    
    # Tokenize the text to maintain compatibility with the expected format
    tokens = text.split()  # Tokenizing using whitespace

    def _remove_html_byte_offsets(span):
        """Removes byte offsets from annotations (start_byte, end_byte)."""
        if "start_byte" in span:
            del span["start_byte"]
        if "end_byte" in span:
            del span["end_byte"]
        return span

    def _clean_annotation(annotation):
        """Cleans up annotation structure to remove unnecessary fields."""
        annotation["long_answer"] = _remove_html_byte_offsets(annotation["long_answer"])
        annotation["short_answers"] = [
            _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation

    # Construct the simplified example
    simplified_nq_example = {
        "question_text": nq_example["question_text"],
        "example_id": nq_example["example_id"],
        "document_url": nq_example["document_url"],
        "document_text": text,  # Keep the cleaned document text
        "long_answer_candidates": [
            _remove_html_byte_offsets(c) for c in nq_example["long_answer_candidates"]
        ],
        "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
    }

    # Ensure token count is correct for validation
    if len(get_nq_tokens(simplified_nq_example)) != len(tokens):
        raise ValueError("Mismatch in token count after processing `document_text`.")

    return simplified_nq_example
