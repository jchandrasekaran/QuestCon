"""
This file contains the implementation of various utility functions used throughout the project.
"""

import string

def clean (s):
	exclude = set(string.punctuation)
	s = ''.join(ch for ch in s if ch not in exclude)
	return s.lower().strip()

def key_max_val_dict(d1):
	v = list(d1.values())
	k = list(d1.keys())
	return k[v.index(max(v))]
