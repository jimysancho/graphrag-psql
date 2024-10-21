import hashlib


def calculate_hash(text, algorithm='sha256'):
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()
