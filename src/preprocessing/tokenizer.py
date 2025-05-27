import re

def simple_tokenize(text):
    # 1. Kleinbuchstaben (Kommt mir so sinnvoll vor??)
    text = text.lower()
    # 2. Satzzeichen entfernen (außer z. B. Emojis falls erwünscht)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # 3. Tokenisierung (Split nach Leerzeichen)
    tokens = text.split()
    return ['<s>'] + tokens + ['</s>']

#print(simple_tokenize("Hallo, ich heisse Lisa.!!"))