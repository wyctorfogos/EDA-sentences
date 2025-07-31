import re

def mascarar_termos(texto: str) -> str:
    termos = ['MEL', 'BCC', 'SCC', 'ACK', 'SEK', 'NEV', 'CANCER', 'NO-CANCER']

    # Cria um padrão regex com os termos, respeitando bordas de palavra
    padrao = r'\b(' + '|'.join(map(re.escape, termos)) + r')\b'

    # Substitui todos os termos encontrados por '******'
    return re.sub(padrao, '******', texto)