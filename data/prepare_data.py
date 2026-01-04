import os
import requests
import numpy as np
import pickle
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
DATA_CACHE_DIR = os.path.dirname(__file__)
SEPARATOR = "\n\n<|END_OF_DOCUMENT|>\n\n" # O modelo aprende que aqui muda o assunto
TARGET_SIZE_PER_DOMAIN = 4_000_000 # 4MB por domínio = ~16MB total (Ótimo para o Mac)

def get_python_data(target_chars):
    print("🔹 Baixando Código Python (The Stack)...")
    # Streaming: Baixa apenas o necessário
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    
    text_accum = []
    current_len = 0
    
    for sample in tqdm(dataset, desc="Coletando Python", total=target_chars//100):
        code = sample['content']
        # Filtro básico: Pula arquivos muito pequenos ou gigantescos
        if 100 < len(code) < 10000:
            text_accum.append(code)
            current_len += len(code)
            if current_len >= target_chars: break
            
    return SEPARATOR.join(text_accum)

def get_math_data(target_chars):
    print("🔹 Baixando Matemática (DeepMind Algebra)...")
    # Algebra linear exige que a rede não "perca o fio da meada"
    dataset = load_dataset("math_dataset", "algebra__linear_1d", split="train", streaming=True)
    
    text_accum = []
    current_len = 0
    
    for sample in tqdm(dataset, desc="Coletando Math"):
        # Formata como pergunta e resposta
        qa = f"Q: {sample['question'].strip()}\nA: {sample['answer'].strip()}"
        text_accum.append(qa)
        current_len += len(qa)
        if current_len >= target_chars: break
            
    return SEPARATOR.join(text_accum)

def get_tinystories_data(target_chars):
    print("🔹 Baixando TinyStories (Raciocínio Narrativo)...")
    # TinyStories é famoso por ensinar IAs pequenas a falar coerentemente
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    text_accum = []
    current_len = 0
    
    for sample in tqdm(dataset, desc="Coletando Stories"):
        story = sample['text']
        text_accum.append(story)
        current_len += len(story)
        if current_len >= target_chars: break
            
    return SEPARATOR.join(text_accum)

def get_classic_lit_data():
    print("🔹 Baixando Literatura (War and Peace)...")
    url = 'https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt'
    try:
        text = requests.get(url).text
        return text
    except:
        print("Erro ao baixar livro. Pulando...")
        return ""

def prepare_super_dataset():
    print(f"--- 🧠 PREPARING MULTI-DOMAIN DATASET ({TARGET_SIZE_PER_DOMAIN/1e6 * 4} MB target) ---")
    
    # 1. Coleta dos 4 domínios
    # Usamos try/except para garantir que o script não pare se a internet falhar em um
    parts = []
    
    try: parts.append(get_python_data(TARGET_SIZE_PER_DOMAIN))
    except Exception as e: print(f"Erro no Python: {e}")
        
    try: parts.append(get_math_data(TARGET_SIZE_PER_DOMAIN))
    except Exception as e: print(f"Erro no Math: {e}")
        
    try: parts.append(get_tinystories_data(TARGET_SIZE_PER_DOMAIN))
    except Exception as e: print(f"Erro no TinyStories: {e}")
        
    try: parts.append(get_classic_lit_data())
    except Exception as e: print(f"Erro no Livro: {e}")

    # 2. Mistura e Limpeza
    print("\nCombinando tudo...")
    full_text = SEPARATOR.join(parts)
    
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"Total Caracteres: {len(full_text):,}")
    print(f"Tamanho em Disco: {len(full_text)/1024/1024:.2f} MB")

    # 3. Tokenizer (Character Level)
    # Como temos matemática e código, o vocabulário vai aumentar um pouco (símbolos)
    print("Construindo vocabulário...")
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)
    print(f"Vocab Size: {vocab_size}")
    print(f"Chars (Amostra): {''.join(chars[30:80])}...")

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }
    with open(os.path.join(DATA_CACHE_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    def encode(s):
        return [stoi[c] for c in s]

    # 4. Split e Salvamento
    print("Codificando e salvando (Isso pode levar um minuto)...")
    n = len(full_text)
    split_idx = int(n * 0.9) # 90% Treino, 10% Validação
    
    train_data = full_text[:split_idx]
    val_data = full_text[split_idx:]
    
    # uint16 suporta vocabulário até 65535 (suficiente para char-level)
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    
    train_ids.tofile(os.path.join(DATA_CACHE_DIR, 'train.bin'))
    val_ids.tofile(os.path.join(DATA_CACHE_DIR, 'val.bin'))
    
    print(f"✅ PRONTO! Arquivos salvos em {DATA_CACHE_DIR}")
    print("Agora rode: python train.py")

if __name__ == '__main__':
    prepare_super_dataset()