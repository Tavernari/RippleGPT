import os
import pickle
import torch
from src.model import RippleGPT
from src.config import RippleConfig

# -----------------------------------------------------------------------------
out_dir = 'out'
num_samples = 1 # Quantas variações de cada prompt
max_new_tokens = 200 # Curto para testar várias coisas rápido
temperature = 0.8 # Criatividade equilibrada
top_k = 200 
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(1337)
    
    # 1. Carrega o Melhor Modelo (Checkpoint com menor Loss)
    ckpt_path = os.path.join(out_dir, 'ckpt_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        print("⚠️ Aviso: 'ckpt_best.pt' não encontrado, usando o último 'ckpt.pt'")
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Configuração e Modelo
    gptconf = RippleConfig(**checkpoint['model_args'])
    model = RippleGPT(gptconf)
    
    # Limpeza de chaves do state_dict (caso venha de compile)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    
    # 2. Carrega o Vocabulário (Meta)
    meta_path = os.path.join('data', 'meta.pkl')
    if os.path.exists(meta_path):
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        # Safe encode: uses '?' (if available) or ignores unknown chars.
        # Fallback to 0 if '?' not in vocab (unlikely for english text but possible)
        unknown_token = stoi.get('?', 0) 
        encode = lambda s: [stoi.get(c, unknown_token) for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("❌ ERRO: meta.pkl não encontrado! Rode prepare_data.py primeiro.")
        return

    # 3. OS TESTES DE INTELIGÊNCIA (Gatilhos Fortes)
    test_cases = [
        # A. Teste de Código (Python)
        # TRUQUE: Adicionar um comentário ou docstring ajuda a firmar o contexto
        {
            "domain": "🐍 PYTHON CODING",
            "prompt": "# Function to calculate factorial\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return"
        },
        
        # B. Teste de Matemática (Algebra)
        # TRUQUE: Dar um exemplo antes (Few-shot prompting)
        {
            "domain": "🧮 MATH LOGIC",
            "prompt": "Q: Solve 2x = 10\nA: x = 5\n\nQ: Solve -5k + 5 = -10\nA:"
        },
        
        # C. Teste de TinyStories
        {
            "domain": "📖 TINY STORY",
            "prompt": "Once upon a time, there was a little frog. The frog liked to jump. One day,"
        },
        
        # D. Teste de Literatura
        {
            "domain": "⚔️ LITERATURE",
            "prompt": "The General looked at the map and shouted,"
        }
    ]

    # 4. Loop de Geração
    print("\n" + "="*40)
    print(f"🤖 RIPPLE GPT: MULTI-DOMAIN TEST")
    print("="*40)

    with torch.no_grad():
        for case in test_cases:
            prompt = case["prompt"]
            domain = case["domain"]
            
            print(f"\n[{domain}] Prompt: {prompt.strip()}")
            print("-" * 20)
            
            # Encode
            start_ids = encode(prompt)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

            # Generate
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            # Decode e Print
            generated_text = decode(y[0].tolist())
            
            # Destaca o que foi gerado vs o que era prompt
            new_content = generated_text[len(prompt):]
            print(f"{prompt}\033[94m{new_content}\033[0m") # Azul para o gerado (no terminal)
            print("-" * 40)

if __name__ == '__main__':
    main()