import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox

# --- 1. НАСТРОЙКИ (ОПТИМИЗИРОВАНО ПОД ~200M ПАРАМЕТРОВ И 8GB VRAM) ---
batch_size = 8
grad_accum_steps = 4
block_size = 256
max_iters = 20000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
save_interval = 1000

# ПАРАМЕТРЫ АРХИТЕКТУРЫ ДЛЯ ~200М:
n_embd = 896
n_head = 14
n_layer = 12
dropout = 0.1

# Глобальные флаги для управления из GUI
stop_training = False
force_save = False


# --- 2. АРХИТЕКТУРА ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MyPersonalAI(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- 3. ФУНКЦИЯ ОБУЧЕНИЯ (В ОТДЕЛЬНОМ ПОТОКЕ) ---
def train_logic(status_label):
    global stop_training, force_save

    enc = tiktoken.get_encoding("gpt2")
    if not os.path.exists('input.txt'):
        messagebox.showerror("Ошибка", "Файл input.txt не найден!")
        return

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        ds = train_data if split == 'train' else val_data
        ix = torch.randint(len(ds) - block_size, (batch_size,))
        x = torch.stack([ds[i:i + block_size] for i in ix])
        y = torch.stack([ds[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = MyPersonalAI(enc.n_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Модель готова: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M параметров.")

    start_time = time.time()
    for iter in range(max_iters):
        if stop_training:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            (loss / grad_accum_steps).backward()
        optimizer.step()

        if iter % 100 == 0:
            dt = time.time() - start_time
            log_msg = f"Шаг {iter}: Loss {loss.item():.4f}"
            print(log_msg)
            status_label.config(text=log_msg)
            start_time = time.time()

        # Логика чекпоинтов
        if (iter % save_interval == 0 and iter > 0) or force_save:
            torch.save(model.state_dict(), 'my_ai_model_checkpoint.pth')
            print(f">>> Чекпоинт сохранен на шаге {iter}")
            force_save = False

    torch.save(model.state_dict(), 'my_ai_model_final.pth')
    status_label.config(text="Обучение завершено!")
    messagebox.showinfo("Готово", "Модель сохранена в my_ai_model_final.pth")


# --- 4. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (TKINTER) ---
def start_gui():
    global stop_training, force_save

    root = tk.Tk()
    root.title("Управление обучением ИИ")
    root.geometry("400x250")

    lbl_info = tk.Label(root, text="Нажмите кнопку для начала обучения", pady=10)
    lbl_info.pack()

    lbl_status = tk.Label(root, text="Статус: Ожидание", fg="blue", font=("Arial", 10, "bold"))
    lbl_status.pack(pady=5)

    def on_start():
        btn_start.config(state=tk.DISABLED)
        t = threading.Thread(target=train_logic, args=(lbl_status,), daemon=True)
        t.start()

    def on_save():
        global force_save
        force_save = True
        messagebox.showinfo("Чекпоинт", "Запрос на сохранение отправлен. Ожидайте следующего шага.")

    def on_stop():
        global stop_training
        if messagebox.askyesno("Выход", "Остановить обучение и сохранить финальную модель?"):
            stop_training = True

    btn_start = tk.Button(root, text="ЗАПУСТИТЬ ОБУЧЕНИЕ", command=on_start, bg="green", fg="white", width=25)
    btn_start.pack(pady=5)

    btn_save = tk.Button(root, text="СОХРАНИТЬ ЧЕКПОИНТ СЕЙЧАС", command=on_save, bg="orange", width=25)
    btn_save.pack(pady=5)

    btn_stop = tk.Button(root, text="ОСТАНОВИТЬ И ВЫЙТИ", command=on_stop, bg="red", fg="white", width=25)
    btn_stop.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    start_gui()