#!/usr/bin/env python3
"""
CallCenter Helper – Whisper + GPT
=================================
App de escritorio (Tkinter) que permite:
- Guardar y leer tu API Key de OpenAI desde la interfaz.
- Editar un archivo de "Conocimientos" (texto libre).
- Probar consultas rápidas (pegas texto de una llamada y te sugiere respuesta con GPT).

Requisitos:
-----------
Python 3.10+ en Windows.
Instala dependencias:
    pip install -r requirements.txt
"""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

try:
    from appdirs import user_data_dir
except Exception:
    user_data_dir = None

APP_NAME = "CallCenterHelper"
ORG = "GG"

# ------------------------
# Paths
# ------------------------
def app_dir() -> Path:
    if user_data_dir:
        base = Path(user_data_dir(APP_NAME, ORG))
    else:
        base = Path.home() / ".callcenter_helper"
    base.mkdir(parents=True, exist_ok=True)
    return base

CONFIG_PATH = app_dir() / "config.json"
API_KEY_PATH = app_dir() / "api.key.enc"
KNOWLEDGE_PATH = app_dir() / "conocimientos.md"
FERNET_KEY_PATH = app_dir() / ".fernet.key"

# ------------------------
# Config
# ------------------------
@dataclass
class AppConfig:
    chat_model: str = "gpt-4o-mini"
    transcribe_model: str = "whisper-1"
    use_encryption: bool = False

    @staticmethod
    def load() -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return AppConfig(**data)
            except Exception:
                pass
        cfg = AppConfig(use_encryption=bool(Fernet))
        cfg.save()
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

# ------------------------
# API Key
# ------------------------
def _ensure_fernet() -> Fernet | None:
    if not Fernet:
        return None
    if not FERNET_KEY_PATH.exists():
        key = Fernet.generate_key()
        FERNET_KEY_PATH.write_bytes(key)
    return Fernet(FERNET_KEY_PATH.read_bytes())

def save_api_key(api_key: str, use_encryption: bool) -> None:
    if not api_key:
        raise ValueError("API key vacía")
    if use_encryption and Fernet:
        f = _ensure_fernet()
        API_KEY_PATH.write_bytes(f.encrypt(api_key.encode("utf-8")))
    else:
        API_KEY_PATH.write_text(api_key, encoding="utf-8")

def load_api_key(use_encryption: bool) -> str | None:
    if not API_KEY_PATH.exists():
        return None
    try:
        if use_encryption and Fernet:
            f = _ensure_fernet()
            return f.decrypt(API_KEY_PATH.read_bytes()).decode("utf-8")
        return API_KEY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return None

# ------------------------
# Knowledge
# ------------------------
def ensure_knowledge_file():
    if not KNOWLEDGE_PATH.exists():
        KNOWLEDGE_PATH.write_text(
            "# Conocimientos del call center\n\n"
            "- Horario: Lun–Vie 9:00–18:00 (GMT-5)\n"
            "- Política de reembolsos: dentro de 30 días con boleta.\n",
            encoding="utf-8",
        )

def open_knowledge_in_editor(root: tk.Tk):
    ensure_knowledge_file()
    editor = tk.Toplevel(root)
    editor.title("Conocimientos")
    editor.geometry("800x600")

    txt = tk.Text(editor, wrap="word")
    txt.pack(fill="both", expand=True)
    txt.insert("1.0", KNOWLEDGE_PATH.read_text(encoding="utf-8"))

    def save_file():
        KNOWLEDGE_PATH.write_text(txt.get("1.0", "end-1c"), encoding="utf-8")
        messagebox.showinfo("Guardado", "Conocimientos guardados.")

    ttk.Button(editor, text="Guardar", command=save_file).pack(pady=6)

# ------------------------
# OpenAI
# ------------------------
def make_openai_client(api_key: str | None):
    if OpenAI is None:
        raise RuntimeError("Falta instalar openai: pip install openai")
    if not api_key:
        raise RuntimeError("Configura tu API key primero.")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def build_prompt(user_text: str) -> str:
    ensure_knowledge_file()
    kb = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return (
        "Eres un asistente de call center en inglés.\n"
        "### CONOCIMIENTOS\n" + kb +
        "\n### CALL SNIPPET\n" + user_text +
        "\n\n### YOUR REPLY\n"
    )

def ask_gpt(client: "OpenAI", model: str, prompt: str) -> str:
    try:
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=300)
        return resp.output[0].content[0].text.strip()
    except Exception as e:
        return f"[ERROR] {e}"

# ------------------------
# UI
# ------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CallCenter Helper – Whisper + GPT")
        self.geometry("900x650")
        self.cfg = AppConfig.load()
        self._build_widgets()
        ensure_knowledge_file()

    def _build_widgets(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # Config
        box = ttk.LabelFrame(root, text="Configuración")
        box.pack(fill="x", padx=12, pady=8)

        ttk.Label(box, text="OpenAI API Key:").grid(row=0, column=0, padx=6, pady=6)
        self.api_var = tk.StringVar()
        saved_key = load_api_key(self.cfg.use_encryption)
        if saved_key: self.api_var.set("*** guardada ***")
        self.api_entry = ttk.Entry(box, textvariable=self.api_var, width=50, show="*")
        self.api_entry.grid(row=0, column=1, padx=6, pady=6)

        self.model_var = tk.StringVar(value=self.cfg.chat_model)
        ttk.Combobox(box, textvariable=self.model_var,
                     values=("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")).grid(row=1, column=1, padx=6, pady=6)

        ttk.Button(box, text="Guardar", command=self._save_config).grid(row=0, column=2, padx=8)
        ttk.Button(box, text="Conocimientos…", command=lambda: open_knowledge_in_editor(self)).grid(row=1, column=2, padx=8)

        # Test
        test_box = ttk.LabelFrame(root, text="Prueba rápida")
        test_box.pack(fill="both", expand=True, padx=12, pady=8)

        self.input_txt = tk.Text(test_box, height=10, wrap="word")
        self.input_txt.pack(fill="both", expand=True, padx=8, pady=6)

        ttk.Button(test_box, text="Sugerir respuesta", command=self._on_suggest).pack(pady=6)

        self.output_txt = tk.Text(test_box, height=10, wrap="word")
        self.output_txt.pack(fill="both", expand=True, padx=8, pady=6)

    def _save_config(self):
        self.cfg.chat_model = self.model_var.get()
        val = self.api_var.get().strip()
        if val and val != "*** guardada ***":
            save_api_key(val, self.cfg.use_encryption)
            self.api_var.set("*** guardada ***")
        self.cfg.save()
        messagebox.showinfo("Guardado", "Configuración guardada.")

    def _on_suggest(self):
        text = self.input_txt.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Vacío", "Escribe o pega un texto.")
            return
        api_key = load_api_key(self.cfg.use_encryption)
        client = make_openai_client(api_key)
        prompt = build_prompt(text)

        def task():
            reply = ask_gpt(client, self.cfg.chat_model, prompt)
            self.output_txt.delete("1.0", "end")
            self.output_txt.insert("1.0", reply)

        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    app = App()
    app.mainloop()

