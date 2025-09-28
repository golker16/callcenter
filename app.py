#!/usr/bin/env python3
"""
CallCenter Helper – Whisper + GPT (Tkinter)
-------------------------------------------
- Modelo por defecto: GPT-3.5 (bajo costo).
- Guarda/lee API Key desde la interfaz (con cifrado local opcional).
- Editor de Conocimientos con 2 secciones: "Conocimientos" y "Rol".
- Pestaña "Logs" para ver eventos y errores.
- Sin consola en Windows (build con PyInstaller --windowed).
- En el primer arranque, si no existe el archivo en APPDATA, copia
  'conocimientos.md' desde los recursos del ejecutable (o del repo si corres local).

Rutas:
  %APPDATA%/CallCenterHelper/config.json
  %APPDATA%/CallCenterHelper/api.key.enc
  %APPDATA%/CallCenterHelper/conocimientos.md

Requisitos:
  Python 3.10+
  pip install -r requirements.txt
"""

from __future__ import annotations
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import scrolledtext

# --- Dependencias externas ---
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
# Rutas y utilidades
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
FERNET_KEY_PATH = app_dir() / ".fernet.key"
KNOWLEDGE_PATH = app_dir() / "conocimientos.md"

def get_resource_path(rel_path: str) -> Path:
    """
    Obtiene la ruta a un recurso empacado con PyInstaller (--add-data)
    o al archivo en el mismo directorio del script si se ejecuta local.
    """
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path:
        return Path(base_path) / rel_path
    return Path(__file__).parent / rel_path

# ------------------------
# Conocimientos (archivo)
# ------------------------
def ensure_knowledge_file():
    """
    Si no existe en APPDATA, intenta copiar desde:
      - recursos empacados (PyInstaller)
      - o el archivo 'conocimientos.md' del repo (junto a app.py)
    """
    if KNOWLEDGE_PATH.exists():
        return
    # Prioridad: recurso empacado con PyInstaller
    src = get_resource_path("conocimientos.md")
    if src.exists():
        KNOWLEDGE_PATH.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # Fallback mínimo si no está el archivo
        KNOWLEDGE_PATH.write_text("# Conocimientos del call center\n\n# Rol\n\n", encoding="utf-8")

def read_knowledge() -> str:
    ensure_knowledge_file()
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")

def write_knowledge(text: str):
    KNOWLEDGE_PATH.write_text(text, encoding="utf-8")

def split_knowledge_sections(text: str) -> tuple[str, str]:
    """
    Separa por '# Conocimientos del call center' y '# Rol'.
    Devuelve (conocimientos_md, rol_md), preservando encabezados.
    """
    t = text.replace("\r\n", "\n")
    cono_h = "# Conocimientos del call center"
    rol_h = "# Rol"
    cpos = t.find(cono_h)
    rpos = t.find(rol_h)

    if cpos == -1 and rpos == -1:
        return (f"{cono_h}\n\n{t.strip()}\n", f"{rol_h}\n\n")
    # Conocimientos
    if cpos == -1:
        cono_md = f"{cono_h}\n\n"
    else:
        if rpos == -1:
            cono_md = t[cpos:].strip() + "\n"
        else:
            cono_md = (t[cpos:rpos] if cpos < rpos else t[cpos:]).strip() + "\n"
    # Rol
    if rpos == -1:
        rol_md = f"{rol_h}\n\n"
    else:
        rol_md = t[rpos:].strip() + "\n"

    return (cono_md, rol_md)

def merge_knowledge_sections(cono_md: str, rol_md: str) -> str:
    cono_md = cono_md.strip()
    rol_md = rol_md.strip()
    if not cono_md.lower().startswith("# conocimientos del call center"):
        cono_md = "# Conocimientos del call center\n\n" + cono_md
    if not rol_md.lower().startswith("# rol"):
        rol_md = "# Rol\n\n" + rol_md
    return f"{cono_md}\n\n{rol_md}\n"

# ------------------------
# Config
# ------------------------
@dataclass
class AppConfig:
    chat_model: str = "gpt-3.5-turbo"  # default barato
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
# API Key (cifrado opcional)
# ------------------------
def _ensure_fernet() -> Fernet | None:
    if not Fernet:
        return None
    if not FERNET_KEY_PATH.exists():
        FERNET_KEY_PATH.write_bytes(Fernet.generate_key())
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
# OpenAI
# ------------------------
def make_openai_client(api_key: str | None, log):
    if OpenAI is None:
        raise RuntimeError("Falta instalar 'openai': pip install openai")
    if not api_key:
        raise RuntimeError("Configura tu API key en Configuración.")
    os.environ["OPENAI_API_KEY"] = api_key
    log("OpenAI client inicializado.")
    return OpenAI()

def build_prompt(user_text: str) -> str:
    kb = read_knowledge()
    return (
        "Eres un asistente de call center en inglés. Usa el conocimiento a continuación. "
        "Responde en inglés, con tono profesional, breve, empático y accionable. "
        "Si falta información, pide los datos mínimos.\n\n"
        "### KNOWLEDGE\n" + kb +
        "\n### CALL SNIPPET\n" + user_text +
        "\n\n### YOUR REPLY\n"
    )

def ask_gpt(client: "OpenAI", model: str, prompt: str, log) -> str:
    try:
        log(f"Llamando modelo: {model}")
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=350, temperature=0.3)
        try:
            text = resp.output[0].content[0].text.strip()
        except Exception:
            text = getattr(resp, "output_text", "(sin texto)").strip()
        log("Respuesta recibida OK.")
        return text
    except Exception as e:
        log(f"[ERROR OpenAI] {e}")
        return f"[ERROR] {e}"

# ------------------------
# UI
# ------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CallCenter Helper – Whisper + GPT")
        self.geometry("1000x720")
        self.minsize(900, 640)

        self.cfg = AppConfig.load()
        ensure_knowledge_file()

        # Notebook principal
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._build_tab_compose()     # Tab de trabajo (config + entrada + salida)
        self._build_tab_logs()        # Logs
        self._build_statusbar()       # Barra de estado
        self._update_knowledge_status()

    # ---------- LOGS ----------
    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.logs_text.configure(state="normal")
        self.logs_text.insert("end", line)
        self.logs_text.see("end")
        self.logs_text.configure(state="disabled")

    # ---------- STATUS ----------
    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Listo")
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=4)

    def _update_knowledge_status(self):
        try:
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(KNOWLEDGE_PATH.stat().st_mtime))
            self.status_var.set(f"Conocimientos: {KNOWLEDGE_PATH} | Actualizado: {mtime}")
        except Exception:
            self.status_var.set(f"Conocimientos: {KNOWLEDGE_PATH}")

    # ---------- TAB: COMPOSE ----------
    def _build_tab_compose(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Compose")

        # Panel superior: Config
        cfg_box = ttk.LabelFrame(tab, text="Configuración")
        cfg_box.pack(fill="x", padx=12, pady=10)

        ttk.Label(cfg_box, text="OpenAI API Key:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.api_var = tk.StringVar()
        saved_key = load_api_key(self.cfg.use_encryption)
        if saved_key:
            self.api_var.set("*** guardada ***")
        api_entry = ttk.Entry(cfg_box, textvariable=self.api_var, width=50, show="*")
        api_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(cfg_box, text="Modelo:").grid(row=1, column=0, sticky="w", padx=6)
        self.model_var = tk.StringVar(value=self.cfg.chat_model)
        model_cb = ttk.Combobox(cfg_box, textvariable=self.model_var, state="readonly",
                                values=("gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"))
        model_cb.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        self.encrypt_var = tk.BooleanVar(value=self.cfg.use_encryption and (Fernet is not None))
        ttk.Checkbutton(cfg_box, text="Cifrar API Key (recomendado)", variable=self.encrypt_var).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Button(cfg_box, text="Guardar config", command=self._save_config).grid(row=0, column=2, padx=8, pady=6)
        ttk.Button(cfg_box, text="Editar Conocimientos…", command=self._open_knowledge_editor).grid(row=1, column=2, padx=8, pady=6)

        for i in range(3):
            cfg_box.grid_columnconfigure(i, weight=1)

        # Panel medio: Entrada de llamada
        input_box = ttk.LabelFrame(tab, text="Pega aquí un fragmento/resumen de la llamada")
        input_box.pack(fill="both", expand=True, padx=12, pady=6)
        self.input_txt = scrolledtext.ScrolledText(input_box, height=10, wrap="word")
        self.input_txt.pack(fill="both", expand=True, padx=8, pady=6)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", padx=12)
        ttk.Button(btn_frame, text="Sugerir respuesta (EN)", command=self._on_suggest).pack(side="left", padx=4, pady=8)
        ttk.Button(btn_frame, text="Limpiar", command=lambda: self.input_txt.delete("1.0", "end")).pack(side="left", padx=4)

        # Panel inferior: Salida
        output_box = ttk.LabelFrame(tab, text="Respuesta sugerida")
        output_box.pack(fill="both", expand=True, padx=12, pady=6)
        self.output_txt = scrolledtext.ScrolledText(output_box, height=10, wrap="word")
        self.output_txt.pack(fill="both", expand=True, padx=8, pady=6)

    # ---------- TAB: LOGS ----------
    def _build_tab_logs(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Logs")
        self.logs_text = scrolledtext.ScrolledText(tab, state="disabled", wrap="word")
        self.logs_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log("Aplicación iniciada.")
        self.log(f"Carpeta de datos: {app_dir()}")

    # ---------- Acciones ----------
    def _save_config(self):
        self.cfg.chat_model = self.model_var.get()
        self.cfg.use_encryption = bool(self.encrypt_var.get() and (Fernet is not None))

        val = self.api_var.get().strip()
        if val and val != "*** guardada ***":
            try:
                save_api_key(val, self.cfg.use_encryption)
                self.api_var.set("*** guardada ***")
                self.log("API key guardada.")
            except Exception as e:
                self.log(f"[ERROR] Al guardar API key: {e}")
                messagebox.showerror("Error", f"No se pudo guardar la API key: {e}")

        self.cfg.save()
        self.log("Configuración guardada.")
        messagebox.showinfo("Listo", "Configuración guardada.")

    def _on_suggest(self):
        user_text = self.input_txt.get("1.0", "end-1c").strip()
        if not user_text:
            messagebox.showwarning("Vacío", "Escribe o pega un texto.")
            return

        api_key = load_api_key(self.cfg.use_encryption)
        model = self.model_var.get()
        prompt = build_prompt(user_text)

        self.output_txt.delete("1.0", "end")
        self.output_txt.insert("1.0", "Generando sugerencia...\n")
        self.log("Generando sugerencia...")

        def task():
            try:
                client = make_openai_client(api_key, self.log)
                reply = ask_gpt(client, model, prompt, self.log)
            except Exception as e:
                reply = f"[ERROR] {e}"
                self.log(f"[ERROR general] {e}")
            self.output_txt.delete("1.0", "end")
            self.output_txt.insert("1.0", reply)

        threading.Thread(target=task, daemon=True).start()

    # ---------- Editor de Conocimientos (2 pestañas) ----------
    def _open_knowledge_editor(self):
        ensure_knowledge_file()
        full = read_knowledge()
        cono_md, rol_md = split_knowledge_sections(full)

        win = tk.Toplevel(self)
        win.title("Conocimientos")
        win.geometry("900x650")

        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)

        # Pestaña Conocimientos
        tab1 = ttk.Frame(nb); nb.add(tab1, text="Conocimientos")
        txt_cono = scrolledtext.ScrolledText(tab1, wrap="word")
        txt_cono.pack(fill="both", expand=True, padx=8, pady=8)
        txt_cono.insert("1.0", cono_md)

        # Pestaña Rol
        tab2 = ttk.Frame(nb); nb.add(tab2, text="Rol")
        txt_rol = scrolledtext.ScrolledText(tab2, wrap="word")
        txt_rol.pack(fill="both", expand=True, padx=8, pady=8)
        txt_rol.insert("1.0", rol_md)

        # Botonera
        bf = ttk.Frame(win); bf.pack(fill="x", padx=8, pady=8)

        def do_save():
            merged = merge_knowledge_sections(
                txt_cono.get("1.0", "end-1c"),
                txt_rol.get("1.0", "end-1c")
            )
            write_knowledge(merged)
            self._update_knowledge_status()
            self.log("Conocimientos guardados.")
            messagebox.showinfo("Guardado", "Conocimientos guardados correctamente.")

        ttk.Button(bf, text="Guardar", command=do_save).pack(side="left")

        def do_import():
            path = filedialog.askopenfilename(
                title="Importar .md",
                filetypes=[("Markdown", "*.md"), ("Texto", "*.txt"), ("Todos", "*.*")]
            )
            if path:
                try:
                    data = Path(path).read_text(encoding="utf-8", errors="ignore")
                    c, r = split_knowledge_sections(data)
                    txt_cono.delete("1.0", "end"); txt_cono.insert("1.0", c)
                    txt_rol.delete("1.0", "end"); txt_rol.insert("1.0", r)
                    self.log(f"Importado conocimientos desde {path}.")
                except Exception as e:
                    self.log(f"[ERROR] Importando: {e}")
                    messagebox.showerror("Error", f"No se pudo importar: {e}")

        ttk.Button(bf, text="Importar .md", command=do_import).pack(side="left", padx=6)

        def do_export():
            path = filedialog.asksaveasfilename(
                title="Exportar .md",
                defaultextension=".md",
                filetypes=[("Markdown", "*.md"), ("Texto", "*.txt")]
            )
            if path:
                try:
                    merged = merge_knowledge_sections(
                        txt_cono.get("1.0", "end-1c"),
                        txt_rol.get("1.0", "end-1c")
                    )
                    Path(path).write_text(merged, encoding="utf-8")
                    self.log(f"Exportado conocimientos a {path}.")
                    messagebox.showinfo("Exportado", f"Archivo exportado a\n{path}")
                except Exception as e:
                    self.log(f"[ERROR] Exportando: {e}")
                    messagebox.showerror("Error", f"No se pudo exportar: {e}")

        ttk.Button(bf, text="Exportar .md", command=do_export).pack(side="left", padx=6)

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()

