#!/usr/bin/env python3
"""
CallCenter Helper – Qt + QDarkStyle (Whisper + GPT)
---------------------------------------------------
- Modelo por defecto: gpt-3.5-turbo (bajo costo).
- UI: PySide6 + QDarkStyle (modo oscuro).
- Settings: guarda/lee API Key (cifrado local opcional).
- Knowledge: editor con 2 pestañas ("Conocimientos" y "Rol") sobre un único conocimientos.md.
- Compose: pegas texto de llamada y genera respuesta en inglés usando el KB.
- Logs: pestaña para ver eventos y errores.
- Build: PyInstaller --onedir --windowed (sin consola). Incluye conocimientos.md como recurso.

Ruta de datos (Windows):
  %APPDATA%/CallCenterHelper/
    config.json
    api.key.enc
    conocimientos.md

Primer arranque:
- Si no existe conocimientos.md en APPDATA, se copia desde el recurso empacado (o desde el repo local).
"""

from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from appdirs import user_data_dir
from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle

# --- Dependencias opcionales ---
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_NAME = "CallCenterHelper"
ORG = "GG"

# ------------------------
# Paths & resources
# ------------------------
APP_DIR = Path(user_data_dir(APP_NAME, ORG))
APP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = APP_DIR / "config.json"
API_KEY_PATH = APP_DIR / "api.key.enc"
FERNET_KEY_PATH = APP_DIR / ".fernet.key"
KNOWLEDGE_PATH = APP_DIR / "conocimientos.md"

def resource_path(rel: str) -> Path:
    """
    Devuelve la ruta de un recurso empacado (--add-data) o del directorio local.
    En build onedir, PyInstaller expone sys._MEIPASS.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    return Path(__file__).parent / rel

def ensure_knowledge_file():
    """Copia conocimientos.md a APPDATA si no existe."""
    if KNOWLEDGE_PATH.exists():
        return
    src = resource_path("conocimientos.md")
    if src.exists():
        KNOWLEDGE_PATH.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # fallback mínimo
        KNOWLEDGE_PATH.write_text("# Conocimientos del call center\n\n# Rol\n\n", encoding="utf-8")

def read_knowledge() -> str:
    ensure_knowledge_file()
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")

def write_knowledge(text: str):
    KNOWLEDGE_PATH.write_text(text, encoding="utf-8")

# ------------------------
# Knowledge split/merge (robusto)
# ------------------------
# Captura secciones por encabezado H1 exacto, preservando TODO el contenido.
H1_CONO = "# Conocimientos del call center"
H1_ROL  = "# Rol"

def split_sections(md: str) -> tuple[str, str]:
    """
    Separa el markdown en dos secciones por H1 exactos:
    '# Conocimientos del call center' y '# Rol'.
    - Si falta alguno, lo crea vacío.
    - No pierde contenido: se basa en posiciones de encabezados.

    Devuelve (cono_md, rol_md) INCLUYENDO los encabezados.
    """
    text = md.replace("\r\n", "\n")
    # Encuentra posiciones de encabezados con regex multilínea estricto
    # ^# Título$
    matches = list(re.finditer(r'(?m)^(# .+)$', text))
    # Mapa de título -> índice en texto
    idx = {m.group(1).strip(): m.start() for m in matches}

    # Helper para extraer desde un header hasta el siguiente o fin
    def extract(start_title: str) -> str | None:
        if start_title not in idx:
            return None
        start = idx[start_title]
        # siguiente header después de 'start'
        end = None
        for m in matches:
            if m.start() > start:
                end = m.start()
                break
        chunk = text[start:end].strip() + "\n"
        return chunk

    cono = extract(H1_CONO)
    rol  = extract(H1_ROL)

    if cono is None and rol is None:
        # No hay encabezados: todo va a conocimientos, rol vacío
        return (f"{H1_CONO}\n\n{text.strip()}\n", f"{H1_ROL}\n\n")

    if cono is None:
        cono = f"{H1_CONO}\n\n"
    if rol is None:
        rol = f"{H1_ROL}\n\n"
    return (cono, rol)

def merge_sections(cono_md: str, rol_md: str) -> str:
    cono_md = cono_md.strip()
    rol_md = rol_md.strip()
    if not cono_md.lower().startswith(H1_CONO.lower()):
        cono_md = f"{H1_CONO}\n\n{cono_md}"
    if not rol_md.lower().startswith(H1_ROL.lower()):
        rol_md = f"{H1_ROL}\n\n{rol_md}"
    return f"{cono_md}\n\n{rol_md}\n"

# ------------------------
# Config & API key
# ------------------------
@dataclass
class AppConfig:
    chat_model: str = "gpt-3.5-turbo"   # DEFAULT ECONÓMICO
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

    def save(self):
        CONFIG_PATH.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

def _ensure_fernet() -> Fernet | None:
    if not Fernet:
        return None
    if not FERNET_KEY_PATH.exists():
        FERNET_KEY_PATH.write_bytes(Fernet.generate_key())
    return Fernet(FERNET_KEY_PATH.read_bytes())

def save_api_key(api_key: str, use_encryption: bool):
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
def make_client(api_key: str | None):
    if OpenAI is None:
        raise RuntimeError("Falta instalar 'openai' (pip install openai).")
    if not api_key:
        raise RuntimeError("Configura tu API key en Settings.")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def build_prompt(user_text: str) -> str:
    kb = read_knowledge()
    return (
        "You are a call center assistant. Use the following company knowledge. "
        "Reply in English with a professional, concise, empathetic and actionable tone. "
        "If information is missing, ask only for the minimum needed.\n\n"
        "### KNOWLEDGE\n" + kb +
        "\n### CALL SNIPPET\n" + user_text +
        "\n\n### YOUR REPLY\n"
    )

def ask_model(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=350,
        temperature=0.3,
    )
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        return getattr(resp, "output_text", "(no text)").strip()

# ------------------------
# UI – Pages
# ------------------------
class SettingsPage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        form = QtWidgets.QFormLayout(self)

        self.api_edit = QtWidgets.QLineEdit()
        self.api_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        if load_api_key(self.cfg.use_encryption):
            self.api_edit.setPlaceholderText("*** saved ***")

        self.show_chk = QtWidgets.QCheckBox("Show")
        self.show_chk.toggled.connect(
            lambda on: self.api_edit.setEchoMode(QtWidgets.QLineEdit.Normal if on else QtWidgets.QLineEdit.Password)
        )

        self.model_cb = QtWidgets.QComboBox()
        # DEFAULT: gpt-3.5-turbo en primer lugar
        self.model_cb.addItems(["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
        self.model_cb.setCurrentText(self.cfg.chat_model)

        self.stt_cb = QtWidgets.QComboBox()
        self.stt_cb.addItems(["whisper-1", "gpt-4o-transcribe"])
        self.stt_cb.setCurrentText(self.cfg.transcribe_model)

        self.encrypt_chk = QtWidgets.QCheckBox("Encrypt API key (recommended)")
        self.encrypt_chk.setChecked(bool(self.cfg.use_encryption and (Fernet is not None)))

        save_btn = QtWidgets.QPushButton("Save settings")
        save_btn.clicked.connect(self._save)

        form.addRow("OpenAI API Key:", self.api_edit)
        form.addRow("", self.show_chk)
        form.addRow("Chat model:", self.model_cb)
        form.addRow("Transcribe model:", self.stt_cb)
        form.addRow("", self.encrypt_chk)
        form.addRow("", save_btn)

    def _save(self):
        self.cfg.chat_model = self.model_cb.currentText()
        self.cfg.transcribe_model = self.stt_cb.currentText()
        self.cfg.use_encryption = bool(self.encrypt_chk.isChecked() and (Fernet is not None))
        self.cfg.save()

        val = self.api_edit.text().strip()
        if val and val != "*** saved ***":
            try:
                save_api_key(val, self.cfg.use_encryption)
                self.api_edit.clear()
                self.api_edit.setPlaceholderText("*** saved ***")
                self.log_signal.emit("API key saved.")
                QtWidgets.QMessageBox.information(self, "Saved", "Settings saved and API key stored.")
            except Exception as e:
                self.log_signal.emit(f"[ERROR] Saving API key: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Cannot save API key: {e}")
                return
        else:
            QtWidgets.QMessageBox.information(self, "Saved", "Settings saved.")
            self.log_signal.emit("Settings saved.")

class KnowledgePage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)
    knowledge_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        ensure_knowledge_file()

        v = QtWidgets.QVBoxLayout(self)

        # Pestañas izquierda/derecha: usamos QTabWidget (primera pestaña por defecto)
        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs)

        self.cono_edit = QtWidgets.QPlainTextEdit()
        self.rol_edit  = QtWidgets.QPlainTextEdit()

        # Cargamos contenido con parser robusto (no se pierde nada)
        full = read_knowledge()
        cono_md, rol_md = split_sections(full)
        self.cono_edit.setPlainText(cono_md)
        self.rol_edit.setPlainText(rol_md)

        self.tabs.addTab(self.cono_edit, "Conocimientos")
        self.tabs.addTab(self.rol_edit, "Rol")
        self.tabs.setCurrentIndex(0)  # por defecto la primera

        # Botonera
        h = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        export_btn = QtWidgets.QPushButton("Export .md")
        import_btn = QtWidgets.QPushButton("Import .md")
        h.addWidget(save_btn); h.addWidget(export_btn); h.addWidget(import_btn); h.addStretch(1)
        v.addLayout(h)

        save_btn.clicked.connect(self._save)
        export_btn.clicked.connect(self._export)
        import_btn.clicked.connect(self._import)

    def _save(self):
        merged = merge_sections(self.cono_edit.toPlainText(), self.rol_edit.toPlainText())
        write_knowledge(merged)
        self.knowledge_changed.emit()
        self.log_signal.emit("Knowledge saved.")
        QtWidgets.QMessageBox.information(self, "Saved", "Knowledge saved successfully.")

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export as...", str(Path.home()/ "conocimientos.md"), "Markdown (*.md)")
        if not path:
            return
        try:
            merged = merge_sections(self.cono_edit.toPlainText(), self.rol_edit.toPlainText())
            Path(path).write_text(merged, encoding="utf-8")
            self.log_signal.emit(f"Knowledge exported to {path}")
            QtWidgets.QMessageBox.information(self, "Exported", f"Wrote file to\n{path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Export: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot export: {e}")

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import .md", str(Path.home()), "Markdown (*.md);;Text (*.txt);;All (*.*)")
        if not path:
            return
        try:
            data = Path(path).read_text(encoding="utf-8", errors="ignore")
            cono_md, rol_md = split_sections(data)
            self.cono_edit.setPlainText(cono_md)
            self.rol_edit.setPlainText(rol_md)
            self.log_signal.emit(f"Knowledge imported from {path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Import: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot import: {e}")

class ComposePage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        v = QtWidgets.QVBoxLayout(self)

        # Input
        in_group = QtWidgets.QGroupBox("Paste a call snippet or summary")
        in_layout = QtWidgets.QVBoxLayout(in_group)
        self.input = QtWidgets.QPlainTextEdit()
        in_layout.addWidget(self.input)
        v.addWidget(in_group, 2)

        # Buttons
        hb = QtWidgets.QHBoxLayout()
        self.btn_suggest = QtWidgets.QPushButton("Suggest reply (EN)")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        hb.addWidget(self.btn_suggest); hb.addWidget(self.btn_clear); hb.addStretch(1)
        v.addLayout(hb)

        # Output
        out_group = QtWidgets.QGroupBox("Suggested reply")
        out_layout = QtWidgets.QVBoxLayout(out_group)
        self.output = QtWidgets.QPlainTextEdit(); self.output.setReadOnly(True)
        out_layout.addWidget(self.output)
        v.addWidget(out_group, 2)

        self.btn_suggest.clicked.connect(self._on_suggest)
        self.btn_clear.clicked.connect(lambda: self.input.setPlainText(""))

    def _on_suggest(self):
        text = self.input.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "Empty", "Paste some text first.")
            return

        api_key = load_api_key(self.cfg.use_encryption)
        try:
            client = make_client(api_key)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        prompt = build_prompt(text)
        self.output.setPlainText("Generating…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            reply = ask_model(client, self.cfg.chat_model, prompt)
            self.output.setPlainText(reply)
            self.log_signal.emit("Suggestion generated.")
        except Exception as e:
            self.output.setPlainText(f"[ERROR] {e}")
            self.log_signal.emit(f"[ERROR] {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

class LogsPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        self.box = QtWidgets.QPlainTextEdit()
        self.box.setReadOnly(True)
        v.addWidget(self.box)

    @QtCore.Slot(str)
    def append(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.box.appendPlainText(f"[{ts}] {msg}")
        self.box.verticalScrollBar().setValue(self.box.verticalScrollBar().maximum())

# ------------------------
# Main Window
# ------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CallCenter Helper – Whisper + GPT")
        self.resize(1100, 750)

        self.cfg = AppConfig.load()
        ensure_knowledge_file()

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Pages
        self.settings = SettingsPage(self.cfg)
        self.knowledge = KnowledgePage()
        self.compose  = ComposePage(self.cfg)
        self.logs     = LogsPage()

        tabs.addTab(self.settings, "Settings")
        tabs.addTab(self.knowledge, "Knowledge")
        tabs.addTab(self.compose, "Compose")
        tabs.addTab(self.logs, "Logs")

        # Wiring logs
        self.settings.log_signal.connect(self.logs.append)
        self.knowledge.log_signal.connect(self.logs.append)
        self.compose.log_signal.connect(self.logs.append)

        # Status bar: knowledge path + last modified
        self.status = self.statusBar()
        self._update_status()

        # Knowledge changed → update status
        self.knowledge.knowledge_changed.connect(self._update_status)

    @QtCore.Slot()
    def _update_status(self):
        try:
            mtime = datetime.fromtimestamp(KNOWLEDGE_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.status.showMessage(f"Knowledge: {KNOWLEDGE_PATH} | Updated: {mtime}")
        except Exception:
            self.status.showMessage(f"Knowledge: {KNOWLEDGE_PATH}")

def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))

    # Icono (opcional): busca assets/icon.ico si lo agregas en el futuro
    icon_path = resource_path('assets/icon.ico')
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())


