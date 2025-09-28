#!/usr/bin/env python3
"""
CallCenter Helper – Qt + QDarkStyle (Whisper + GPT)
---------------------------------------------------
- Modelo por defecto: gpt-3.5-turbo (bajo costo).
- UI: PySide6 + QDarkStyle (oscuro).
- Settings: guarda/lee API Key (cifrado local opcional).
- Knowledge: editor con 2 pestañas ("Conocimientos" y "Rol") sobre un único conocimientos.md.
- Compose (ES): dos modos -> "Escuchar PC (beta)" (transcribe) y "Escribir" (genera respuesta).
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
import tempfile
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

# Audio (Escuchar PC / micrófono)
try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

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
        # Fallback mínimo
        KNOWLEDGE_PATH.write_text("# Conocimientos del call center\n\n# Rol\n\n", encoding="utf-8")

def read_knowledge() -> str:
    ensure_knowledge_file()
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")

def write_knowledge(text: str):
    KNOWLEDGE_PATH.write_text(text, encoding="utf-8")

# ------------------------
# Knowledge split/merge (robusto)
# ------------------------
# Acepta variaciones de espacios y mayúsculas/minúsculas.
H1_CONO = "# Conocimientos del call center"
H1_ROL  = "# Rol"

def _find_h1_positions(md: str) -> list[tuple[int, str]]:
    """Devuelve [(pos, título-normalizado)] de todos los H1."""
    text = md.replace("\r\n", "\n")
    out = []
    for m in re.finditer(r'(?im)^\s*#\s*(.+?)\s*$', text):
        title = m.group(1).strip()
        out.append((m.start(), "# " + title))
    return out

def split_sections(md: str) -> tuple[str, str]:
    """
    Separa el markdown en dos secciones por H1:
    '# Conocimientos del call center' y '# Rol'.
    Preserva TODO el contenido. Si falta alguno, se crea vacío.
    """
    text = md.replace("\r\n", "\n")
    headers = _find_h1_positions(text)
    # mapa por título normalizado (lower, sin dobles espacios)
    def norm(s: str) -> str:
        return re.sub(r'\s+', ' ', s.strip().lower())

    idx = {norm(h): pos for pos, h in headers}
    n_cono = norm(H1_CONO)
    n_rol  = norm(H1_ROL)

    def extract(start_pos: int) -> str:
        # tramo desde start_pos hasta el siguiente H1 o fin
        h_starts = sorted([p for p, _ in headers])
        next_starts = [p for p in h_starts if p > start_pos]
        end = next_starts[0] if next_starts else len(text)
        return text[start_pos:end].strip() + "\n"

    # Buscar posiciones (aceptando variantes de espacios y case)
    pos_cono = None
    pos_rol = None
    for pos, h in headers:
        if norm(h) == n_cono:
            pos_cono = pos
        if norm(h) == n_rol:
            pos_rol = pos

    if pos_cono is None and pos_rol is None:
        # No hay H1 reconocidos → todo a conocimientos
        return (f"{H1_CONO}\n\n{text.strip()}\n", f"{H1_ROL}\n\n")

    cono = extract(pos_cono) if pos_cono is not None else f"{H1_CONO}\n\n"
    rol  = extract(pos_rol)  if pos_rol  is not None else f"{H1_ROL}\n\n"
    return (cono, rol)

def merge_sections(cono_md: str, rol_md: str) -> str:
    cono_md = cono_md.strip()
    rol_md  = rol_md.strip()
    if not re.match(r'(?is)^\s*#\s*conocimientos del call center', cono_md):
        cono_md = f"{H1_CONO}\n\n{cono_md}"
    if not re.match(r'(?is)^\s*#\s*rol', rol_md):
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

def transcribe_file(client: OpenAI, model: str, file_path: Path) -> str:
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(model=model, file=f)
    # SDK suele exponer .text
    text = getattr(tr, "text", None)
    if not text:
        # por si cambia el esquema
        text = json.dumps(tr.__dict__, ensure_ascii=False)
    return text

# ------------------------
# Audio helpers (Escuchar PC / Mic)
# ------------------------
def record_system_or_mic_wav(seconds: int, samplerate: int = 48000, channels: int = 2, logs: callable | None = None) -> Path:
    """
    Intenta capturar LOOPBACK (audio del sistema) con WASAPI.
    Si falla, cae a micrófono. Requiere sounddevice + soundfile.
    Devuelve ruta a WAV temporal.
    """
    if sd is None or sf is None:
        raise RuntimeError("Falta instalar 'sounddevice' y 'soundfile'.")

    tmp = Path(tempfile.gettempdir()) / f"cchelper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    try:
        # WASAPI loopback (Windows)
        wasapi = sd.WasapiSettings(loopback=True)
        if logs: logs("Intentando capturar audio del sistema (WASAPI loopback)…")
        frames = int(seconds * samplerate)
        buf = []

        def callback(indata, frames_, time_, status):
            if status and logs:
                logs(f"sd status: {status}")
            buf.append(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32",
                            extra_settings=wasapi, callback=callback):
            sd.sleep(int(seconds * 1000))

        import numpy as np
        data = np.concatenate(buf, axis=0) if buf else np.zeros((1, channels), dtype="float32")
        sf.write(tmp.as_posix(), data, samplerate)
        if logs: logs(f"Audio capturado en {tmp}")
        return tmp

    except Exception as e:
        if logs: logs(f"Loopback falló ({e}). Usando micrófono…")
        # Micrófono
        import numpy as np
        recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        sf.write(tmp.as_posix(), recording, samplerate)
        if logs: logs(f"Audio de micrófono guardado en {tmp}")
        return tmp

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
            self.api_edit.setPlaceholderText("*** guardada ***")

        self.show_chk = QtWidgets.QCheckBox("Mostrar")
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

        self.encrypt_chk = QtWidgets.QCheckBox("Cifrar API key (recomendado)")
        self.encrypt_chk.setChecked(bool(self.cfg.use_encryption and (Fernet is not None)))

        save_btn = QtWidgets.QPushButton("Guardar")
        save_btn.clicked.connect(self._save)

        form.addRow("OpenAI API Key:", self.api_edit)
        form.addRow("", self.show_chk)
        form.addRow("Modelo de chat:", self.model_cb)
        form.addRow("Modelo de transcripción:", self.stt_cb)
        form.addRow("", self.encrypt_chk)
        form.addRow("", save_btn)

    def _save(self):
        self.cfg.chat_model = self.model_cb.currentText()
        self.cfg.transcribe_model = self.stt_cb.currentText()
        self.cfg.use_encryption = bool(self.encrypt_chk.isChecked() and (Fernet is not None))
        self.cfg.save()

        val = self.api_edit.text().strip()
        if val and val != "*** guardada ***":
            try:
                save_api_key(val, self.cfg.use_encryption)
                self.api_edit.clear()
                self.api_edit.setPlaceholderText("*** guardada ***")
                self.log_signal.emit("API key guardada.")
                QtWidgets.QMessageBox.information(self, "Guardado", "Configuración guardada y API key almacenada.")
            except Exception as e:
                self.log_signal.emit(f"[ERROR] Guardando API key: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo guardar la API key: {e}")
                return
        else:
            QtWidgets.QMessageBox.information(self, "Guardado", "Configuración guardada.")
            self.log_signal.emit("Configuración guardada.")

class KnowledgePage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)
    knowledge_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        ensure_knowledge_file()

        v = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs)

        self.cono_edit = QtWidgets.QPlainTextEdit()
        self.rol_edit  = QtWidgets.QPlainTextEdit()

        full = read_knowledge()
        cono_md, rol_md = split_sections(full)
        self.cono_edit.setPlainText(cono_md)
        self.rol_edit.setPlainText(rol_md)

        self.tabs.addTab(self.cono_edit, "Conocimientos")
        self.tabs.addTab(self.rol_edit, "Rol")
        self.tabs.setCurrentIndex(0)  # por defecto la primera

        h = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Guardar")
        export_btn = QtWidgets.QPushButton("Exportar .md")
        import_btn = QtWidgets.QPushButton("Importar .md")
        h.addWidget(save_btn); h.addWidget(export_btn); h.addWidget(import_btn); h.addStretch(1)
        v.addLayout(h)

        save_btn.clicked.connect(self._save)
        export_btn.clicked.connect(self._export)
        import_btn.clicked.connect(self._import)

    def _save(self):
        merged = merge_sections(self.cono_edit.toPlainText(), self.rol_edit.toPlainText())
        write_knowledge(merged)
        self.knowledge_changed.emit()
        self.log_signal.emit("Conocimientos guardados.")
        QtWidgets.QMessageBox.information(self, "Guardado", "Conocimientos guardados correctamente.")

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar como...", str(Path.home()/ "conocimientos.md"), "Markdown (*.md)")
        if not path:
            return
        try:
            merged = merge_sections(self.cono_edit.toPlainText(), self.rol_edit.toPlainText())
            Path(path).write_text(merged, encoding="utf-8")
            self.log_signal.emit(f"Exportado a {path}")
            QtWidgets.QMessageBox.information(self, "Exportado", f"Archivo escrito en:\n{path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Exportar: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo exportar: {e}")

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Importar .md", str(Path.home()), "Markdown (*.md);;Text (*.txt);;All (*.*)")
        if not path:
            return
        try:
            data = Path(path).read_text(encoding="utf-8", errors="ignore")
            cono_md, rol_md = split_sections(data)
            self.cono_edit.setPlainText(cono_md)
            self.rol_edit.setPlainText(rol_md)
            self.log_signal.emit(f"Importado: {path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Importar: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo importar: {e}")

class ComposePage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)
    paste_transcript = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        # Subpestañas (dos modos)
        self.modes = QtWidgets.QTabWidget()
        layout.addWidget(self.modes)

        # --- Modo 1: Escuchar PC (beta) ---
        self._build_listen_tab()

        # --- Modo 2: Escribir ---
        self._build_write_tab()

    # ====== Escuchar PC (beta) ======
    def _build_listen_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)

        info = QtWidgets.QLabel("Graba audio del sistema (WASAPI loopback) o micrófono si no está disponible.\n"
                                "Luego se transcribe con Whisper.")
        info.setWordWrap(True)
        v.addWidget(info)

        # Controles
        form = QtWidgets.QHBoxLayout()
        self.seconds_sb = QtWidgets.QSpinBox()
        self.seconds_sb.setRange(3, 120)
        self.seconds_sb.setValue(15)
        form.addWidget(QtWidgets.QLabel("Duración (s):"))
        form.addWidget(self.seconds_sb)

        self.btn_record = QtWidgets.QPushButton("Grabar y transcribir")
        form.addWidget(self.btn_record)
        form.addStretch(1)
        v.addLayout(form)

        # Resultado de transcripción
        self.transcript = QtWidgets.QPlainTextEdit()
        self.transcript.setPlaceholderText("Transcripción…")
        v.addWidget(self.transcript, 2)

        # Botones aplicar/enviar
        hb = QtWidgets.QHBoxLayout()
        self.btn_to_writer = QtWidgets.QPushButton("Pegar en 'Escribir'")
        hb.addWidget(self.btn_to_writer)
        hb.addStretch(1)
        v.addLayout(hb)

        self.modes.addTab(tab, "Escuchar PC (beta)")

        # Signals
        self.btn_record.clicked.connect(self._on_record_transcribe)
        self.btn_to_writer.clicked.connect(lambda: self.paste_transcript.emit(self.transcript.toPlainText().strip()))

    def _on_record_transcribe(self):
        secs = int(self.seconds_sb.value())
        self.transcript.setPlainText("Grabando/transcribiendo…")

        api_key = load_api_key(False) or load_api_key(True)
        if not api_key:
            QtWidgets.QMessageBox.warning(self, "API Key", "Configura tu API key en Settings.")
            self.transcript.setPlainText("")
            return
        try:
            client = make_client(api_key)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.transcript.setPlainText("")
            return

        def worker():
            try:
                wav_path = record_system_or_mic_wav(secs, logs=lambda m: self.log_signal.emit(m))
                text = transcribe_file(client, self.cfg.transcribe_model, wav_path)
                self.transcript.setPlainText(text)
                self.log_signal.emit("Transcripción lista.")
            except Exception as e:
                self.transcript.setPlainText(f"[ERROR] {e}")
                self.log_signal.emit(f"[ERROR] {e}")

        QtCore.QThreadPool.globalInstance().start(_Runnable(worker))

    # ====== Escribir ======
    def _build_write_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)

        in_group = QtWidgets.QGroupBox("Escribe o pega un fragmento / resumen de la llamada")
        in_layout = QtWidgets.QVBoxLayout(in_group)
        self.input = QtWidgets.QPlainTextEdit()
        in_layout.addWidget(self.input)
        v.addWidget(in_group, 2)

        hb = QtWidgets.QHBoxLayout()
        self.btn_suggest = QtWidgets.QPushButton("Sugerir respuesta (inglés)")
        self.btn_clear = QtWidgets.QPushButton("Limpiar")
        hb.addWidget(self.btn_suggest); hb.addWidget(self.btn_clear); hb.addStretch(1)
        v.addLayout(hb)

        out_group = QtWidgets.QGroupBox("Respuesta sugerida")
        out_layout = QtWidgets.QVBoxLayout(out_group)
        self.output = QtWidgets.QPlainTextEdit(); self.output.setReadOnly(True)
        out_layout.addWidget(self.output)
        v.addWidget(out_group, 2)

        self.modes.addTab(tab, "Escribir")

        # Conectar señales
        self.btn_clear.clicked.connect(lambda: self.input.setPlainText(""))
        self.btn_suggest.clicked.connect(self._on_suggest)
        self.paste_transcript.connect(lambda t: self._paste_into_writer(t))

    def _paste_into_writer(self, text: str):
        self.modes.setCurrentIndex(1)  # Cambiar a "Escribir"
        if text:
            self.input.setPlainText(text)

    def _on_suggest(self):
        text = self.input.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "Vacío", "Escribe o pega algo primero.")
            return

        api_key = load_api_key(False) or load_api_key(True)
        try:
            client = make_client(api_key)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        prompt = build_prompt(text)
        self.output.setPlainText("Generando…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            reply = ask_model(client, self.cfg.chat_model, prompt)
            self.output.setPlainText(reply)
            self.log_signal.emit("Sugerencia generada.")
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

# Helper para ejecutar funciones en ThreadPool
class _Runnable(QtCore.QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def run(self):
        self.fn()

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
        self.compose  = ComposePage(self.cfg)
        self.settings = SettingsPage(self.cfg)
        self.knowledge = KnowledgePage()
        self.logs     = LogsPage()

        # Orden: COMPOSE primero (por defecto)
        tabs.addTab(self.compose, "Compose")
        tabs.addTab(self.settings, "Settings")
        tabs.addTab(self.knowledge, "Knowledge")
        tabs.addTab(self.logs, "Logs")
        tabs.setCurrentWidget(self.compose)  # pestaña por defecto

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
            self.status.showMessage(f"Conocimientos: {KNOWLEDGE_PATH} | Actualizado: {mtime}")
        except Exception:
            self.status.showMessage(f"Conocimientos: {KNOWLEDGE_PATH}")

def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))

    # Icono (opcional futuro): assets/icon.ico
    icon_path = resource_path('assets/icon.ico')
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())



