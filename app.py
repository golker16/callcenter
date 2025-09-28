#!/usr/bin/env python3
"""
CallCenter Helper – Qt + QDarkStyle (Whisper + GPT)
---------------------------------------------------
Cambios claves:
- 'Generar respuesta' usa TODO el bloque de mensajes del cliente desde la última respuesta del agente.
- Burbujas sin etiquetas (solo colores): gris = cliente, azul = agente.
- Historial en onedir: ./historial/conv_YYYYmmdd_HHMMSS/chat.txt (junto al .exe).
- Chunks ~2 s @ 32 kHz mono (loopback) para baja latencia.

Requisitos:
  pip install -r requirements.txt
"""

from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import qdarkstyle

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Audio: loopback (soundcard) + fallback mic (sounddevice)
try:
    import soundcard as sc
except Exception:
    sc = None
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except Exception:
    sd = None
    sf = None
    np = None

# ------------------------
# Rutas
# ------------------------
def runtime_dir() -> Path:
    """Directorio 'onedir' donde vive el .exe/.py (no el temp _MEIPASS)."""
    try:
        return Path(sys.argv[0]).resolve().parent
    except Exception:
        return Path.cwd()

RUNTIME_DIR = runtime_dir()
ASSETS_DIR  = RUNTIME_DIR / "assets"
HISTORY_ROOT = RUNTIME_DIR / "historial"
HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

# Conocimientos en runtime (inclúyelo en el onedir)
KNOWLEDGE_PATH = RUNTIME_DIR / "conocimientos.md"

# API key en escritorio
DESKTOP_API_DIR = Path("D:/Desktop")
DESKTOP_API_PATH = DESKTOP_API_DIR / "main_api_key.txt"

# ------------------------
# Utilidades
# ------------------------
def ensure_knowledge_file():
    """Asegura conocimientos.md en el onedir (si no existe, crea uno básico)."""
    if KNOWLEDGE_PATH.exists():
        return
    KNOWLEDGE_PATH.write_text("# Conocimientos del call center\n\n# Rol\n\n", encoding="utf-8")

def read_knowledge() -> str:
    ensure_knowledge_file()
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")

def write_knowledge(text: str):
    KNOWLEDGE_PATH.write_text(text, encoding="utf-8")

H1_CONO = "# Conocimientos del call center"
H1_ROL  = "# Rol"

def _find_h1_positions(md: str) -> list[tuple[int, str]]:
    text = md.replace("\r\n", "\n")
    return [(m.start(), "# " + m.group(1).strip())
            for m in re.finditer(r'(?im)^\s*#\s*(.+?)\s*$', text)]

def split_sections(md: str) -> tuple[str, str]:
    text = md.replace("\r\n", "\n")
    headers = _find_h1_positions(text)
    def norm(s: str) -> str: return re.sub(r'\s+', ' ', s.strip().lower())
    def extract(start_pos: int) -> str:
        starts = sorted([p for p,_ in headers])
        nxt = [p for p in starts if p > start_pos]
        end = nxt[0] if nxt else len(text)
        return text[start_pos:end].strip()+"\n"
    pos_cono = pos_rol = None
    for pos,h in headers:
        nh = norm(h)
        if nh == norm(H1_CONO): pos_cono = pos
        if nh == norm(H1_ROL):  pos_rol  = pos
    if pos_cono is None and pos_rol is None:
        return (f"{H1_CONO}\n\n{text.strip()}\n", f"{H1_ROL}\n\n")
    cono = extract(pos_cono) if pos_cono is not None else f"{H1_CONO}\n\n"
    rol  = extract(pos_rol)  if pos_rol  is not None else f"{H1_ROL}\n\n"
    return (cono, rol)

def merge_sections(cono_md: str, rol_md: str) -> str:
    cono_md = cono_md.strip(); rol_md = rol_md.strip()
    if not re.match(r'(?is)^\s*#\s*conocimientos del call center', cono_md):
        cono_md = f"{H1_CONO}\n\n{cono_md}"
    if not re.match(r'(?is)^\s*#\s*rol', rol_md):
        rol_md = f"{H1_ROL}\n\n{rol_md}"
    return f"{cono_md}\n\n{rol_md}\n"

# ------------------------
# Config & API Key (simple)
# ------------------------
@dataclass
class AppConfig:
    chat_model: str = "gpt-3.5-turbo"
    transcribe_model: str = "whisper-1"

CONFIG_PATH = RUNTIME_DIR / "config.json"

def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        try:
            return AppConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    cfg = AppConfig()
    CONFIG_PATH.write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    return cfg

def save_config(cfg: AppConfig):
    CONFIG_PATH.write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

def save_api_key_to_desktop(api_key: str):
    if not api_key:
        raise ValueError("API key vacía")
    DESKTOP_API_DIR.mkdir(parents=True, exist_ok=True)
    DESKTOP_API_PATH.write_text(api_key.strip(), encoding="utf-8")

def load_api_key_from_desktop() -> str | None:
    try:
        if DESKTOP_API_PATH.exists():
            s = DESKTOP_API_PATH.read_text(encoding="utf-8").strip()
            return s or None
    except Exception:
        return None
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

def build_prompt(block_text: str) -> str:
    """Usa SOLO el bloque del cliente desde la última respuesta + knowledge opcional."""
    kb = read_knowledge()
    return (
        "You are a call center assistant. Consider the company knowledge IF relevant; "
        "otherwise answer normally. Always reply in English with a professional, concise, "
        "empathetic and actionable tone. Ask for the minimum missing info only.\n\n"
        "### COMPANY KNOWLEDGE (optional)\n" + kb +
        "\n### CUSTOMER BLOCK (since last agent reply)\n" + block_text +
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

def transcribe_file(client: OpenAI, model: str, file_path: Path, logs=None) -> str:
    if logs: logs(f"Enviando a Whisper: {file_path.name}")
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(model=model, file=f)
    text = getattr(tr, "text", "") or json.dumps(getattr(tr, "__dict__", {}), ensure_ascii=False)
    if logs: logs(f"Texto recibido: {text[:200]}{'...' if len(text)>200 else ''}")
    return text

# ------------------------
# Audio helpers
# ------------------------
def _to_mono_float32(data) -> "np.ndarray":
    if np is None:
        return data
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32, copy=False)

def record_chunk_wav(seconds: int = 2, samplerate: int = 32000, channels: int = 2, logs=None) -> Path:
    out = RUNTIME_DIR / f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"

    # Loopback preferido
    if sc is not None:
        mic = None
        try:
            spk = sc.default_speaker()
            mic = sc.get_microphone(spk.name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            try:
                loopbacks = [m for m in sc.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]
                if logs:
                    names = ", ".join(m.name for m in loopbacks)
                    logs(f"Loopbacks disponibles: {names or '(ninguno)'}")
                mic = loopbacks[0] if loopbacks else None
            except Exception as e:
                if logs: logs(f"No se pudo listar loopbacks: {e}")
                mic = None
        if mic is not None:
            try:
                if logs: logs(f"Grabando altavoces (loopback) con '{mic.name}'…")
                with mic.recorder(samplerate=samplerate) as rec:
                    data = rec.record(numframes=int(seconds * samplerate))
                data_mono = _to_mono_float32(data)
                if sf is not None:
                    sf.write(out.as_posix(), data_mono, samplerate)
                else:
                    import wave, numpy as _np
                    data16 = (_np.clip(data_mono, -1.0, 1.0)*32767).astype(_np.int16)
                    with wave.open(out.as_posix(), "wb") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(samplerate); wf.writeframes(data16.tobytes())
                if logs: logs(f"WAV escrito: {out.name} ({out.stat().st_size} bytes)")
                return out
            except Exception as e:
                if logs: logs(f"Loopback (soundcard) falló: {e}")

    # Fallback mic
    if sd is None or sf is None:
        raise RuntimeError("No loopback y falta 'sounddevice/soundfile' para fallback.")
    if logs: logs("Grabando micrófono (fallback)…")
    rec = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(out.as_posix(), rec, samplerate)
    if logs: logs(f"WAV escrito (mic): {out.name} ({out.stat().st_size} bytes)")
    return out

# ------------------------
# Chat bubbles (sin etiquetas visibles)
# ------------------------
def bubble_html(role: str, text: str) -> str:
    text = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n","<br/>")
    if role == "client":
        align = "flex-start"; bg = "#3a3f44"; fg = "#e6e6e6"; corner = "border-top-left-radius:4px;"
    else:
        align = "flex-end";   bg = "#2d7ef7"; fg = "#ffffff"; corner = "border-top-right-radius:4px;"
    return f"""
    <div style="display:flex; justify-content:{align}; margin:6px 0;">
      <div style="max-width:70%; background:{bg}; color:{fg}; padding:10px 12px; border-radius:12px; {corner}
                  font-size:14px; line-height:1.4;">
        {text}
      </div>
    </div>
    """

# ------------------------
# Páginas
# ------------------------
class SettingsPage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)
    cfg_changed = QtCore.Signal()

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        form = QtWidgets.QFormLayout(self)

        self.api_edit = QtWidgets.QLineEdit()
        self.api_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        if load_api_key_from_desktop():
            self.api_edit.setPlaceholderText("*** guardada en D:\\Desktop ***")

        self.model_cb = QtWidgets.QComboBox()
        self.model_cb.addItems(["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
        self.model_cb.setCurrentText(self.cfg.chat_model)

        self.stt_cb = QtWidgets.QComboBox()
        self.stt_cb.addItems(["whisper-1", "gpt-4o-transcribe"])
        self.stt_cb.setCurrentText(self.cfg.transcribe_model)

        save_btn = QtWidgets.QPushButton("Guardar")
        save_btn.clicked.connect(self._save)

        form.addRow("OpenAI API Key:", self.api_edit)
        form.addRow("Modelo de chat:", self.model_cb)
        form.addRow("Modelo de transcripción:", self.stt_cb)
        form.addRow("", save_btn)

    def _save(self):
        self.cfg.chat_model = self.model_cb.currentText()
        self.cfg.transcribe_model = self.stt_cb.currentText()
        save_config(self.cfg)
        val = self.api_edit.text().strip()
        try:
            if val and val != "*** guardada en D:\\Desktop ***":
                save_api_key_to_desktop(val)
                self.api_edit.clear()
                self.api_edit.setPlaceholderText("*** guardada en D:\\Desktop ***")
                self.log_signal.emit(f"API key guardada en {DESKTOP_API_PATH}")
            self.log_signal.emit("Configuración guardada.")
            QtWidgets.QMessageBox.information(self, "Guardado", "Configuración guardada.")
            self.cfg_changed.emit()
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Guardando API key: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo guardar la API key: {e}")

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
        self.tabs.setCurrentIndex(0)

        h = QtWidgets.QHBoxLayout()
        save_btn   = QtWidgets.QPushButton("Guardar")
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar como...", str(RUNTIME_DIR / "conocimientos.md"), "Markdown (*.md)")
        if not path: return
        try:
            merged = merge_sections(self.cono_edit.toPlainText(), self.rol_edit.toPlainText())
            Path(path).write_text(merged, encoding="utf-8")
            self.log_signal.emit(f"Exportado a {path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Exportar: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo exportar: {e}")

    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Importar .md", str(RUNTIME_DIR), "Markdown (*.md);;Text (*.txt);;All (*.*)")
        if not path: return
        try:
            data = Path(path).read_text(encoding="utf-8", errors="ignore")
            cono_md, rol_md = split_sections(data)
            self.cono_edit.setPlainText(cono_md)
            self.rol_edit.setPlainText(rol_md)
            self.log_signal.emit(f"Importado: {path}")
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Importar: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo importar: {e}")

class ListenWorker(QtCore.QThread):
    chunk_seconds = 2
    samplerate = 32000
    channels = 2

    new_text = QtCore.Signal(str)
    log = QtCore.Signal(str)

    def __init__(self, api_key: str, stt_model: str, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.stt_model = stt_model
        self._running = False

    def run(self):
        self._running = True
        try:
            client = make_client(self.api_key)
        except Exception as e:
            self.log.emit(f"[ERROR] {e}")
            return
        while self._running:
            try:
                wav = record_chunk_wav(self.chunk_seconds, self.samplerate, self.channels, logs=lambda m: self.log.emit(m))
                size = 0
                try: size = wav.stat().st_size
                except Exception: pass
                if size < 2000:
                    self.log.emit("(silencio)")
                else:
                    text = transcribe_file(client, self.stt_model, wav, logs=lambda m: self.log.emit(m)).strip()
                    if text:
                        self.new_text.emit(text)
                    else:
                        self.log.emit("(Whisper no devolvió texto)")
                try: wav.unlink(missing_ok=True)
                except Exception: pass
            except Exception as e:
                self.log.emit(f"[ERROR] Transcripción: {e}")

    def stop(self):
        self._running = False

class ComposePage(QtWidgets.QWidget):
    log_signal = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        # Event log en memoria: [{'role': 'client'|'agent', 'text': str}]
        self.events: list[dict] = []
        self.last_agent_index = -1  # índice del último 'agent' en self.events

        # Historial por conversación
        self.conv_dir: Path | None = None
        self.conv_file: Path | None = None
        self._start_new_conversation_dir()

        v = QtWidgets.QVBoxLayout(self)

        hb = QtWidgets.QHBoxLayout()
        self.btn_listen   = QtWidgets.QPushButton("▶️ Empezar a escuchar")
        self.btn_generate = QtWidgets.QPushButton("🧠 Generar respuesta")
        self.btn_newconv  = QtWidgets.QPushButton("🆕 Nueva conversación")
        hb.addWidget(self.btn_listen); hb.addWidget(self.btn_generate); hb.addWidget(self.btn_newconv); hb.addStretch(1)
        v.addLayout(hb)

        self.chat = QtWidgets.QTextBrowser()
        self.chat.setOpenExternalLinks(True); self.chat.setReadOnly(True)
        pal = self.chat.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1f22"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#E0E0E0"))
        self.chat.setPalette(pal)
        v.addWidget(self.chat, 1)

        self.btn_listen.clicked.connect(self._toggle_listen)
        self.btn_generate.clicked.connect(self._generate_reply)
        self.btn_newconv.clicked.connect(self._new_conversation)

    # ---- Historial en onedir ----
    def _start_new_conversation_dir(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_dir  = HISTORY_ROOT / f"conv_{ts}"
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self.conv_file = self.conv_dir / "chat.txt"
        self._write_history(f"=== Conversation started {datetime.now().isoformat()} ===\n")

    def _write_history(self, line: str):
        if self.conv_file:
            with self.conv_file.open("a", encoding="utf-8") as f:
                f.write(line)

    def _add_bubble(self, role: str, text: str):
        # role: 'client' | 'agent'
        self.chat.insertHtml(bubble_html(role, text))
        self.chat.moveCursor(QtGui.QTextCursor.End)

    # ---- Escuchar ----
    def _toggle_listen(self):
        if getattr(self, "worker", None):
            self.worker.stop(); self.worker.wait(2000); self.worker = None
            self.btn_listen.setText("▶️ Empezar a escuchar")
            self.log_signal.emit("Escucha detenida.")
            return
        api_key = load_api_key_from_desktop()
        if not api_key:
            QtWidgets.QMessageBox.warning(self, "API Key", "Configura tu API key en Settings (se guardará en D:\\Desktop).")
            return
        self.worker = ListenWorker(api_key, self.cfg.transcribe_model)
        self.worker.new_text.connect(self._on_new_transcript)
        self.worker.log.connect(lambda m: self.log_signal.emit(m))
        self.worker.start()
        self.btn_listen.setText("⏸️ Dejar de escuchar")
        self.log_signal.emit("Escuchando…")

    @QtCore.Slot(str)
    def _on_new_transcript(self, text: str):
        t = text.strip()
        if not t:
            return
        self.events.append({"role":"client", "text": t})
        self._add_bubble("client", t)
        self._write_history(f"Cliente: {t}\n")

    # ---- Generar: bloque completo desde la última respuesta ----
    def _generate_reply(self):
        # recolecta todos los eventos 'client' posteriores a last_agent_index
        block_msgs = [e["text"] for i,e in enumerate(self.events) if i > self.last_agent_index and e["role"]=="client"]
        block_text = "\n".join(block_msgs).strip()
        if not block_text:
            QtWidgets.QMessageBox.information(self, "Sin contenido", "No hay mensajes nuevos del cliente desde tu última respuesta.")
            return

        api_key = load_api_key_from_desktop()
        try:
            client = make_client(api_key)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        prompt = build_prompt(block_text)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            reply = ask_model(client, self.cfg.chat_model, prompt).strip() or "(No reply)"
            self.events.append({"role":"agent", "text": reply})
            self.last_agent_index = len(self.events)-1
            self._add_bubble("agent", reply)
            self._write_history(f"Yo: {reply}\n")
            self.log_signal.emit("Respuesta generada.")
        except Exception as e:
            self._add_bubble("agent", f"[ERROR] {e}")
            self.log_signal.emit(f"[ERROR] {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---- Nueva conversación ----
    def _new_conversation(self):
        if getattr(self, "worker", None):
            self.worker.stop(); self.worker.wait(2000); self.worker = None
            self.btn_listen.setText("▶️ Empezar a escuchar")
        self.chat.clear()
        self.events.clear()
        self.last_agent_index = -1
        self._start_new_conversation_dir()
        self.log_signal.emit("Nueva conversación iniciada.")

class LogsPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        self.box = QtWidgets.QPlainTextEdit(); self.box.setReadOnly(True)
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
        self.resize(1080, 700)

        self.cfg = load_config()
        ensure_knowledge_file()

        tabs = QtWidgets.QTabWidget(); self.setCentralWidget(tabs)
        self.compose   = ComposePage(self.cfg)
        self.settings  = SettingsPage(self.cfg)
        self.knowledge = KnowledgePage()
        self.logs      = LogsPage()

        tabs.addTab(self.compose,   "Compose")
        tabs.addTab(self.settings,  "Settings")
        tabs.addTab(self.knowledge, "Knowledge")
        tabs.addTab(self.logs,      "Logs")
        tabs.setCurrentWidget(self.compose)

        self.settings.log_signal.connect(self.logs.append)
        self.compose.log_signal.connect(self.logs.append)
        self.knowledge.knowledge_changed.connect(self._update_status)

        self.status = self.statusBar()
        self._update_status()

    @QtCore.Slot()
    def _update_status(self):
        try:
            mtime = datetime.fromtimestamp(KNOWLEDGE_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.status.showMessage(f"Conocimientos: {KNOWLEDGE_PATH} | Actualizado: {mtime} | Historial: {HISTORY_ROOT}")
        except Exception:
            self.status.showMessage(f"Conocimientos: {KNOWLEDGE_PATH} | Historial: {HISTORY_ROOT}")

    def closeEvent(self, e: QtGui.QCloseEvent):
        # Historial ya está escrito incrementalmente. Solo registramos cierre.
        try:
            if self.compose.conv_file:
                with self.compose.conv_file.open("a", encoding="utf-8") as f:
                    f.write(f"=== Conversation closed {datetime.now().isoformat()} ===\n")
        except Exception:
            pass
        return super().closeEvent(e)

def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    icon_path = ASSETS_DIR / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    w = MainWindow(); w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())





