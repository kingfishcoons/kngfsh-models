"""Streaming recognizers behind one small interface, for the draft-text bake-off (kngfsh-translate D125, D127).

Each engine takes 16 kHz mono float32 audio in chunks and, after every chunk, returns the text it would show now:
everything it has finished plus what it is still working on, as one string. `finish()` flushes the end of the stream.

    moonshine   moonshine-voice (the Android library's engine, ONNX Runtime .ort int8 files)
    sherpa      sherpa-onnx online transducer (streaming Zipformer)
    vosk        Vosk / Kaldi online decoding
"""
import json
import os


class Engine:
    def feed(self, chunk):
        raise NotImplementedError

    def cut(self):
        """Called where the app's segmenter closes a segment, for engines run with `cut: segments`."""

    def finish(self):
        raise NotImplementedError


class Moonshine(Engine):
    """moonshine-voice's streaming transcriber. Lines it has completed are kept; the open line is shown as it grows.

    `update_interval` is the library's floor between passes (it also waits for a pass's own cost in audio, so a slow
    machine makes fewer, longer passes rather than falling behind)."""

    def __init__(self, model_dir, arch, update_interval=0.25):
        from moonshine_voice import ModelArch
        from moonshine_voice.transcriber import Transcriber

        self.transcriber = Transcriber(model_path=str(model_dir), model_arch=ModelArch(arch), update_interval=update_interval)
        self.update_interval = update_interval
        self.lines = {}
        self.generation = 0
        self._open_stream()

    def _open_stream(self):
        self.stream = self.transcriber.create_stream(update_interval=self.update_interval)
        self.stream.add_listener(self._on_event)
        self.stream.start()

    def _on_event(self, event):
        line = getattr(event, "line", None)
        if line is not None:
            key = (self.generation, line.line_id)
            start = self.lines.get(key, ((self.generation, line.start_time), ""))[0]
            self.lines[key] = (start, line.text or "")

    def cut(self):
        """Ends the stream where the app's segmenter closes a segment and starts a fresh one, so no line runs on past
        the 8 s cap (the variant `cut: segments` in models.json)."""
        self.stream.stop()
        self.generation += 1
        self._open_stream()

    def _text(self):
        return " ".join(t for _, t in sorted(self.lines.values()) if t.strip())

    def feed(self, chunk):
        self.stream.add_audio(chunk.tolist(), 16000)
        return self._text()

    def finish(self):
        self.stream.stop()
        text = self._text()
        self.transcriber.close()
        return text


class Sherpa(Engine):
    """sherpa-onnx online transducer, greedy search, with its endpoint rules: at an endpoint the utterance is kept and
    the stream reset, as its Android example does."""

    def __init__(self, model_dir, encoder, decoder, joiner, tokens="tokens.txt", threads=1):
        import sherpa_onnx

        p = lambda name: os.path.join(model_dir, name)
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=p(tokens), encoder=p(encoder), decoder=p(decoder), joiner=p(joiner), num_threads=threads,
            sample_rate=16000, feature_dim=80, decoding_method="greedy_search", enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4, rule2_min_trailing_silence=1.2, rule3_min_utterance_length=20.0,
        )
        self.stream = self.recognizer.create_stream()
        self.done = []

    def _step(self):
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        current = self.recognizer.get_result(self.stream).strip()
        if self.recognizer.is_endpoint(self.stream):
            if current:
                self.done.append(current)
            current = ""
            self.recognizer.reset(self.stream)
        return " ".join(self.done + ([current] if current else []))

    def feed(self, chunk):
        self.stream.accept_waveform(16000, chunk)
        return self._step()

    def finish(self):
        import numpy as np

        self.stream.accept_waveform(16000, np.zeros(int(0.66 * 16000), np.float32))  # tail padding, as its examples do
        self.stream.input_finished()
        return self._step()


class Vosk(Engine):
    """Vosk's KaldiRecognizer: finished utterances from Result(), the open one from PartialResult()."""

    def __init__(self, model_dir):
        import vosk

        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_dir))
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.done = []

    def _join(self, current):
        return " ".join(self.done + ([current] if current else []))

    def feed(self, chunk):
        import numpy as np

        pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
        if self.recognizer.AcceptWaveform(pcm):
            text = json.loads(self.recognizer.Result()).get("text", "").strip()
            if text:
                self.done.append(text)
            return self._join("")
        return self._join(json.loads(self.recognizer.PartialResult()).get("partial", "").strip())

    def finish(self):
        text = json.loads(self.recognizer.FinalResult()).get("text", "").strip()
        if text:
            self.done.append(text)
        return self._join("")


def make(spec):
    """spec: the engine's entry from models.json, with `dir` set to where its files are."""
    kind = spec["kind"]
    if kind == "moonshine":
        return Moonshine(spec["dir"], spec["arch"], spec.get("update_interval", 0.25))
    if kind == "sherpa":
        return Sherpa(spec["dir"], spec["encoder"], spec["decoder"], spec["joiner"], spec.get("tokens", "tokens.txt"))
    if kind == "vosk":
        return Vosk(spec["dir"])
    raise ValueError(f"unknown engine kind {kind}")
