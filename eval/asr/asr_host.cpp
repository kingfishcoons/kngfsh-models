// Decodes the clips of an ASR comparison set with whisper.cpp the way the kngfsh-translate app does
// (app/src/main/cpp/whisper_jni.cpp, core DecodePlanner): greedy, temperature 0 without fallback, no timestamps, no
// previous-text context, non-speech tokens suppressed, no prompt, and the encoder window scaled to the clip:
// audio_ctx = ceil((clip + 1 s) / 30 s * n_audio_ctx), at least 512.
//
//   asr_host gate   --model ggml-base-q5_1.bin --manifest clips.tsv --root DIR --out languages.tsv [--threads N]
//   asr_host decode --model MODEL.bin --manifest clips.tsv --root DIR --out hypotheses.tsv [--threads N] [--no-flash]
//
// clips.tsv: a header line, then id, set, language (es, ca or auto), file (relative to --root), duration_ms, ...
// gate: for each clip, the app's language gate (es/ca restricted detection on the same window): id, p_es, p_ca, ms.
// decode: for each clip whose language is es or ca: id, language, ms, token_p, text (tabs and newlines in the text
// become spaces). A clip with language "auto" is skipped; the workflow resolves it with the gate first.
//
// Audio must be 16 kHz mono PCM16 WAV (prepare_sets.py writes it so).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "whisper.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Args {
    std::string mode, model, manifest, root, out;
    int threads = 4;
    bool flash = true;
};

bool parse(int argc, char **argv, Args &a) {
    if (argc < 2) return false;
    a.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&](std::string &v) { if (i + 1 < argc) { v = argv[++i]; return true; } return false; };
        std::string v;
        if (k == "--model" && next(v)) a.model = v;
        else if (k == "--manifest" && next(v)) a.manifest = v;
        else if (k == "--root" && next(v)) a.root = v;
        else if (k == "--out" && next(v)) a.out = v;
        else if (k == "--threads" && next(v)) a.threads = std::max(1, std::stoi(v));
        else if (k == "--no-flash") a.flash = false;
        else return false;
    }
    return (a.mode == "gate" || a.mode == "decode") && !a.model.empty() && !a.manifest.empty() && !a.out.empty();
}

struct Clip {
    std::string id, set, language, file;
    long duration_ms = 0;
};

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, '\t')) out.push_back(cell);
    return out;
}

std::vector<Clip> read_manifest(const std::string &path) {
    std::ifstream in(path);
    std::vector<Clip> clips;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto c = split_tabs(line);
        if (c.size() < 5) continue;
        clips.push_back({c[0], c[1], c[2], c[3], std::stol(c[4])});
    }
    return clips;
}

// 16 kHz mono PCM16 WAV to floats in [-1, 1); empty on anything else.
std::vector<float> read_wav(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) return {};
    size_t pos = 12;
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    while (pos + 8 <= bytes.size()) {
        uint32_t size;
        std::memcpy(&size, bytes.data() + pos + 4, 4);
        const char *id = bytes.data() + pos;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::memcpy(&channels, bytes.data() + pos + 10, 2);
            std::memcpy(&rate, bytes.data() + pos + 12, 4);
            std::memcpy(&bits, bytes.data() + pos + 22, 2);
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (channels != 1 || rate != WHISPER_SAMPLE_RATE || bits != 16) return {};
            const size_t n = std::min<size_t>(size, bytes.size() - pos - 8) / 2;
            std::vector<float> pcm(n);
            for (size_t i = 0; i < n; i++) {
                int16_t s;
                std::memcpy(&s, bytes.data() + pos + 8 + 2 * i, 2);
                pcm[i] = static_cast<float>(s) / 32768.0f;
            }
            return pcm;
        }
        pos += 8 + size + (size & 1);
    }
    return {};
}

// The app's DecodePlanner: the 30 s window scaled to the clip plus 1 s, at least 512 frames.
int audio_ctx_for(whisper_context *ctx, long duration_ms) {
    const int max_ctx = whisper_n_audio_ctx(ctx);
    const int ctx_len = static_cast<int>(std::ceil((duration_ms + 1000.0) / 30000.0 * max_ctx));
    return std::clamp(ctx_len, std::min(512, max_ctx), max_ctx);
}

bool stop_before_encoder(whisper_context *, whisper_state *, void *) { return false; }

whisper_full_params base_params(int threads, int audio_ctx, const char *language) {
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = threads;
    p.audio_ctx = audio_ctx;
    p.language = language;
    p.detect_language = false;
    p.translate = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.print_special = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.suppress_nst = true;
    p.temperature = 0.0f;
    p.temperature_inc = 0.0f;
    return p;
}

std::string one_line(std::string s) {
    for (char &c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    const auto b = s.find_first_not_of(' ');
    const auto e = s.find_last_not_of(' ');
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

void quiet_log(ggml_log_level level, const char *text, void *) {
    // The gate stops whisper_full at the encoder on purpose; whisper.cpp logs that as an error every time.
    if (level == GGML_LOG_LEVEL_ERROR && std::strstr(text, "encoder_begin_callback returned false") == nullptr) std::fputs(text, stderr);
}

}  // namespace

int main(int argc, char **argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: asr_host gate|decode --model M --manifest clips.tsv --root DIR --out OUT [--threads N] [--no-flash]\n");
        return 2;
    }
    whisper_log_set(quiet_log, nullptr);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    cp.flash_attn = a.flash;
    const auto load_start = Clock::now();
    whisper_context *ctx = whisper_init_from_file_with_params(a.model.c_str(), cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "could not load %s\n", a.model.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded %s in %.0f ms; %s\n", a.model.c_str(), ms_since(load_start), whisper_print_system_info());

    const auto clips = read_manifest(a.manifest);
    std::ofstream out(a.out);
    if (a.mode == "gate") out << "id\tp_es\tp_ca\tms\n";
    else out << "id\tlanguage\tms\ttoken_p\ttext\n";
    int done = 0, failed = 0;
    double audio_ms = 0, work_ms = 0;
    for (const auto &clip : clips) {
        if (a.mode == "decode" && clip.language != "es" && clip.language != "ca") continue;
        const auto pcm = read_wav(a.root + "/" + clip.file);
        if (pcm.empty()) {
            std::fprintf(stderr, "%s: unreadable audio %s\n", clip.id.c_str(), clip.file.c_str());
            failed++;
            continue;
        }
        const int ctx_len = audio_ctx_for(ctx, clip.duration_ms);
        const auto start = Clock::now();
        if (a.mode == "gate") {
            // The app's gate: compute the mel and set the window without encoding, then detect over that window.
            whisper_full_params p = base_params(a.threads, ctx_len, "es");
            p.encoder_begin_callback = stop_before_encoder;
            std::vector<float> probs(whisper_lang_max_id() + 1, 0.0f);
            if (whisper_full(ctx, p, pcm.data(), static_cast<int>(pcm.size())) != 0 ||
                whisper_lang_auto_detect(ctx, 0, a.threads, probs.data()) < 0) {
                failed++;
                continue;
            }
            const double ms = ms_since(start);
            out << clip.id << '\t' << probs[whisper_lang_id("es")] << '\t' << probs[whisper_lang_id("ca")] << '\t' << ms << '\n';
            work_ms += ms;
        } else {
            whisper_full_params p = base_params(a.threads, ctx_len, clip.language.c_str());
            if (whisper_full(ctx, p, pcm.data(), static_cast<int>(pcm.size())) != 0) {
                failed++;
                continue;
            }
            const double ms = ms_since(start);
            std::string text;
            double p_sum = 0;
            int p_n = 0;
            const whisper_token eot = whisper_token_eot(ctx);
            for (int i = 0; i < whisper_full_n_segments(ctx); i++) {
                const std::string seg = whisper_full_get_segment_text(ctx, i);
                if (seg.find_first_not_of(" \t\r\n") == std::string::npos) continue;
                text += seg;
                // Mean probability of the text tokens of segments that hold text, as the app's meanTokenProb.
                for (int k = 0; k < whisper_full_n_tokens(ctx, i); k++) {
                    if (whisper_full_get_token_id(ctx, i, k) >= eot) continue;
                    p_sum += whisper_full_get_token_p(ctx, i, k);
                    p_n++;
                }
            }
            char token_p[32];
            if (p_n > 0) std::snprintf(token_p, sizeof(token_p), "%.4f", p_sum / p_n);
            else token_p[0] = '\0';
            out << clip.id << '\t' << clip.language << '\t' << static_cast<long>(ms) << '\t' << token_p << '\t' << one_line(text) << '\n';
            out.flush();  // a run cut short by the job's time limit keeps what it decoded
            work_ms += ms;
        }
        audio_ms += static_cast<double>(clip.duration_ms);
        if (++done % 25 == 0) std::fprintf(stderr, "%d clips, %.2f x real time so far\n", done, work_ms / std::max(1.0, audio_ms));
    }
    std::fprintf(stderr, "%s: %d clips, %d failed, %.0f s of audio in %.0f s (%.2f x real time), %d threads\n", a.mode.c_str(), done, failed,
                 audio_ms / 1000, work_ms / 1000, work_ms / std::max(1.0, audio_ms), a.threads);
    whisper_free(ctx);
    return failed > 0 && done == 0 ? 1 : 0;
}
