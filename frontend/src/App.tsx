import { useState, useEffect, useRef, useCallback } from "react";
import "./index.css";
import {
  fetchModels,
  fetchModelInfo,
  loadModel,
  generateTTS,
  generateVoiceClone,
  type ModelVariant,
} from "./api";

// ─── Tab Ids ─────────────────────────────────────────────────────────
type TabId = "custom_voice" | "voice_design" | "voice_clone";

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: "custom_voice", label: "Custom Voice", icon: "🎙️" },
  { id: "voice_design", label: "Voice Design", icon: "🎨" },
  { id: "voice_clone", label: "Voice Clone", icon: "🧬" },
];

// ─── Component ───────────────────────────────────────────────────────
export default function App() {
  // ── Model state ──
  const [models, setModels] = useState<ModelVariant[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const [modelLoading, setModelLoading] = useState(false);

  // ── Model info ──
  const [languages, setLanguages] = useState<string[]>([]);
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [modelType, setModelType] = useState<string | null>(null);

  // ── Tab ──
  const [activeTab, setActiveTab] = useState<TabId>("custom_voice");

  // ── Form state ──
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [speaker, setSpeaker] = useState("");
  const [instruct, setInstruct] = useState("");

  // Voice clone
  const [refAudio, setRefAudio] = useState<File | null>(null);
  const [refText, setRefText] = useState("");
  const [xVectorOnly, setXVectorOnly] = useState(false);

  // ── Generation state ──
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // ── Load models list on mount ──
  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        setCurrentModel(data.current);
      })
      .catch(() => setError("Cannot connect to server. Is it running on :8000?"));
  }, []);

  // ── Fetch model info when model changes ──
  useEffect(() => {
    if (!currentModel) return;
    fetchModelInfo()
      .then((info) => {
        setLanguages(info.supported_languages || []);
        setSpeakers(info.supported_speakers || []);
        setModelType(info.model_type);
        // Auto-select first speaker
        if (info.supported_speakers?.length) setSpeaker(info.supported_speakers[0]);
      })
      .catch(() => { });
  }, [currentModel]);

  // ── Switch model ──
  const handleModelSwitch = useCallback(
    async (modelPath: string) => {
      if (modelPath === currentModel) return;
      setModelLoading(true);
      setError(null);
      try {
        const result = await loadModel(modelPath);
        setCurrentModel(result.model_path);
        setModelType(result.model_type);
      } catch (e: any) {
        setError(`Model load failed: ${e.message}`);
      } finally {
        setModelLoading(false);
      }
    },
    [currentModel]
  );

  // ── Generate ──
  const handleGenerate = useCallback(async () => {
    if (!text.trim()) return;
    setGenerating(true);
    setError(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);

    try {
      let blob: Blob;

      if (activeTab === "voice_clone") {
        if (!refAudio) throw new Error("Please upload a reference audio file");
        blob = await generateVoiceClone({
          text,
          language,
          refAudio,
          refText: refText || undefined,
          xVectorOnly,
        });
      } else {
        blob = await generateTTS({
          text,
          language,
          mode: activeTab,
          speaker: activeTab === "custom_voice" ? speaker : undefined,
          instruct:
            activeTab === "voice_design" || activeTab === "custom_voice"
              ? instruct || undefined
              : undefined,
        });
      }

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      // Auto play
      setTimeout(() => audioRef.current?.play(), 100);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  }, [text, language, speaker, instruct, activeTab, refAudio, refText, xVectorOnly, audioUrl]);

  // ── Current tab matching model type? ──
  const currentModelObj = models.find((m) => m.id === currentModel);

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-[var(--color-border)] px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[var(--color-accent)] flex items-center justify-center text-xl font-bold">
              Q
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">Qwen3-TTS</h1>
              <p className="text-xs text-[var(--color-text-muted)]">
                Text-to-Speech Studio
              </p>
            </div>
          </div>

          {/* ── Model Selector ── */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-[var(--color-text-secondary)]">Model:</label>
            <select
              value={currentModel}
              onChange={(e) => handleModelSwitch(e.target.value)}
              disabled={modelLoading}
              className="bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors min-w-[280px]"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            {modelLoading && (
              <div className="w-5 h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
            )}
            {currentModelObj && (
              <span className="text-xs px-2 py-1 rounded-md bg-[var(--color-accent)]/20 text-[var(--color-accent)]">
                {currentModelObj.type}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8">
        {/* ── Tabs ── */}
        <div className="flex gap-1 mb-6 bg-[var(--color-bg-secondary)] p-1 rounded-xl w-fit">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab.id
                  ? "bg-[var(--color-accent)] text-white shadow-lg shadow-[var(--color-accent-glow)]"
                  : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-white/5"
                }`}
            >
              <span className="mr-1.5">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ── Left: Input ── */}
          <div className="lg:col-span-2 space-y-4">
            {/* Text input */}
            <div className="glass-card p-5">
              <label className="text-sm font-medium text-[var(--color-text-secondary)] mb-2 block">
                Text to Synthesize
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                placeholder="Enter text here… supports Chinese, English, Japanese, Korean and more."
                className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-xl p-4 text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-accent)] transition-colors resize-none text-sm leading-relaxed"
              />
              <div className="flex justify-between mt-2 text-xs text-[var(--color-text-muted)]">
                <span>{text.length} characters</span>
                <span>Language: {language}</span>
              </div>
            </div>

            {/* Mode-specific inputs */}
            {activeTab === "custom_voice" && (
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                  🎙️ Custom Voice Settings
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                      Speaker
                    </label>
                    <select
                      value={speaker}
                      onChange={(e) => setSpeaker(e.target.value)}
                      className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]"
                    >
                      {speakers.length ? (
                        speakers.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))
                      ) : (
                        <option value="">No speakers available</option>
                      )}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                      Language
                    </label>
                    <select
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                      className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]"
                    >
                      <option value="Auto">Auto</option>
                      {languages.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Instruct (optional — style description)
                  </label>
                  <input
                    value={instruct}
                    onChange={(e) => setInstruct(e.target.value)}
                    placeholder="e.g. Speak cheerfully and warmly"
                    className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-accent)]"
                  />
                </div>
              </div>
            )}

            {activeTab === "voice_design" && (
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                  🎨 Voice Design Settings
                </h3>
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Language
                  </label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]"
                  >
                    <option value="Auto">Auto</option>
                    {languages.map((l) => (
                      <option key={l} value={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Voice Description (instruct)
                  </label>
                  <textarea
                    value={instruct}
                    onChange={(e) => setInstruct(e.target.value)}
                    rows={3}
                    placeholder="Describe the voice: e.g. A warm, mature female voice with a gentle British accent…"
                    className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-xl p-3 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-accent)] resize-none"
                  />
                </div>
              </div>
            )}

            {activeTab === "voice_clone" && (
              <div className="glass-card p-5 space-y-4">
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                  🧬 Voice Clone Settings
                </h3>
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Language
                  </label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]"
                  >
                    <option value="Auto">Auto</option>
                    {languages.map((l) => (
                      <option key={l} value={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
                {/* Reference Audio Upload */}
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Reference Audio File
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => setRefAudio(e.target.files?.[0] || null)}
                      className="hidden"
                      id="ref-audio-input"
                    />
                    <label
                      htmlFor="ref-audio-input"
                      className="flex items-center gap-3 w-full bg-[var(--color-bg-input)] border-2 border-dashed border-[var(--color-border)] rounded-xl p-4 cursor-pointer hover:border-[var(--color-accent)] transition-colors"
                    >
                      <span className="text-2xl">📁</span>
                      <div>
                        <p className="text-sm text-[var(--color-text-primary)]">
                          {refAudio ? refAudio.name : "Click to upload reference audio"}
                        </p>
                        <p className="text-xs text-[var(--color-text-muted)]">
                          {refAudio
                            ? `${(refAudio.size / 1024).toFixed(1)} KB`
                            : "WAV, MP3, FLAC supported"}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>
                {/* Reference Text */}
                <div>
                  <label className="text-xs text-[var(--color-text-muted)] mb-1 block">
                    Reference Text (transcript of the audio)
                  </label>
                  <textarea
                    value={refText}
                    onChange={(e) => setRefText(e.target.value)}
                    rows={2}
                    placeholder="Transcript of the reference audio (required for ICL mode)"
                    className="w-full bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-xl p-3 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-accent)] resize-none"
                  />
                </div>
                {/* X-Vector only toggle */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={xVectorOnly}
                    onChange={(e) => setXVectorOnly(e.target.checked)}
                    className="w-4 h-4 rounded accent-[var(--color-accent)]"
                  />
                  <span className="text-sm text-[var(--color-text-secondary)]">
                    X-Vector Only Mode
                  </span>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    (skip ICL, faster but less accurate)
                  </span>
                </label>
              </div>
            )}

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={generating || !text.trim()}
              className={`btn-glow w-full text-base py-3 ${generating ? "animate-pulse-glow" : ""
                }`}
            >
              {generating ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Generating…
                </span>
              ) : (
                "🔊 Generate Speech"
              )}
            </button>
          </div>

          {/* ── Right: Output ── */}
          <div className="space-y-4">
            {/* Audio Player */}
            <div className="glass-card p-5">
              <h3 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">
                🔈 Output
              </h3>
              {audioUrl ? (
                <div className="space-y-3">
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    controls
                    className="w-full"
                  />
                  <a
                    href={audioUrl}
                    download="qwen3_tts_output.wav"
                    className="block text-center text-sm text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] transition-colors"
                  >
                    ⬇ Download WAV
                  </a>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-10 text-[var(--color-text-muted)]">
                  <span className="text-4xl mb-2 opacity-30">🎶</span>
                  <p className="text-sm">Generated audio will appear here</p>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="glass-card p-4 border-[var(--color-danger)]/50">
                <p className="text-sm text-[var(--color-danger)]">⚠ {error}</p>
              </div>
            )}

            {/* Model Info */}
            <div className="glass-card p-5">
              <h3 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">
                ℹ️ Model Info
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Type</span>
                  <span className="text-[var(--color-text-primary)]">
                    {modelType || "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Languages</span>
                  <span className="text-[var(--color-text-primary)]">
                    {languages.length || "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Speakers</span>
                  <span className="text-[var(--color-text-primary)]">
                    {speakers.length || "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Model</span>
                  <span className="text-[var(--color-text-primary)] text-[10px] max-w-[160px] truncate">
                    {currentModel || "—"}
                  </span>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="glass-card p-5">
              <h3 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">
                💡 Tips
              </h3>
              <ul className="space-y-1.5 text-xs text-[var(--color-text-muted)]">
                <li>
                  • <strong>CustomVoice</strong> models use built-in speakers
                </li>
                <li>
                  • <strong>VoiceDesign</strong> models accept natural language voice descriptions
                </li>
                <li>
                  • <strong>Base</strong> models support Voice Clone with reference audio
                </li>
                <li>• Switch model in the top-right dropdown to match your mode</li>
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-[var(--color-border)] px-6 py-3 text-center text-xs text-[var(--color-text-muted)]">
        Qwen3-TTS Server • Powered by Alibaba Qwen Team
      </footer>
    </div>
  );
}
