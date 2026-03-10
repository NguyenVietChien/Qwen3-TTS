// Qwen3-TTS API service layer

const API_BASE = "/api";

// ─── Types ───────────────────────────────────────────────────────────

export interface ModelVariant {
  id: string;
  name: string;
  type: "custom_voice" | "voice_design" | "base";
}

export interface ModelListResponse {
  models: ModelVariant[];
  current: string;
}

export interface ModelInfoResponse {
  model_type: string | null;
  supported_languages: string[] | null;
  supported_speakers: string[] | null;
  status: string;
}

export interface GenerationParams {
  top_k?: number;
  top_p?: number;
  temperature?: number;
  repetition_penalty?: number;
  max_new_tokens?: number;
}

// ─── API functions ───────────────────────────────────────────────────

export async function fetchModels(): Promise<ModelListResponse> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  return res.json();
}

export async function loadModel(modelPath: string) {
  const res = await fetch(`${API_BASE}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: modelPath }),
  });
  if (!res.ok) throw new Error(`Failed to load model: ${res.statusText}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/tts/info`);
  if (!res.ok) throw new Error(`Failed to fetch info: ${res.statusText}`);
  return res.json();
}

export async function generateTTS(params: {
  text: string;
  language: string;
  mode: "custom_voice" | "voice_design";
  speaker?: string;
  instruct?: string;
  generation_params?: GenerationParams;
}): Promise<Blob> {
  const res = await fetch(`${API_BASE}/tts/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Generation failed");
  }
  return res.blob();
}

export async function generateVoiceClone(params: {
  text: string;
  language: string;
  refAudio: File;
  refText?: string;
  xVectorOnly?: boolean;
  top_k?: number;
  top_p?: number;
  temperature?: number;
  max_new_tokens?: number;
}): Promise<Blob> {
  const form = new FormData();
  form.append("text", params.text);
  form.append("language", params.language);
  form.append("ref_audio", params.refAudio);
  if (params.refText) form.append("ref_text", params.refText);
  if (params.xVectorOnly !== undefined)
    form.append("x_vector_only", String(params.xVectorOnly));
  if (params.top_k !== undefined) form.append("top_k", String(params.top_k));
  if (params.top_p !== undefined) form.append("top_p", String(params.top_p));
  if (params.temperature !== undefined)
    form.append("temperature", String(params.temperature));
  if (params.max_new_tokens !== undefined)
    form.append("max_new_tokens", String(params.max_new_tokens));

  const res = await fetch(`${API_BASE}/tts/voice-clone`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Voice clone failed");
  }
  return res.blob();
}
