import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Download,
  FileAudio,
  HardDriveDownload,
  X,
  Mic,
  Pause,
  Play,
  Radio,
  RefreshCcw,
  Sparkles,
  Trash2,
  Upload,
  AudioWaveform
} from "lucide-react";
import "./styles.css";

type VoiceProfile = {
  id: string;
  name: string;
  created_at: string;
  ref_text: string | null;
};

type Job = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  voice_id: string;
  text: string;
  lang_code: "zh" | "en";
  engine: EngineName;
  output_path: string | null;
  duration_seconds: number | null;
  generation_seconds: number | null;
  model: string | null;
  error: string | null;
};

type ModelStatus = {
  key: string;
  repo_id: string;
  status: "idle" | "partial" | "downloading" | "ready" | "failed" | "cancelled";
  error: string | null;
  local_path: string | null;
  download_url: string;
  cache_dir: string;
  manual_dir: string;
  downloaded_bytes: number;
  total_bytes: number | null;
  progress: number | null;
  bytes_per_second: number | null;
  seconds_since_progress: number | null;
  log_path: string;
  current_file: string | null;
  required_files: string[];
  missing_files: string[];
};

type HfTokenStatus = {
  configured: boolean;
  path: string;
  preview: string | null;
};

type HfEndpointStatus = {
  endpoint: string;
  default_endpoint: string;
  path: string;
};

const engineLabels = {
  qwen3_mlx: "Qwen3 MLX 8bit",
  qwen3_torch_cpu: "Qwen3 PyTorch CPU",
  qwen3_torch_cuda: "Qwen3 PyTorch CUDA",
  chatterbox_mlx: "Chatterbox MLX",
  chatterbox_torch_cpu: "Chatterbox PyTorch CPU",
  chatterbox_torch_cuda: "Chatterbox PyTorch CUDA",
  tone: "Tone test"
} as const;

type EngineName = keyof typeof engineLabels;

const allModelKeys = ["qwen3", "chatterbox", "whisper_stt"];

function App() {
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [voiceName, setVoiceName] = useState("My reference voice");
  const [refText, setRefText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [previewPlayable, setPreviewPlayable] = useState(true);
  const [text, setText] = useState("你好，这是一个本地运行的声音克隆工具。");
  const [langCode, setLangCode] = useState<"zh" | "en">("zh");
  const [engine, setEngine] = useState<EngineName>("qwen3_mlx");
  const [job, setJob] = useState<Job | null>(null);
  const [modelStatuses, setModelStatuses] = useState<Record<string, ModelStatus | undefined>>({});
  const [hfTokenStatus, setHfTokenStatus] = useState<HfTokenStatus | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [hfEndpointStatus, setHfEndpointStatus] = useState<HfEndpointStatus | null>(null);
  const [hfEndpoint, setHfEndpoint] = useState("https://hf-mirror.com");
  const [view, setView] = useState<"synthesize" | "models">("synthesize");
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Ready");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    refreshVoices();
    refreshModelStatuses();
    refreshHfTokenStatus();
    refreshHfEndpointStatus();
  }, []);

  useEffect(() => {
    refreshModelStatuses();
  }, [engine, refText, view]);

  useEffect(() => {
    if (!Object.values(modelStatuses).some((status) => status?.status === "downloading")) {
      return;
    }
    const timer = window.setInterval(refreshModelStatuses, 1800);
    return () => window.clearInterval(timer);
  }, [modelStatuses]);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      const next = await fetchJob(job.id);
      setJob(next);
      if (next.status === "succeeded") {
        setMessage("Generation complete");
        window.clearInterval(timer);
      }
      if (next.status === "failed") {
        setMessage(next.error ?? "Generation failed");
        window.clearInterval(timer);
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job]);

  async function refreshVoices() {
    const response = await fetch("/api/voices");
    const data = (await response.json()) as VoiceProfile[];
    setVoices(data);
    if (!selectedVoice && data.length > 0) {
      setSelectedVoice(data[0].id);
    }
  }

  async function refreshAll() {
    setRefreshing(true);
    try {
      await Promise.all([refreshVoices(), refreshModelStatuses(), refreshHfTokenStatus(), refreshHfEndpointStatus()]);
      setMessage("Refreshed");
    } finally {
      setRefreshing(false);
    }
  }

  async function refreshModelStatuses() {
    const keys = view === "models" ? allModelKeys : requiredModelKeys(engine, refText);
    const entries = await Promise.all(
      keys.map(async (key) => {
        const response = await fetch(`/api/models/${key}`);
        return [key, (await response.json()) as ModelStatus] as const;
      })
    );
    setModelStatuses((current) => ({ ...current, ...Object.fromEntries(entries) }));
  }

  async function refreshHfTokenStatus() {
    const response = await fetch("/api/hf-token");
    const data = (await response.json()) as HfTokenStatus;
    setHfTokenStatus(data);
  }

  async function refreshHfEndpointStatus() {
    const response = await fetch("/api/hf-endpoint");
    const data = (await response.json()) as HfEndpointStatus;
    setHfEndpointStatus(data);
    setHfEndpoint(data.endpoint);
  }

  async function saveHfEndpoint() {
    const endpoint = hfEndpoint.trim();
    if (!endpoint) {
      setMessage("Enter a Hugging Face endpoint first");
      return;
    }
    await saveHfEndpointValue(endpoint);
  }

  async function saveHfEndpointValue(endpoint: string) {
    const response = await fetch("/api/hf-endpoint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint })
    });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "Could not save Hugging Face endpoint");
      return;
    }
    const data = (await response.json()) as HfEndpointStatus;
    setHfEndpointStatus(data);
    setHfEndpoint(data.endpoint);
    setMessage("Hugging Face endpoint saved locally");
  }

  async function saveHfToken() {
    const token = hfToken.trim();
    if (!token) {
      setMessage("Paste a Hugging Face token first");
      return;
    }
    const response = await fetch("/api/hf-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token })
    });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "Could not save Hugging Face token");
      return;
    }
    const data = (await response.json()) as HfTokenStatus;
    setHfTokenStatus(data);
    setHfToken("");
    setMessage("Hugging Face token saved locally");
  }

  async function deleteHfToken() {
    const response = await fetch("/api/hf-token", { method: "DELETE" });
    const data = (await response.json()) as HfTokenStatus;
    setHfTokenStatus(data);
    setMessage("Hugging Face token removed");
  }

  async function downloadModel(key: string) {
    setMessage("Starting model download");
    const response = await fetch(`/api/models/${key}/download`, { method: "POST" });
    const data = (await response.json()) as ModelStatus;
    setModelStatuses((current) => ({ ...current, [key]: data }));
  }

  async function cancelModelDownload(key: string) {
    setMessage("Cancelling model download");
    const response = await fetch(`/api/models/${key}/cancel`, { method: "POST" });
    const data = (await response.json()) as ModelStatus;
    setModelStatuses((current) => ({ ...current, [key]: data }));
    setMessage("Model download cancelled");
  }

  async function uploadVoice() {
    if (!file) {
      setMessage("Choose or record an audio file first");
      return;
    }
    setBusy(true);
    setMessage(refText.trim() ? "Uploading and normalizing audio" : "Uploading audio and auto transcribing reference");
    const form = new FormData();
    form.append("file", file);
    form.append("name", voiceName);
    form.append("ref_text", refText);
    const response = await fetch("/api/voices", { method: "POST", body: form });
    setBusy(false);
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "Upload failed");
      return;
    }
    const created = (await response.json()) as VoiceProfile;
    setSelectedVoice(created.id);
    if (!refText.trim() && created.ref_text) {
      setRefText(created.ref_text);
      setMessage("Voice profile saved and reference transcript filled");
    } else {
      setMessage("Voice profile saved");
    }
    await refreshVoices();
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setMessage("Microphone recording is not supported in this browser");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "MicrophoneError";
      setMessage(`${name}: allow microphone access for this site`);
      return;
    }
    chunksRef.current = [];
    const recordingType = getRecordingType();
    const recorder = new MediaRecorder(
      stream,
      recordingType.mimeType ? { mimeType: recordingType.mimeType } : undefined
    );
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => chunksRef.current.push(event.data);
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recordingType.mimeType || "audio/webm" });
      const recordedFile = new File([blob], `recording.${recordingType.extension}`, {
        type: recordingType.mimeType || "audio/webm"
      });
      setFile(recordedFile);
      setPreviewError("");
      setPreviewPlayable(true);
      setRecordedUrl(URL.createObjectURL(blob));
      stream.getTracks().forEach((track) => track.stop());
    };
    recorder.start();
    setRecording(true);
    setMessage("Recording");
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
    setMessage("Recording captured");
  }

  async function createJob() {
    if (!selectedVoice) {
      setMessage("Create or select a voice first");
      return;
    }
    setBusy(true);
    setMessage("Queueing synthesis");
    const response = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice_id: selectedVoice,
        text,
        lang_code: langCode,
        engine,
        ref_text: refText || undefined
      })
    });
    setBusy(false);
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "Could not start synthesis");
      return;
    }
    const created = await response.json();
    const next = await fetchJob(created.id);
    setJob(next);
    setMessage(statusMessage(next));
  }

  async function deleteVoice(voiceId: string) {
    const response = await fetch(`/api/voices/${voiceId}`, { method: "DELETE" });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "Could not delete voice");
      return;
    }
    if (selectedVoice === voiceId) {
      setSelectedVoice("");
    }
    setMessage("Voice deleted");
    await refreshVoices();
  }

  async function fetchJob(id: string) {
    const response = await fetch(`/api/jobs/${id}`);
    return (await response.json()) as Job;
  }

  const outputUrl = job?.status === "succeeded" ? `/api/outputs/${job.id}.wav` : "";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><AudioWaveform size={22} /></div>
          <div>
            <h1>VoiceClone</h1>
            <p>Local voice synthesis workspace</p>
          </div>
        </div>
        <button className="ghost-button" onClick={refreshAll} disabled={refreshing}>
          <RefreshCcw size={17} />
          {refreshing ? "Refreshing" : "Refresh"}
        </button>
      </header>

      <nav className="view-tabs">
        <button className={view === "synthesize" ? "active" : ""} onClick={() => setView("synthesize")}>
          Synthesize
        </button>
        <button className={view === "models" ? "active" : ""} onClick={() => setView("models")}>
          Models
        </button>
      </nav>

      {view === "synthesize" ? <section className="workspace">
        <aside className="source-panel">
          <PanelTitle icon={<Mic size={18} />} title="Voice source" />
          <label className="field">
            Voice name
            <input value={voiceName} onChange={(event) => setVoiceName(event.target.value)} />
          </label>
          <div className="upload-zone">
            <FileAudio size={34} />
            <strong>{file ? file.name : "Drop in a WAV, MP3, M4A, WebM, or record audio"}</strong>
            <span>Use a clean 5-30 second single-speaker sample.</span>
            <input
              type="file"
              accept="audio/*"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </div>
          <div className="record-row">
            <button className="icon-button primary" onClick={recording ? stopRecording : startRecording}>
              {recording ? <Pause size={18} /> : <Radio size={18} />}
              {recording ? "Stop" : "Record"}
            </button>
            <button className="icon-button" onClick={uploadVoice} disabled={busy}>
              <Upload size={18} />
              Save voice
            </button>
          </div>
          {recordedUrl && (
            <>
              {previewPlayable && (
                <audio
                  className="mini-player"
                  src={recordedUrl}
                  controls
                  onError={() => {
                    setPreviewPlayable(false);
                    setPreviewError("Recording captured. This browser cannot preview the format, but Save voice will still upload it.");
                  }}
                />
              )}
              {previewError && <p className="inline-error">{previewError}</p>}
            </>
          )}
          <label className="field">
            Reference transcript
            <textarea
              rows={4}
              value={refText}
              onChange={(event) => setRefText(event.target.value)}
              placeholder="Optional text spoken in the reference audio"
            />
          </label>
          <div className="voice-list">
            <span className="section-label">Saved voices</span>
            {voices.length === 0 && <p className="empty">No voice profiles yet.</p>}
            {voices.map((voice) => (
              <div key={voice.id} className={voice.id === selectedVoice ? "voice-row selected" : "voice-row"}>
                <button className="voice-select" onClick={() => setSelectedVoice(voice.id)}>
                  <span>{voice.name}</span>
                  <small>{new Date(voice.created_at).toLocaleString()}</small>
                </button>
                <audio className="voice-player" src={`/api/voices/${voice.id}/audio`} controls />
                <button className="delete-voice" onClick={() => deleteVoice(voice.id)} title="Delete voice">
                  <Trash2 size={16} />
                  Delete
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="synthesis-panel">
          <PanelTitle icon={<Sparkles size={19} />} title="Text synthesis" />
          <div className="control-grid">
            <div className="segmented">
              <button className={langCode === "zh" ? "active" : ""} onClick={() => setLangCode("zh")}>Chinese</button>
              <button className={langCode === "en" ? "active" : ""} onClick={() => setLangCode("en")}>English</button>
            </div>
            <div className="model-select-row">
              <select value={engine} onChange={(event) => setEngine(event.target.value as EngineName)}>
                {Object.entries(engineLabels).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
              <button className="ghost-button compact" onClick={() => setView("models")}>Models</button>
            </div>
          </div>
          <textarea
            className="script-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Enter text to synthesize"
          />
          <div className="action-row">
            <button className="generate-button" onClick={createJob} disabled={busy}>
              <Play size={18} />
              Generate WAV
            </button>
            <div className="status">
              <span className={job?.status ?? "idle"} />
              {message}
            </div>
          </div>

          <div className="output-panel">
            <div>
              <span className="section-label">Output</span>
              <h2>{job ? job.status : "Waiting for synthesis"}</h2>
              {job?.model && <p>{job.model}</p>}
              {job?.generation_seconds && <p>Generated in {job.generation_seconds}s</p>}
            </div>
            {outputUrl ? (
              <div className="output-actions">
                <audio src={outputUrl} controls />
                <a className="download-button" href={outputUrl} download>
                  <Download size={18} />
                  Download WAV
                </a>
              </div>
            ) : (
              <div className="wave-placeholder">
                {Array.from({ length: 34 }, (_, index) => (
                  <span key={index} style={{ height: `${18 + ((index * 17) % 46)}px` }} />
                ))}
              </div>
            )}
          </div>
        </section>
      </section> : (
        <section className="models-page">
          <div className="models-header">
            <PanelTitle icon={<HardDriveDownload size={19} />} title="Model manager" />
            <button className="ghost-button compact" onClick={() => refreshModelStatuses()} disabled={refreshing}>
              Refresh
            </button>
          </div>
          <div className="hf-token-card">
            <div>
              <span className="section-label">Hugging Face endpoint</span>
              <p>
                Downloads use <code>{hfEndpointStatus?.endpoint ?? "https://hf-mirror.com"}</code>.
                This is equivalent to setting <code>HF_ENDPOINT</code> for model downloads.
              </p>
            </div>
            <div className="hf-token-controls">
              <input
                value={hfEndpoint}
                onChange={(event) => setHfEndpoint(event.target.value)}
                placeholder="https://hf-mirror.com"
                autoComplete="off"
              />
              <button className="icon-button" onClick={saveHfEndpoint}>
                Save endpoint
              </button>
              <button className="ghost-button compact" onClick={() => saveHfEndpointValue("https://hf-mirror.com")}>
                Use mirror
              </button>
              <button className="ghost-button compact" onClick={() => saveHfEndpointValue("https://huggingface.co")}>
                Use official
              </button>
            </div>
          </div>
          <div className="hf-token-card">
            <div>
              <span className="section-label">Hugging Face token</span>
              <p>
                {hfTokenStatus?.configured
                  ? `Using local token ${hfTokenStatus.preview} from ${hfTokenStatus.path}`
                  : "Optional. Save a local token to use authenticated Hugging Face downloads."}
              </p>
            </div>
            <div className="hf-token-controls">
              <input
                type="password"
                value={hfToken}
                onChange={(event) => setHfToken(event.target.value)}
                placeholder="hf_..."
                autoComplete="off"
              />
              <button className="icon-button" onClick={saveHfToken}>
                Save token
              </button>
              {hfTokenStatus?.configured && (
                <button className="icon-button danger" onClick={deleteHfToken}>
                  Remove
                </button>
              )}
            </div>
          </div>
          <div className="model-grid">
            <article className="model-card">
              <ModelStatusPanel
                status={modelStatuses.qwen3}
                fallbackRepo="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
                onDownload={() => downloadModel("qwen3")}
                onCancel={() => cancelModelDownload("qwen3")}
              />
            </article>
            <article className="model-card">
              <ModelStatusPanel
                status={modelStatuses.chatterbox}
                fallbackRepo="mlx-community/chatterbox-fp16"
                onDownload={() => downloadModel("chatterbox")}
                onCancel={() => cancelModelDownload("chatterbox")}
              />
            </article>
            <article className="model-card">
              <ModelStatusPanel
                status={modelStatuses.whisper_stt}
                fallbackRepo="mlx-community/whisper-large-v3-turbo-asr-fp16"
                onDownload={() => downloadModel("whisper_stt")}
                onCancel={() => cancelModelDownload("whisper_stt")}
              />
            </article>
            <article className="model-card">
              <PackageModelPanel
                title="Qwen3 PyTorch CPU/CUDA"
                model="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                engines="qwen3_torch_cpu, qwen3_torch_cuda"
                installCommand="uv sync --extra dev --extra qwen-torch && uv pip install qwen-tts"
                note="Use this on Windows/Linux when you want Qwen without MLX. CUDA requires a CUDA-enabled PyTorch install."
              />
            </article>
            <article className="model-card">
              <PackageModelPanel
                title="Chatterbox PyTorch CPU/CUDA"
                model="chatterbox-tts default package model"
                engines="chatterbox_torch_cpu, chatterbox_torch_cuda"
                installCommand="uv sync --extra dev --extra chatterbox-torch && uv pip install chatterbox-tts"
                note="Use this on Windows/Linux when you want Chatterbox without MLX. Install it in a separate environment from qwen-tts."
              />
            </article>
          </div>
        </section>
      )}
    </main>
  );
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function ModelStatusPanel({
  status,
  fallbackRepo,
  onDownload,
  onCancel
}: {
  status: ModelStatus | undefined;
  fallbackRepo: string;
  onDownload: () => void;
  onCancel: () => void;
}) {
  const progress = status?.progress ?? (status?.status === "downloading" ? 12 : 0);

  return (
    <>
      <div className="model-meta">
        <strong>{status?.repo_id ?? fallbackRepo}</strong>
        <span>{status ? modelLabel(status) : "Checking model cache"}</span>
        {status?.status === "downloading" && (
          <span>
            {status.current_file ? `${status.current_file} · ` : ""}
            Speed {formatBytes(status.bytes_per_second ?? 0)}/s
            {status.seconds_since_progress && status.seconds_since_progress > 10
              ? ` · waiting ${Math.round(status.seconds_since_progress)}s for new bytes`
              : ""}
          </span>
        )}
      </div>
      <div className={status?.status === "downloading" && !status.progress ? "progress-track indeterminate" : "progress-track"}>
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="model-actions">
        <button className="icon-button" onClick={onDownload} disabled={status?.status === "downloading" || status?.status === "ready"}>
          <Download size={18} />
          {status?.status === "ready" ? "Ready" : status?.status === "downloading" ? "Downloading" : "Download model"}
        </button>
        {status?.status === "downloading" && (
          <button className="icon-button danger" onClick={onCancel}>
            <X size={18} />
            Cancel
          </button>
        )}
        <a className="manual-link" href={status?.download_url ?? `https://huggingface.co/${fallbackRepo}`} target="_blank" rel="noreferrer">
          Manual download
        </a>
      </div>
      <p className="model-help">
        Manual files should live under <code>{status?.manual_dir ?? `~/.cache/huggingface/hub/models--${fallbackRepo.replace("/", "--")}`}</code>.
      </p>
      <div className="required-files">
        <span className="section-label">Required files</span>
        <p>
          {status?.missing_files?.length
            ? `Missing: ${status.missing_files.join(", ")}`
            : status
              ? "All required files are present."
              : "Loading required file list."}
        </p>
      </div>
      {status?.error && <p className="inline-error">{status.error}</p>}
    </>
  );
}

function PackageModelPanel({
  title,
  model,
  engines,
  installCommand,
  note
}: {
  title: string;
  model: string;
  engines: string;
  installCommand: string;
  note: string;
}) {
  return (
    <>
      <div className="model-meta">
        <strong>{title}</strong>
        <span>{model}</span>
      </div>
      <div className="package-model-info">
        <span className="section-label">Engines</span>
        <p>{engines}</p>
        <span className="section-label">Install</span>
        <code>{installCommand}</code>
        <p>{note}</p>
      </div>
    </>
  );
}

function statusMessage(job: Job) {
  if (job.status === "succeeded") {
    return "Generation complete";
  }
  if (job.status === "failed") {
    return job.error ?? "Generation failed";
  }
  if (job.status === "queued") {
    return "Synthesis queued";
  }
  return "Synthesis running";
}

function modelLabel(model: ModelStatus) {
  const downloaded = formatBytes(model.downloaded_bytes);
  const total = model.total_bytes ? formatBytes(model.total_bytes) : "unknown size";
  if (model.status === "ready") {
    return `Ready at ${model.local_path}`;
  }
  if (model.status === "downloading") {
    return `${model.progress ?? 0}% downloaded (${downloaded} / ${total})`;
  }
  if (model.status === "partial") {
    return `Partially downloaded. Continue download to finish (${downloaded} / ${total})`;
  }
  if (model.status === "failed") {
    return `Download failed (${downloaded} / ${total})`;
  }
  if (model.status === "cancelled") {
    return `Download cancelled (${downloaded} / ${total})`;
  }
  return `Not downloaded (${downloaded} / ${total})`;
}

function formatBytes(bytes: number) {
  if (bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function getRecordingType() {
  const candidates = [
    { mimeType: "audio/mp4", extension: "m4a" },
    { mimeType: "audio/webm;codecs=opus", extension: "webm" },
    { mimeType: "audio/webm", extension: "webm" },
    { mimeType: "", extension: "webm" }
  ];
  return candidates.find((candidate) => !candidate.mimeType || MediaRecorder.isTypeSupported(candidate.mimeType)) ?? candidates[3];
}

function requiredModelKeys(engine: EngineName, refText: string) {
  const keys: string[] = [];
  if (engine === "qwen3_mlx") {
    keys.push("qwen3");
  }
  if (engine === "chatterbox_mlx") {
    keys.push("chatterbox");
  }
  if (engine !== "tone" && !refText.trim()) {
    keys.push("whisper_stt");
  }
  return keys;
}

const rootElement = document.getElementById("root")!;
const browserWindow = window as Window & {
  voiceCloneRoot?: ReturnType<typeof createRoot>;
};
browserWindow.voiceCloneRoot ??= createRoot(rootElement);
browserWindow.voiceCloneRoot.render(<App />);
