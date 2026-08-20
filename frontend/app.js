/**
 * SV2 VOICE RAG - Continuous Scrolling Neo-Brutalist Pop-Art Web Application
 * ai4bharat/MSMARCO-XI Multilingual Pipeline
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // API Configuration (Supports relative path or remote backend)
  // =========================================================================
  const API_BASE = window.API_BASE_URL || "";

  // =========================================================================
  // State & Global Variables
  // =========================================================================
  let audioContext = null;
  let analyser = null;
  let microphone = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let animationFrameId = null;
  let speechRecognition = null;
  let lastAudioBlob = null;
  let lastResponseAnswer = "";
  let activeStrategy = "indic_semantic";

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const navButtons = document.querySelectorAll(".slide-nav-btn");
  const sections = document.querySelectorAll(".page-section");

  // Slide 2 Inputs
  const btnRecord = document.getElementById("btn-record");
  const recordBtnText = document.getElementById("record-btn-text");
  const audioFileInput = document.getElementById("audio-file-input");
  const btnRunQuery = document.getElementById("btn-run-query");
  const transcriptInput = document.getElementById("transcript-input");
  const languageSelect = document.getElementById("language-select");
  const sttProviderSelect = document.getElementById("stt-provider-select");
  const activeStrategySelect = document.getElementById("active-strategy-select");
  const detectedScriptLabel = document.getElementById("detected-script-label");
  const samplePromptButtons = document.querySelectorAll(".sample-prompt-btn");

  // Waveform Canvas
  const canvas = document.getElementById("waveform-canvas");
  const canvasCtx = canvas.getContext("2d");
  const waveformStatusText = document.getElementById("waveform-status-text");

  // Slide 3 Outputs
  const answerBox = document.getElementById("answer-box");
  const answerGuardBadge = document.getElementById("answer-guardrail-badge");
  const ttsBar = document.getElementById("tts-bar");
  const btnPlayTts = document.getElementById("btn-play-tts");
  const factsSection = document.getElementById("facts-section");
  const factsList = document.getElementById("facts-list");
  const citationsSection = document.getElementById("citations-section");
  const citationsList = document.getElementById("citations-list");

  // Waterfall
  const wfTotalVal = document.getElementById("waterfall-total-val");
  const wfSlaPill = document.getElementById("waterfall-sla-pill");
  const wfBarStt = document.getElementById("wf-bar-stt");
  const wfBarGuardIn = document.getElementById("wf-bar-guard-in");
  const wfBarEmbed = document.getElementById("wf-bar-embed");
  const wfBarRetrieval = document.getElementById("wf-bar-retrieval");
  const wfBarTools = document.getElementById("wf-bar-tools");
  const wfBarGen = document.getElementById("wf-bar-gen");
  const wfBarGuardOut = document.getElementById("wf-bar-guard-out");

  const wfLegendStt = document.getElementById("wf-legend-stt");
  const wfLegendEmbed = document.getElementById("wf-legend-embed");
  const wfLegendRetrieval = document.getElementById("wf-legend-retrieval");
  const wfLegendGen = document.getElementById("wf-legend-gen");
  const wfLegendGuard = document.getElementById("wf-legend-guard");

  // Guardrail Badges
  const gStage1Badge = document.getElementById("g-stage1-badge");
  const gStage2Badge = document.getElementById("g-stage2-badge");
  const gStage3Badge = document.getElementById("g-stage3-badge");

  // =========================================================================
  // ScrollSpy & Navigation Tracking
  // =========================================================================
  const observerOptions = {
    root: null,
    rootMargin: "-20% 0px -60% 0px",
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute("id");
        navButtons.forEach(btn => {
          if (btn.getAttribute("href") === `#${id}`) {
            btn.classList.add("active");
          } else {
            btn.classList.remove("active");
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => {
    observer.observe(section);
  });

  // =========================================================================
  // Language Autodetection Logic
  // =========================================================================
  function detectScript(text) {
    if (!text) return { script: "Devanagari (Hindi)", code: "hi-IN" };
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code >= 0x0900 && code <= 0x097F) return { script: "Devanagari (Hindi)", code: "hi-IN" };
      if (code >= 0x0C00 && code <= 0x0C7F) return { script: "Telugu (తెలుగు)", code: "te-IN" };
      if (code >= 0x0B80 && code <= 0x0BFF) return { script: "Tamil (தமிழ்)", code: "ta-IN" };
      if (code >= 0x0980 && code <= 0x09FF) return { script: "Bengali (বাংলা)", code: "bn-IN" };
    }
    return { script: "Latin (English)", code: "en-IN" };
  }

  transcriptInput.addEventListener("input", () => {
    const info = detectScript(transcriptInput.value);
    detectedScriptLabel.textContent = info.script;
  });

  // =========================================================================
  // Waveform & Audio Recording
  // =========================================================================
  function drawNeoWaveform() {
    if (!analyser) return;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = "#000000";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 3;
    canvasCtx.strokeStyle = "#FFE600";
    canvasCtx.beginPath();

    const sliceWidth = (canvas.width * 1.0) / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();

    animationFrameId = requestAnimationFrame(drawNeoWaveform);
  }

  function drawIdleWaveform() {
    canvasCtx.fillStyle = "#000000";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "rgba(255, 230, 0, 0.4)";
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
  drawIdleWaveform();

  // Web Speech API for real-time speech preview
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;

    speechRecognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          transcriptInput.value = event.results[i][0].transcript;
          const info = detectScript(transcriptInput.value);
          detectedScriptLabel.textContent = info.script;
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      if (interim) {
        waveformStatusText.textContent = `HEARING: "${interim.toUpperCase()}"`;
      }
    };
  }

  // Record Button Click
  btnRecord.addEventListener("click", async () => {
    if (!isRecording) {
      // Start Recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);

        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
          lastAudioBlob = new Blob(audioChunks, { type: "audio/webm" });
        };
        mediaRecorder.start();

        if (speechRecognition) {
          speechRecognition.lang = languageSelect.value;
          speechRecognition.start();
        }

        isRecording = true;
        btnRecord.classList.add("recording");
        recordBtnText.textContent = "STOP RECORDING & TRANSCRIBE";
        waveformStatusText.textContent = "LISTENING... SPEAK CLEARLY";
        drawNeoWaveform();
      } catch (err) {
        console.error("Mic error:", err);
        alert("Microphone access unavailable or blocked. You can still type queries or select sample prompts!");
      }
    } else {
      // Stop Recording
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      if (speechRecognition) {
        try { speechRecognition.stop(); } catch (e) {}
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      isRecording = false;
      btnRecord.classList.remove("recording");
      recordBtnText.textContent = "START VOICE RECORDING";
      waveformStatusText.textContent = "AUDIO CAPTURED. CLICK 'EXECUTE RAG' TO VIEW OUTPUT BELOW!";
      drawIdleWaveform();
    }
  });

  // Audio File Upload
  audioFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      lastAudioBlob = file;
      waveformStatusText.textContent = `LOADED AUDIO FILE: ${file.name.toUpperCase()}`;
    }
  });

  // Sample Prompt Buttons Click
  samplePromptButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      const l = btn.getAttribute("data-lang");
      transcriptInput.value = q;
      languageSelect.value = l;
      const info = detectScript(q);
      detectedScriptLabel.textContent = info.script;
      waveformStatusText.textContent = `SELECTED PROMPT: "${q.substring(0, 30)}..."`;
    });
  });

  // Strategy Switcher
  activeStrategySelect.addEventListener("change", async (e) => {
    activeStrategy = e.target.value;
    try {
      await fetch(`${API_BASE}/api/switch-strategy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: activeStrategy })
      });
    } catch (err) {
      console.error("Strategy switch error:", err);
    }
  });

  // =========================================================================
  // Execute Pipeline (Voice / Text RAG) & Smooth Scroll to Output Section
  // =========================================================================
  async function executePipeline() {
    const queryText = transcriptInput.value.trim();
    const lang = languageSelect.value;
    const sttProv = sttProviderSelect.value;

    if (!queryText && !lastAudioBlob) {
      alert("Please record audio, upload an audio file, or choose a sample prompt!");
      return;
    }

    // Smooth scroll down to Output Section (#slide-3)
    const outSection = document.getElementById("slide-3");
    if (outSection) {
      outSection.scrollIntoView({ behavior: "smooth" });
    }

    answerBox.innerHTML = `
      <div class="answer-placeholder">
        <span class="placeholder-icon">⚡</span>
        <p>EXECUTING SV2 VOICE RAG PIPELINE ACROSS MSMARCO-XI...</p>
      </div>
    `;
    answerGuardBadge.className = "neo-badge-guardrail guardrail-idle";
    answerGuardBadge.textContent = "PROCESSING...";

    const formData = new FormData();
    if (lastAudioBlob) {
      formData.append("audio", lastAudioBlob, "voice_input.webm");
    }
    if (queryText) {
      formData.append("client_transcript", queryText);
    }
    formData.append("language", lang);
    formData.append("stt_provider", sttProv);
    formData.append("strategy", activeStrategy);
    formData.append("llm_provider", "fast_synthesizer");

    try {
      const resp = await fetch(`${API_BASE}/api/voice-query`, {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        throw new Error(`Pipeline error: HTTP ${resp.status}`);
      }

      const data = await resp.json();
      renderOutputResults(data);
    } catch (err) {
      console.error(err);
      answerBox.innerHTML = `<div style="color: #FF007A; font-weight:800;">ERROR EXECUTING PIPELINE: ${err.message}</div>`;
      answerGuardBadge.className = "neo-badge-guardrail guardrail-block";
      answerGuardBadge.textContent = "ERROR";
    }
  }

  btnRunQuery.addEventListener("click", executePipeline);

  // =========================================================================
  // Render Output Results & Telemetry Waterfall
  // =========================================================================
  function renderOutputResults(data) {
    lastResponseAnswer = data.answer;

    // Guardrail Badge
    if (data.safety_verdict === "PASSED") {
      answerGuardBadge.className = "neo-badge-guardrail guardrail-pass";
      answerGuardBadge.textContent = "PASSED ✓";
      gStage1Badge.className = "radar-status status-pass";
      gStage1Badge.textContent = "PASSED ✓";
      gStage2Badge.className = "radar-status status-pass";
      gStage2Badge.textContent = "PASSED ✓";
      gStage3Badge.className = "radar-status status-pass";
      gStage3Badge.textContent = "PASSED ✓";
    } else {
      answerGuardBadge.className = "neo-badge-guardrail guardrail-block";
      answerGuardBadge.textContent = data.safety_verdict;
      if (data.safety_verdict.includes("INJECTION") || data.safety_verdict.includes("TOXIC")) {
        gStage1Badge.className = "radar-status status-block";
        gStage1Badge.textContent = "BLOCKED ✗";
      } else if (data.safety_verdict.includes("OFF_TOPIC")) {
        gStage2Badge.className = "radar-status status-block";
        gStage2Badge.textContent = "OFF-TOPIC ✗";
      } else {
        gStage3Badge.className = "radar-status status-block";
        gStage3Badge.textContent = "UNGROUNDED ✗";
      }
    }

    // Answer Content
    answerBox.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
    ttsBar.style.display = "flex";

    // Verified Claims
    if (data.grounded_facts && data.grounded_facts.length > 0) {
      factsSection.style.display = "block";
      factsList.innerHTML = data.grounded_facts.map(f => `<li>${escapeHtml(f)}</li>`).join("");
    } else {
      factsSection.style.display = "none";
    }

    // Citations
    if (data.citations && data.citations.length > 0) {
      citationsSection.style.display = "block";
      citationsList.innerHTML = data.citations.map(c => `
        <div class="citation-card-neo">
          <div class="citation-header-neo">
            <span>[#${c.rank}] LANG: ${c.language.toUpperCase()} | STRATEGY: ${c.strategy}</span>
            <span>SCORE: ${(c.score * 100).toFixed(1)}%</span>
          </div>
          <p>${escapeHtml(c.snippet)}</p>
        </div>
      `).join("");
    } else {
      citationsSection.style.display = "none";
    }

    // Waterfall Bars & Latency Profiling
    const prof = data.latency_profile || {};
    const totalMs = prof.total_pipeline_ms || 1.0;
    wfTotalVal.textContent = `${totalMs.toFixed(1)} ms`;

    if (totalMs < 200.0) {
      wfSlaPill.textContent = "TARGET: < 200MS SLA PASS ✓";
      wfSlaPill.style.background = "#10b981";
    } else {
      wfSlaPill.textContent = "TARGET: > 200MS EXCEEDED ✗";
      wfSlaPill.style.background = "#FF007A";
    }

    const sttPct = Math.max(1, (prof.stt_ms / totalMs) * 100);
    const guardInPct = Math.max(1, (prof.input_guardrail_ms / totalMs) * 100);
    const embedPct = Math.max(1, (prof.embedding_ms / totalMs) * 100);
    const retPct = Math.max(1, (prof.retrieval_ms / totalMs) * 100);
    const toolsPct = Math.max(1, (prof.tool_calls_ms / totalMs) * 100);
    const genPct = Math.max(1, (prof.llm_generation_ms / totalMs) * 100);
    const guardOutPct = Math.max(1, (prof.output_guardrail_ms / totalMs) * 100);

    wfBarStt.style.width = `${sttPct}%`;
    wfBarGuardIn.style.width = `${guardInPct}%`;
    wfBarEmbed.style.width = `${embedPct}%`;
    wfBarRetrieval.style.width = `${retPct}%`;
    wfBarTools.style.width = `${toolsPct}%`;
    wfBarGen.style.width = `${genPct}%`;
    wfBarGuardOut.style.width = `${guardOutPct}%`;

    wfLegendStt.textContent = `${prof.stt_ms.toFixed(1)}ms`;
    wfLegendEmbed.textContent = `${prof.embedding_ms.toFixed(1)}ms`;
    wfLegendRetrieval.textContent = `${prof.retrieval_ms.toFixed(1)}ms`;
    wfLegendGen.textContent = `${prof.llm_generation_ms.toFixed(1)}ms`;
    wfLegendGuard.textContent = `${prof.output_guardrail_ms.toFixed(1)}ms`;
  }

  // TTS Speech Playback
  btnPlayTts.addEventListener("click", () => {
    if (!lastResponseAnswer) return;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(lastResponseAnswer);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  });

  // Utility
  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
