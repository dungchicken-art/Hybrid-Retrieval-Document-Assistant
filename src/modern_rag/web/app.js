const healthText = document.getElementById("health-text");
const statusDot = document.getElementById("status-dot");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadOutput = document.getElementById("upload-output");
const ingestButton = document.getElementById("ingest-button");
const ingestOutput = document.getElementById("ingest-output");
const askForm = document.getElementById("ask-form");
const queryInput = document.getElementById("query-input");
const answerWarning = document.getElementById("answer-warning");
const answerOutput = document.getElementById("answer-output");
const citationsOutput = document.getElementById("citations-output");
const refreshSourcesButton = document.getElementById("refresh-sources");
const sourcesOutput = document.getElementById("sources-output");

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail || "Request failed.";
    throw new Error(detail);
  }

  return data;
}

async function refreshHealth() {
  try {
    const data = await request("/health");
    const parts = [
      data.index_ready ? "index ready" : "index missing",
      data.llm_configured ? "LLM configured" : "retrieval-only mode",
    ];

    healthText.textContent = parts.join(" | ");
    statusDot.classList.add("ready");
  } catch (error) {
    healthText.textContent = error.message;
  }
}

async function refreshSources() {
  try {
    const sources = await request("/sources");
    if (!sources.length) {
      sourcesOutput.textContent = "No indexed sources yet.";
      return;
    }

    sourcesOutput.innerHTML = sources
      .map(
        (source) => `
          <article class="source-item">
            <strong>${escapeHtml(source.source)}</strong>
            <span>${source.chunk_count} chunks indexed</span>
          </article>
        `,
      )
      .join("");
  } catch (error) {
    sourcesOutput.textContent = error.message;
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    uploadOutput.textContent = "Choose at least one file first.";
    return;
  }

  const formData = new FormData();
  for (const file of fileInput.files) {
    formData.append("files", file);
  }

  uploadOutput.textContent = "Uploading files...";

  try {
    const data = await request("/upload", {
      method: "POST",
      body: formData,
    });
    uploadOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    uploadOutput.textContent = error.message;
  }
});

ingestButton.addEventListener("click", async () => {
  ingestButton.disabled = true;
  ingestOutput.textContent = "Running ingestion. The first run may take a while on CPU...";

  try {
    const data = await request("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    ingestOutput.textContent = JSON.stringify(data, null, 2);
    await refreshHealth();
    await refreshSources();
  } catch (error) {
    ingestOutput.textContent = error.message;
  } finally {
    ingestButton.disabled = false;
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) {
    answerOutput.textContent = "Enter a question first.";
    return;
  }

  answerOutput.textContent = "Retrieving context and generating answer...";
  answerWarning.hidden = true;
  answerWarning.textContent = "";
  citationsOutput.textContent = "Loading citations...";

  try {
    const data = await request("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    answerOutput.textContent = data.answer;
    if (data.warning) {
      answerWarning.hidden = false;
      answerWarning.textContent = data.warning;
    }
    if (!data.citations.length) {
      citationsOutput.textContent = "No citations returned.";
      return;
    }

    citationsOutput.innerHTML = data.citations
      .map(
        (citation) => `
          <article class="citation-item">
            <strong>${escapeHtml(citation.source)}</strong>
            <div>Chunk: ${escapeHtml(citation.chunk_id)}</div>
            <div>Score: ${citation.score.toFixed(4)}</div>
            <p>${escapeHtml(citation.excerpt)}</p>
          </article>
        `,
      )
      .join("");
  } catch (error) {
    answerOutput.textContent = error.message;
    answerWarning.hidden = true;
    answerWarning.textContent = "";
    citationsOutput.textContent = "No citations available.";
  }
});

refreshSourcesButton.addEventListener("click", refreshSources);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

refreshHealth();
refreshSources();
