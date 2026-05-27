(function () {
  const api = "http://127.0.0.1:7861/api";
  let resolved = null;
  let searchCursor = null;
  let currentItems = [];
  let selectedIndex = -1;
  let loadingMore = false;
  let viewMode = "explorer";
  let currentDetail = null;
  let mediaItems = [];
  let mediaIndex = 0;
  let explorerScrollTop = 0;
  let suppressExplorerScrollCapture = false;
  let activeSearchController = null;
  let searchRunId = 0;
  let searchDebounceTimer = null;
  const tokenStorageKey = "nexus_civitai_api_key";
  const downloadJobs = new Map();
  const lazyPageSize = 10;
  let modalVideoObserver = null;

  function el(id) {
    return document.getElementById(id);
  }

  function html(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function stripHtml(value) {
    const node = document.createElement("div");
    node.innerHTML = String(value || "");
    return node.textContent || node.innerText || "";
  }

  function status(text) {
    const node = el("civitaiStatusText");
    if (node) node.innerText = text;
  }

  function normalizedCivitaiQuery(value) {
    return String(value || "")
      .normalize("NFKC")
      .replace(/^#+/, "")
      .replace(/[,_]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function queueSearch() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      search(false).catch((error) => {
        if (error?.name === "AbortError") return;
        status("Search failed.");
      });
    }, 360);
  }

  function explorerScroller() {
    return el("civitaiExplorerScroll");
  }

  function rememberExplorerScroll() {
    const node = explorerScroller();
    if (node) explorerScrollTop = node.scrollTop;
  }

  function scrollExplorerTo(value) {
    const node = explorerScroller();
    if (!node) return;
    const target = Math.max(0, Number(value) || 0);
    suppressExplorerScrollCapture = true;
    requestAnimationFrame(() => {
      node.scrollTop = target;
      requestAnimationFrame(() => {
        node.scrollTop = target;
        suppressExplorerScrollCapture = false;
      });
    });
  }

  function restoreExplorerScroll() {
    scrollExplorerTo(explorerScrollTop);
  }

  function wireControls() {
    const blur = el("civitaiBlurMatureToggle");
    const show = el("civitaiNsfwToggle");
    const base = el("civitaiBaseModelFilter");
    const type = el("civitaiTypeFilter");
    const sort = el("civitaiSortFilter");
    const searchInput = el("civitaiUrlInput");
    wireTokenPersistence();
    if (blur && !blur.dataset.nexusWired) {
      blur.dataset.nexusWired = "1";
      blur.addEventListener("change", () => {
        if (viewMode === "detail" && currentDetail) renderResult(currentDetail.data, currentDetail.downloaded);
        else renderSearch(currentItems, false);
        if (el("civitaiMediaModal") && !el("civitaiMediaModal").classList.contains("hidden")) renderMediaModal();
      });
    }
    if (show && !show.dataset.nexusWired) {
      show.dataset.nexusWired = "1";
      show.addEventListener("change", () => {
        if (viewMode === "detail" && currentDetail) renderResult(currentDetail.data, currentDetail.downloaded);
        else search(false).catch(() => status("Search failed."));
        if (el("civitaiMediaModal") && !el("civitaiMediaModal").classList.contains("hidden")) closeMediaModal();
      });
    }
    [base, type, sort].forEach((control) => {
      if (!control || control.dataset.nexusWired) return;
      control.dataset.nexusWired = "1";
      control.addEventListener("change", () => search(false).catch(() => status("Search failed.")));
    });
    updateTypeButtons();
    if (searchInput && !searchInput.dataset.nexusWired) {
      searchInput.dataset.nexusWired = "1";
      searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") resolve().catch(() => status("Submit failed."));
      });
      searchInput.addEventListener("input", () => {
        if (isCivitaiUrl(searchInput.value)) return;
        queueSearch();
      });
    }
  }

  function updateTypeButtons() {
    const selected = el("civitaiTypeFilter")?.value || "";
    document.querySelectorAll("[data-civitai-type]").forEach((button) => {
      const active = button.getAttribute("data-civitai-type") === selected;
      button.className = active
        ? "bg-nexus-hover text-white px-3 py-1 text-[9px] font-bold uppercase"
        : "text-nexus-muted hover:text-white px-3 py-1 text-[9px] font-bold uppercase";
    });
  }

  function setTypeFilter(value) {
    const select = el("civitaiTypeFilter");
    if (select) select.value = value;
    updateTypeButtons();
    search(false).catch(() => status("Search failed."));
  }

  async function post(path, body, signal = null) {
    const response = await fetch(api + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }

  async function get(path) {
    const response = await fetch(api + path);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }

  function savedToken() {
    return localStorage.getItem(tokenStorageKey) || "";
  }

  function wireTokenPersistence() {
    const token = el("civitaiTokenInput");
    if (!token) return;
    if (!token.value && savedToken()) token.value = savedToken();
    if (token.dataset.nexusTokenWired) return;
    token.dataset.nexusTokenWired = "1";
    token.addEventListener("input", () => {
      const value = token.value.trim();
      if (value) localStorage.setItem(tokenStorageKey, value);
      else localStorage.removeItem(tokenStorageKey);
    });
  }

  function basePayload() {
    return {
      token: el("civitaiTokenInput")?.value?.trim() || savedToken() || null,
      target_kind: el("civitaiTargetSelect")?.value || "auto",
      preset: window.activePreset || localStorage.getItem("nexus_active_preset") || "Anima",
      save_preview: !!el("civitaiPreviewToggle")?.checked,
    };
  }

  function formPayload() {
    const value = el("civitaiUrlInput")?.value?.trim() || "";
    return {
      ...basePayload(),
      url: isCivitaiUrl(value) ? value : "",
    };
  }

  function isCivitaiUrl(value) {
    return /^https?:\/\/(www\.)?civitai\.(red|com)\//i.test(String(value || "").trim())
      || /^https?:\/\/.*\/api\/download\/models\//i.test(String(value || "").trim());
  }

  function modelUrl(item, version) {
    return version?.url || item?.url || "";
  }

  function firstVersion(item) {
    return item?.versions?.[0] || null;
  }

  function previewList(item, version = firstVersion(item)) {
    const previews = version?.previews?.length ? version.previews : [];
    if (previews.length) return previews;
    const preview = version?.preview || item?.preview;
    return preview ? [{ url: preview, type: mediaType(preview), nsfw: item?.nsfw }] : [];
  }

  function mediaType(urlOrPreview) {
    const url = typeof urlOrPreview === "string" ? urlOrPreview : urlOrPreview?.url || "";
    return /\.(mp4|webm|mov)(\?|$)/i.test(url) ? "video" : "image";
  }

  function lowResVideoUrl(preview) {
    if (typeof preview === "string") return preview;
    const candidates = [
      preview?.lowResUrl,
      preview?.lowresUrl,
      preview?.lowRes,
      preview?.mobileUrl,
      preview?.previewUrl,
      preview?.video?.lowResUrl,
      preview?.video?.url,
      ...(Array.isArray(preview?.sources) ? preview.sources.map((source) => source?.url || source?.src) : []),
      ...(Array.isArray(preview?.variants) ? preview.variants.map((variant) => variant?.url || variant?.src) : []),
      preview?.url,
    ].filter(Boolean);
    const videoCandidates = candidates.filter((candidate) => mediaType(candidate) === "video");
    const url = String(videoCandidates[0] || preview?.url || "");
    return url.replace(/\/width=\d+\//i, "/width=450/");
  }

  function mediaUrl(preview) {
    const url = typeof preview === "string" ? preview : preview?.url;
    if (!url) return "";
    return (preview?.type || mediaType(url)) === "video" ? lowResVideoUrl(preview) : url;
  }

  function mediaHtml(preview, className, alt) {
    const url = mediaUrl(preview);
    if (!url) return `<div class="${className} flex items-center justify-center text-nexus-muted"><i class="fa-regular fa-image text-2xl"></i></div>`;
    if ((preview?.type || mediaType(url)) === "video") {
      return `<video src="${html(url)}" preload="none" muted loop playsinline class="${className} object-cover"></video>`;
    }
    return `<img src="${html(url)}" loading="lazy" decoding="async" alt="${html(alt)}" class="${className} object-cover">`;
  }

  function isInstalled(data, downloaded = false) {
    const flag = data?.installed ?? data?.downloaded ?? data?.already_downloaded ?? data?.exists ?? downloaded;
    if (flag === true) return true;
    if (String(flag).toLowerCase() === "true") return true;
    return !!(data?.path || data?.relative_path || data?.local_path || data?.installed_path);
  }

  function installedPath(data) {
    return data?.relative_path || data?.path || data?.local_path || data?.installed_path || "";
  }

  function wireModalVideoPlayback() {
    if (modalVideoObserver) modalVideoObserver.disconnect();
    const modal = el("civitaiMediaModal");
    const video = el("civitaiMediaStage")?.querySelector("video");
    if (!modal || !video) return;
    modalVideoObserver = new IntersectionObserver((entries) => {
      const visible = entries.some((entry) => entry.isIntersecting && entry.intersectionRatio >= 0.6);
      if (!visible || modal.classList.contains("hidden")) {
        video.pause();
        return;
      }
      video.play().catch(() => {});
    }, { threshold: [0, 0.6, 1] });
    modalVideoObserver.observe(video);
  }

  function mediaIsMature(preview, owner) {
    const flag = preview?.nsfw ?? owner?.nsfw ?? owner?.needsReview ?? false;
    if (Array.isArray(flag)) return flag.some(Boolean);
    return flag === true || String(flag).toLowerCase() === "true" || String(flag).toLowerCase().includes("x");
  }

  function visiblePreviews(data) {
    const showMature = !!el("civitaiNsfwToggle")?.checked;
    const source = data.previews?.length
      ? data.previews
      : data.preview
        ? [{ url: data.preview, type: mediaType(data.preview), nsfw: data.nsfw }]
        : [];
    return showMature ? source : source.filter((preview) => !mediaIsMature(preview, data));
  }

  function previewTile(preview, index, data, shape = "aspect-square") {
    const blurMature = !!el("civitaiBlurMatureToggle")?.checked;
    const mature = mediaIsMature(preview, data);
    const blurClass = mature && blurMature ? "blur-xl scale-110" : "";
    return `
      <button type="button" onclick="window.NexusCivitai?.openMedia(${index})" class="${shape} border border-nexus-border overflow-hidden bg-black relative group">
        <div class="w-full h-full transition-transform duration-300 group-hover:scale-[1.03] ${blurClass}">
          ${mediaHtml(preview, "w-full h-full", data.model_name)}
        </div>
        ${mature ? `<span class="absolute top-1 right-1 bg-yellow-400 text-black text-[7px] font-bold px-1 py-0.5 uppercase">Mature</span>` : ""}
      </button>
    `;
  }

  function ensureMediaModal() {
    let modal = el("civitaiMediaModal");
    if (modal) return modal;
    modal = document.createElement("div");
    modal.id = "civitaiMediaModal";
    modal.className = "fixed inset-0 z-[160] hidden flex items-center justify-center bg-black/90 backdrop-blur-sm";
    modal.innerHTML = `
      <div class="bg-nexus-panel border border-nexus-border w-[82vw] h-[84vh] max-w-[1500px] flex flex-col rounded-sm overflow-hidden">
        <div class="h-11 px-4 border-b border-nexus-border flex items-center justify-between bg-nexus-bg shrink-0">
          <span class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2"><i class="fa-solid fa-photo-film text-nexus-red"></i>Civitai Preview</span>
          <button type="button" onclick="window.NexusCivitai?.closeMedia()" class="text-nexus-muted hover:text-nexus-red transition-colors text-sm"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="relative flex-1 bg-black flex items-center justify-center overflow-hidden p-4">
          <button type="button" onclick="window.NexusCivitai?.navigateMedia(-1)" class="absolute left-4 top-1/2 -translate-y-1/2 z-10 w-10 h-10 bg-nexus-panel border border-nexus-border rounded-sm text-nexus-muted hover:text-white hover:border-nexus-red"><i class="fa-solid fa-chevron-left"></i></button>
          <div id="civitaiMediaStage" class="w-full h-full flex items-center justify-center"></div>
          <button type="button" onclick="window.NexusCivitai?.navigateMedia(1)" class="absolute right-4 top-1/2 -translate-y-1/2 z-10 w-10 h-10 bg-nexus-panel border border-nexus-border rounded-sm text-nexus-muted hover:text-white hover:border-nexus-red"><i class="fa-solid fa-chevron-right"></i></button>
          <span id="civitaiMediaCounter" class="absolute bottom-4 left-4 bg-nexus-panel border border-nexus-border text-[10px] font-mono text-nexus-muted px-2 py-1 rounded-sm">1 / 1</span>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    return modal;
  }

  function renderMediaModal() {
    const stage = el("civitaiMediaStage");
    const counter = el("civitaiMediaCounter");
    const preview = mediaItems[mediaIndex];
    if (!stage || !preview) return;
    const blurMature = !!el("civitaiBlurMatureToggle")?.checked;
    const mature = mediaIsMature(preview, currentDetail?.data);
    const blurClass = mature && blurMature ? "blur-xl scale-110" : "";
    const url = mediaUrl(preview);
    const isVideo = (preview?.type || mediaType(url)) === "video";
    stage.innerHTML = `
      <div class="relative max-w-full max-h-full overflow-hidden border border-nexus-border bg-black">
        ${isVideo
          ? `<video src="${html(url)}" controls muted playsinline class="max-w-full max-h-[76vh] object-contain ${blurClass}"></video>`
          : `<img src="${html(url)}" alt="Civitai preview" class="max-w-full max-h-[76vh] object-contain ${blurClass}">`}
        ${mature && blurMature ? `<span class="absolute top-3 left-3 bg-yellow-400 text-black text-[9px] font-bold px-2 py-1 uppercase">Mature preview blurred</span>` : ""}
      </div>
    `;
    if (counter) counter.innerText = `${mediaIndex + 1} / ${mediaItems.length}`;
    wireModalVideoPlayback();
  }

  function openMediaModal(index = 0) {
    if (!currentDetail) return;
    mediaItems = visiblePreviews(currentDetail.data);
    if (!mediaItems.length) {
      window.showToast?.("No Preview", "No visible preview is available with the current mature filter.");
      return;
    }
    mediaIndex = Math.max(0, Math.min(mediaItems.length - 1, index));
    ensureMediaModal().classList.remove("hidden");
    renderMediaModal();
  }

  function closeMediaModal() {
    const modal = el("civitaiMediaModal");
    if (!modal) return;
    modal.classList.add("hidden");
    if (modalVideoObserver) modalVideoObserver.disconnect();
    const stage = el("civitaiMediaStage");
    stage?.querySelectorAll("video").forEach((video) => video.pause());
    if (stage) stage.innerHTML = "";
  }

  function navigateMedia(direction) {
    if (!mediaItems.length) return;
    mediaIndex = (mediaIndex + direction + mediaItems.length) % mediaItems.length;
    renderMediaModal();
  }

  function downloadCount(item) {
    return Number(item?.stats?.downloadCount || item?.stats?.download_count || 0).toLocaleString();
  }

  function formatSize(kb) {
    const value = Number(kb || 0);
    if (!Number.isFinite(value) || value <= 0) return "Size unknown";
    if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(2)} GB`;
    if (value >= 1024) return `${(value / 1024).toFixed(1)} MB`;
    return `${Math.round(value)} KB`;
  }

  function formatBytes(bytes) {
    const value = Number(bytes || 0);
    if (!Number.isFinite(value) || value <= 0) return "0 B";
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
    if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
    return `${Math.round(value)} B`;
  }

  function downloadIsActive(job) {
    return ["queued", "resolving", "downloading", "saving_preview", "downloaded"].includes(String(job?.status || ""));
  }

  function updateDownloadUi(job = null) {
    const jobs = [...downloadJobs.values()];
    const active = jobs.filter(downloadIsActive);
    const latest = job || active.at(-1) || jobs.at(-1) || null;
    const activeNode = el("civitaiActiveDownloads");
    const summary = el("civitaiDownloadSummary");
    const panel = el("civitaiDownloadPanel");
    const name = el("civitaiDownloadName");
    const percent = el("civitaiDownloadPercent");
    const speed = el("civitaiDownloadSpeed");
    const bar = el("civitaiDownloadProgressBar");
    if (activeNode) activeNode.innerText = String(active.length);
    if (!latest) {
      if (summary) summary.innerText = "Idle";
      panel?.classList.add("hidden");
      return;
    }

    const progress = Math.max(0, Math.min(100, Number(latest.progress || 0)));
    const label = latest.model_name || latest.file_name || latest.message || "Civitai download";
    const statusText = latest.status === "completed"
      ? "Complete"
      : latest.status === "failed"
        ? "Failed"
        : `${progress.toFixed(progress < 10 && progress > 0 ? 1 : 0)}%`;
    if (summary) summary.innerText = `${statusText}: ${label}`;
    if (name) name.innerText = `${latest.status || "download"} · ${label}`;
    if (percent) percent.innerText = `${progress.toFixed(progress < 10 && progress > 0 ? 1 : 0)}%`;
    if (speed) {
      const total = latest.bytes_total ? ` / ${formatBytes(latest.bytes_total)}` : "";
      speed.innerText = `${formatBytes(latest.bytes_downloaded)}${total} · ${formatBytes(latest.speed_bps)}/s`;
    }
    if (bar) bar.style.width = `${progress}%`;
    panel?.classList.remove("hidden");
    if (latest.status === "completed" || latest.status === "failed") {
      setTimeout(() => {
        const stillLatest = [...downloadJobs.values()].at(-1);
        if (stillLatest?.job_id === latest.job_id && !downloadJobsHasActive()) panel?.classList.add("hidden");
      }, 8000);
    }
  }

  function downloadJobsHasActive() {
    return [...downloadJobs.values()].some(downloadIsActive);
  }

  function rememberDownloadJob(job) {
    if (!job?.job_id) return;
    downloadJobs.set(job.job_id, job);
    updateDownloadUi(job);
  }

  async function pollDownloadJob(jobId) {
    while (jobId) {
      const job = await get(`/civitai/download/${encodeURIComponent(jobId)}`);
      rememberDownloadJob(job);
      if (job.status === "completed") {
        resolved = job.result;
        if (job.result) renderResult(job.result, true);
        status("Downloaded.");
        window.refreshModelCatalog?.();
        window.refreshLoraLibrary?.();
        window.showToast?.("Civitai Download Complete", job.file_name || job.model_name || "Model saved.");
        return job;
      }
      if (job.status === "failed") {
        throw new Error(job.error || job.message || "Download failed.");
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  }

  async function syncDownloadJobs() {
    try {
      const data = await get("/civitai/downloads");
      (data.jobs || []).forEach(rememberDownloadJob);
      const active = (data.jobs || []).filter(downloadIsActive);
      active.forEach((job) => pollDownloadJob(job.job_id).catch(() => updateDownloadUi()));
      updateDownloadUi(active.at(-1) || (data.jobs || []).at(-1) || null);
    } catch {
      updateDownloadUi();
    }
  }

  function renderResult(data, downloaded) {
    const panel = el("civitaiResultPanel");
    if (!panel) return;
    if (viewMode === "explorer") rememberExplorerScroll();
    viewMode = "detail";
    currentDetail = { data, downloaded };
    const previews = visiblePreviews(data);
    const installed = isInstalled(data, downloaded);
    const localPath = installedPath(data);
    panel.innerHTML = `
      <div class="mb-4 flex justify-between items-center border-b border-nexus-border pb-3">
        <button onclick="window.NexusCivitai?.backToSearch()" class="flat-button px-3 py-2 text-xs font-bold"><i class="fa-solid fa-arrow-left text-nexus-red mr-1"></i>Back to exploration list</button>
        <button onclick="window.open(${JSON.stringify(String(data.url || data.model_url || ""))} || document.getElementById('civitaiUrlInput')?.value, '_blank')" class="flat-button px-3 py-2 text-xs font-bold"><i class="fa-solid fa-arrow-up-right-from-square text-nexus-red mr-1"></i>Open Civitai</button>
      </div>
      <div class="grid grid-cols-1 xl:grid-cols-[420px_1fr] gap-5">
        <div class="space-y-3">
          ${previews[0] ? previewTile(previews[0], 0, data, "aspect-[3/4]") : `<div class="aspect-[3/4] bg-nexus-bg border border-nexus-border flex items-center justify-center text-center text-nexus-muted p-4">No visible preview with current mature filter.</div>`}
          <div class="bg-nexus-bg border border-nexus-border p-2 text-[10px] text-nexus-muted font-mono">${installed ? "Installed" : "Resolved"}</div>
        </div>
        <div class="space-y-4">
          <div>
            <h3 class="text-xl font-bold text-white">${html(data.model_name)}</h3>
            <p class="text-[11px] text-nexus-muted font-mono">Version ID: <span class="text-nexus-red">${html(data.version_id)}</span> - ${html(data.base_model || "Base model unknown")}</p>
          </div>
          ${(data.trained_words || []).length ? `
            <section class="bg-nexus-bg border border-nexus-border p-3">
              <div class="flex justify-between items-center mb-2"><span class="mini-label">Recommended trigger words</span><button onclick="navigator.clipboard?.writeText(${JSON.stringify((data.trained_words || []).join(", "))})" class="text-[9px] text-nexus-red hover:text-white"><i class="fa-regular fa-copy"></i> Copy Trigger</button></div>
              <code class="block bg-black border border-nexus-border p-2 text-xs text-white">${html((data.trained_words || []).join(", "))}</code>
            </section>
          ` : ""}
          <section class="bg-nexus-bg border border-nexus-border p-3 space-y-3">
            <div class="flex justify-between items-center border-b border-nexus-border pb-2">
              <span class="text-xs font-bold text-white uppercase"><i class="fa-regular fa-file-lines text-nexus-red mr-1"></i>Downloadable model file</span>
              <span class="text-[9px] font-mono text-nexus-muted">Target: <b class="text-white">${html(data.target_folder || data.relative_path || "Auto folder")}</b></span>
            </div>
            <div class="grid grid-cols-[1fr_140px] gap-2 items-center bg-black border border-nexus-border p-3">
              <div class="min-w-0">
                <p class="text-xs text-white font-bold break-all">${html(data.file_name)}</p>
                <p class="text-[10px] text-nexus-muted font-mono">${html(data.model_type || data.target_kind)} - ${html(data.version_name || "Version")} - ${html(formatSize(data.file_size_kb))}</p>
              </div>
              <button ${installed ? "disabled" : `onclick="window.NexusCivitai?.download()"`} class="${installed ? "bg-nexus-hover text-nexus-muted cursor-default" : "bg-nexus-red hover:bg-nexus-darkRed text-white"} px-3 py-2 rounded-sm text-[10px] font-bold uppercase"><i class="fa-solid ${installed ? "fa-circle-check" : "fa-cloud-arrow-down"} mr-1"></i>${installed ? "Installed" : "Download"}</button>
            </div>
          </section>
          ${data.description ? `<section><span class="mini-label">Description & author notes</span><p class="mt-2 text-xs leading-5 text-white">${html(stripHtml(data.description)).slice(0, 1400)}</p></section>` : ""}
          ${previews.length > 1 ? `<section><span class="mini-label">Model gallery</span><div class="grid grid-cols-4 lg:grid-cols-6 gap-2 mt-2">${previews.slice(0, 24).map((preview, index) => previewTile(preview, index, data)).join("")}</div></section>` : ""}
          ${installed && localPath ? `<div class="border border-nexus-red bg-nexus-bg p-2 text-[10px] text-white font-mono break-all">${html(localPath)}</div>` : ""}
        </div>
      </div>
    `;
  }

  function renderSearch(items, append) {
    const panel = el("civitaiResultPanel");
    if (!panel) return;
    const scrollTopBeforeRender = explorerScroller()?.scrollTop || explorerScrollTop;
    viewMode = "explorer";
    currentDetail = null;
    if (!append) currentItems = [];
    currentItems = append ? [...currentItems, ...items] : items;
    if (!currentItems.length) {
      panel.innerHTML = `<div class="h-full min-h-[300px] flex items-center justify-center text-center text-nexus-muted">No Civitai models found.</div>`;
      return;
    }
    const blurMature = !!el("civitaiBlurMatureToggle")?.checked;
    panel.innerHTML = `
      <div class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
        ${currentItems.map((item, index) => {
          const version = firstVersion(item) || {};
          const preview = previewList(item, version)[0];
          const mature = mediaIsMature(preview, item);
          const matureClass = mature && blurMature ? "blur-xl scale-110" : "";
          return `
            <article class="group bg-nexus-panel border border-nexus-border hover:border-nexus-red rounded-sm overflow-hidden">
              <button class="block w-full text-left" onclick="window.NexusCivitai?.selectSearchResult(${index})">
                <div class="aspect-[3/4] bg-nexus-bg border-b border-nexus-border overflow-hidden relative">
                  <div class="w-full h-full transition-transform duration-300 group-hover:scale-[1.03] ${matureClass}">
                    ${mediaHtml(preview || "", "w-full h-full", item.name)}
                  </div>
                  <span class="absolute top-2 left-2 bg-nexus-red text-white text-[7px] font-mono font-bold px-1.5 py-0.5 uppercase">${html(item.type)}</span>
                  ${mature ? `<span class="absolute top-2 right-2 bg-yellow-400 text-black text-[7px] font-bold px-1.5 py-0.5 uppercase">Mature</span>` : ""}
                  <span class="absolute bottom-2 right-2 bg-black/80 text-zinc-300 text-[8px] font-mono px-1.5 py-0.5 rounded-sm">${html(version.base_model || "Base")}</span>
                </div>
                <div class="p-2 space-y-1">
                  <h4 class="text-xs font-bold text-white truncate group-hover:text-nexus-red">${html(item.name)}</h4>
                  <p class="text-[10px] text-nexus-muted truncate">by ${html(item.creator || "Unknown creator")}</p>
                  <p class="text-[10px] text-nexus-muted font-mono"><i class="fa-solid fa-cloud-arrow-down text-nexus-red"></i> ${downloadCount(item)}</p>
                </div>
              </button>
            </article>
          `;
        }).join("")}
      </div>
      ${searchCursor ? `<div class="py-6 flex items-center justify-center gap-2 text-nexus-muted font-mono text-[10px]"><i class="fa-solid fa-spinner text-nexus-red"></i> Lazy loading more models as you scroll...</div>` : ""}
    `;
    if (append) scrollExplorerTo(scrollTopBeforeRender);
  }

  async function resolve() {
    const payload = formPayload();
    if (!payload.url) {
      return search(false);
    }
    status("Resolving...");
    resolved = await post("/civitai/resolve", payload);
    renderResult(resolved, false);
    status("Resolved.");
  }

  async function download(urlOverride) {
    const payload = formPayload();
    if (urlOverride) payload.url = urlOverride;
    if (!payload.url) {
      status("Paste a Civitai URL first.");
      return;
    }
    status("Starting download...");
    const job = await post("/civitai/download/start", payload);
    rememberDownloadJob(job);
    window.showToast?.("Civitai Download Started", currentDetail?.data?.file_name || payload.url);
    return pollDownloadJob(job.job_id);
  }

  async function search(append = false) {
    if (append && loadingMore) return;
    if (!append && activeSearchController) activeSearchController.abort();
    const runId = ++searchRunId;
    activeSearchController = new AbortController();
    loadingMore = true;
    status(append ? "Loading more..." : "Searching Civitai...");
    const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
    const payload = {
      ...basePayload(),
      query: isCivitaiUrl(rawInput) ? "" : normalizedCivitaiQuery(rawInput),
      types: el("civitaiTypeFilter")?.value || "",
      base_model: el("civitaiBaseModelFilter")?.value || "",
      sort: el("civitaiSortFilter")?.value || "Newest",
      period: "AllTime",
      nsfw: !!el("civitaiNsfwToggle")?.checked,
      limit: lazyPageSize,
      cursor: append ? searchCursor : null,
    };
    try {
      const result = await post("/civitai/search", payload, activeSearchController.signal);
      if (runId !== searchRunId) return;
      searchCursor = result.metadata?.nextCursor || null;
      renderSearch(result.items || [], append);
      if (!append) {
        explorerScrollTop = 0;
        scrollExplorerTo(0);
      }
      status(`${currentItems.length} model(s) loaded.`);
    } catch (error) {
      if (error?.name === "AbortError") return;
      throw error;
    } finally {
      if (runId === searchRunId) {
        loadingMore = false;
        activeSearchController = null;
      }
    }
  }

  window.NexusCivitai = {
    open() {
      el("civitaiModal")?.classList.remove("hidden");
      wireControls();
      updateTypeButtons();
      syncDownloadJobs();
      status("Idle");
      if (!currentItems.length) search(false).catch(() => status("Browse unavailable."));
    },
    close() {
      el("civitaiModal")?.classList.add("hidden");
      closeMediaModal();
    },
    resolve() {
      resolve().catch((error) => {
        status("Resolve failed.");
        window.showToast?.("Civitai Resolve Failed", String(error.message || error).slice(0, 180));
      });
    },
    download() {
      download().catch((error) => {
        updateDownloadUi();
        status("Download failed.");
        window.showToast?.("Civitai Download Failed", String(error.message || error).slice(0, 180));
      });
    },
    search() {
      search(false).catch((error) => {
        status("Search failed.");
        window.showToast?.("Civitai Search Failed", String(error.message || error).slice(0, 180));
      });
    },
    loadMore() {
      if (!searchCursor) {
        status("No more results.");
        return;
      }
      search(true).catch((error) => {
        status("Load more failed.");
        window.showToast?.("Civitai Search Failed", String(error.message || error).slice(0, 180));
      });
    },
    handleScroll(container) {
      if (viewMode !== "explorer") return;
      if (container && !suppressExplorerScrollCapture) explorerScrollTop = container.scrollTop;
      if (!container || !searchCursor || loadingMore) return;
      if (container.scrollHeight - container.scrollTop - container.clientHeight < 500) this.loadMore();
    },
    selectSearchResult(index) {
      selectedIndex = index;
      const item = currentItems[index];
      const version = firstVersion(item);
      if (!item || !version) return;
      const url = modelUrl(item, version);
      if (el("civitaiUrlInput")) el("civitaiUrlInput").value = url;
      renderResult({
        model_name: item.name,
        model_type: item.type,
        version_id: version.id,
        version_name: version.name,
        base_model: version.base_model,
        file_name: version.file_name,
        file_size_kb: version.file_size_kb,
        installed: version.installed ?? item.installed,
        downloaded: version.downloaded ?? item.downloaded,
        already_downloaded: version.already_downloaded ?? item.already_downloaded,
        exists: version.exists ?? item.exists,
        relative_path: version.relative_path || item.relative_path || "",
        path: version.path || item.path || "",
        nsfw: item.nsfw,
        preview: version.preview || item.preview,
        previews: previewList(item, version),
        trained_words: version.trained_words || [],
        target_kind: "auto",
        target_folder: "Auto folder",
        description: version.description || item.description || "",
        creator: item.creator,
        stats: item.stats,
        url,
      }, false);
    },
    backToSearch() {
      renderSearch(currentItems, false);
      restoreExplorerScroll();
      status(`${currentItems.length} model(s) loaded.`);
    },
    handleEscape() {
      if (viewMode !== "detail") return false;
      this.backToSearch();
      return true;
    },
    resolveSearchResult(index = selectedIndex) {
      const item = currentItems[index];
      const version = firstVersion(item);
      if (el("civitaiUrlInput")) el("civitaiUrlInput").value = modelUrl(item, version);
      resolve().catch((error) => {
        status("Resolve failed.");
        window.showToast?.("Civitai Resolve Failed", String(error.message || error).slice(0, 180));
      });
    },
    downloadSearchResult(index = selectedIndex) {
      const item = currentItems[index];
      const version = firstVersion(item);
      download(modelUrl(item, version)).catch((error) => {
        updateDownloadUi();
        status("Download failed.");
        window.showToast?.("Civitai Download Failed", String(error.message || error).slice(0, 180));
      });
    },
    openMedia(index) {
      openMediaModal(index);
    },
    closeMedia() {
      closeMediaModal();
    },
    navigateMedia(direction) {
      navigateMedia(direction);
    },
    setType(value) {
      setTypeFilter(value);
    },
    clearToken() {
      localStorage.removeItem(tokenStorageKey);
      const token = el("civitaiTokenInput");
      if (token) token.value = "";
      window.showToast?.("Civitai API Key Cleared", "Saved token removed from this device.");
    },
  };
})();
