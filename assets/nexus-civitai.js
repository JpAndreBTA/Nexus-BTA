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
  let lastSearchSignature = "";
  let searchDebounceTimer = null;
  let tokenRefreshTimer = null;
  let autoLoadMoreCount = 0;
  const tokenStorageKey = "nexus_civitai_api_key";
  const downloadJobs = new Map();
  const lazyPageSize = 36;
  const autoLoadMoreMaxPages = 3;
  const galleryPreviewLimit = 60;
  const tagPresets = {
    "": ["character", "style", "concept", "realistic", "photorealistic", "anime", "female", "male", "digital art", "video game", "base model", "upscaler"],
    Checkpoint: ["base model", "realistic", "photorealistic", "anime", "semi-realistic", "cgi", "digital art", "style", "concept", "upscaler"],
    LORA: ["character", "style", "concept", "clothing", "pose", "anime", "female", "male", "ponyxl", "digital art", "western art", "video game"],
    TextualInversion: ["style", "concept", "anime", "realistic", "photorealistic", "female", "male"],
    VAE: ["vae", "base model", "sdxl", "anime", "realistic"],
    Controlnet: ["controlnet", "pose", "depth", "lineart", "canny"],
  };
  const matureTokens = new Set(["18+", "adult", "boobs", "explicit", "genital", "hentai", "mature", "naked", "nsfl", "nsfw", "nude", "nudity", "porn", "pussy", "sex", "sexual", "vagina"]);
  let modalVideoObserver = null;
  let tileVideoObserver = null;

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

  function normalizeTagValue(value) {
    return normalizedCivitaiQuery(value).replace(/\s+/g, " ");
  }

  function selectedTag() {
    return normalizeTagValue(el("civitaiTagFilter")?.value || "");
  }

  function tagsForCurrentType() {
    const type = el("civitaiTypeFilter")?.value || "";
    const active = selectedTag();
    const tags = tagPresets[type] || tagPresets[""];
    return active && !tags.includes(active) ? [active, ...tags] : tags;
  }

  function tagButtonClass(active) {
    return active
      ? "bg-nexus-red text-white border border-nexus-red px-2 py-1 rounded-sm text-[9px] font-bold uppercase shrink-0"
      : "bg-nexus-bg text-nexus-muted hover:text-white hover:border-nexus-red border border-nexus-border px-2 py-1 rounded-sm text-[9px] font-bold uppercase shrink-0";
  }

  function renderTagChips() {
    const node = el("civitaiTagChips");
    if (!node) return;
    const active = selectedTag();
    node.innerHTML = [
      `<button type="button" onclick="window.NexusCivitai?.setTag('')" class="${tagButtonClass(!active)}">All tags</button>`,
      ...tagsForCurrentType().map((tag) => `<button type="button" onclick="window.NexusCivitai?.setTag(${JSON.stringify(tag)})" class="${tagButtonClass(active === tag)}">#${html(tag)}</button>`),
    ].join("");
  }

  function setTagFilter(value) {
    const input = el("civitaiTagFilter");
    if (input) input.value = normalizeTagValue(value);
    const searchInput = el("civitaiUrlInput");
    if (searchInput && String(searchInput.value || "").trim().startsWith("#")) searchInput.value = "";
    clearUrlInputForBrowse();
    renderTagChips();
    startFreshSearch().catch(() => status("Search failed."));
  }

  function submitSearchInput() {
    const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
    clearTimeout(searchDebounceTimer);
    return isCivitaiUrl(rawInput) ? resolve() : startFreshSearch();
  }

  function clearUrlInputForBrowse() {
    const searchInput = el("civitaiUrlInput");
    if (searchInput && isCivitaiUrl(searchInput.value)) searchInput.value = "";
  }

  function activeSearchQuery() {
    const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
    if (!rawInput || isCivitaiUrl(rawInput) || rawInput.startsWith("#")) return "";
    return normalizedCivitaiQuery(rawInput).toLowerCase();
  }

  function itemSearchHaystack(item) {
    const version = firstVersion(item) || {};
    return [
      item?.name,
      item?.creator,
      item?.description,
      item?.type,
      ...(item?.tags || []),
      version?.name,
      version?.base_model,
      version?.file_name,
      ...(version?.trained_words || []),
    ].join(" ").normalize("NFKC").toLowerCase();
  }

  function itemMatchesSearchQuery(item, query = activeSearchQuery()) {
    const terms = String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return true;
    const haystack = itemSearchHaystack(item);
    return terms.every((term) => haystack.includes(term));
  }

  function queueSearch() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      startFreshSearch().catch((error) => {
        if (error?.name === "AbortError") return;
        status("Search failed.");
      });
    }, 360);
  }

  function queueTokenRefresh() {
    clearTimeout(tokenRefreshTimer);
    tokenRefreshTimer = setTimeout(() => {
      const modal = el("civitaiModal");
      if (!modal || modal.classList.contains("hidden")) return;
      const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
      const refresh = isCivitaiUrl(rawInput) || viewMode === "detail" ? submitSearchInput : startFreshSearch;
      refresh().catch((error) => {
        if (error?.name === "AbortError") return;
        status("Token refresh failed.");
      });
    }, 500);
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
    const installed = el("civitaiInstalledToggle");
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
        else startFreshSearch().catch(() => status("Search failed."));
        if (el("civitaiMediaModal") && !el("civitaiMediaModal").classList.contains("hidden")) closeMediaModal();
      });
    }
    if (installed && !installed.dataset.nexusWired) {
      installed.dataset.nexusWired = "1";
      installed.addEventListener("change", () => {
        renderSearch(currentItems, false);
        restoreExplorerScroll();
      });
    }
    [base, type, sort].forEach((control) => {
      if (!control || control.dataset.nexusWired) return;
      control.dataset.nexusWired = "1";
      control.addEventListener("change", () => {
        clearUrlInputForBrowse();
        if (control === type) renderTagChips();
        startFreshSearch().catch(() => status("Search failed."));
      });
    });
    updateTypeButtons();
    renderTagChips();
    if (searchInput && !searchInput.dataset.nexusWired) {
      searchInput.dataset.nexusWired = "1";
      searchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") submitSearchInput().catch(() => status("Submit failed."));
      });
      searchInput.addEventListener("input", () => {
        if (isCivitaiUrl(searchInput.value)) return;
        if (String(searchInput.value || "").trim().startsWith("#")) {
          const nextTag = normalizeTagValue(searchInput.value);
          const tag = el("civitaiTagFilter");
          if (tag && tag.value !== nextTag) {
            tag.value = nextTag;
            renderTagChips();
          }
        } else {
          const tag = el("civitaiTagFilter");
          if (tag && tag.value) {
            tag.value = "";
            renderTagChips();
          }
        }
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
    clearUrlInputForBrowse();
    updateTypeButtons();
    renderTagChips();
    startFreshSearch().catch(() => status("Search failed."));
  }

  function apiErrorMessage(responseText, fallback = "Civitai request failed.") {
    const text = String(responseText || "").trim();
    if (!text) return fallback;
    try {
      const data = JSON.parse(text);
      const detail = data?.detail || data?.message || data?.error;
      if (detail) return String(detail).slice(0, 220);
    } catch {
      // Keep the raw text fallback below for non-JSON responses.
    }
    return text.slice(0, 220);
  }

  async function post(path, body, signal = null, timeoutMs = 25000) {
    const controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    const abortFromCaller = () => controller.abort();
    if (signal) {
      if (signal.aborted) abortFromCaller();
      else signal.addEventListener("abort", abortFromCaller, { once: true });
    }
    try {
      const response = await fetch(api + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(apiErrorMessage(await response.text()));
      return response.json();
    } catch (error) {
      if (timedOut) throw new Error("Civitai request timed out. Try again or reduce filters.");
      if (signal?.aborted && error?.name !== "AbortError") {
        let abortError;
        try {
          abortError = new DOMException("Aborted", "AbortError");
        } catch {
          abortError = new Error("Aborted");
          abortError.name = "AbortError";
        }
        throw abortError;
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      if (signal) signal.removeEventListener?.("abort", abortFromCaller);
    }
  }

  async function get(path, timeoutMs = 25000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(api + path, { signal: controller.signal });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    } finally {
      clearTimeout(timeout);
    }
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
      queueTokenRefresh();
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

  function versionHasPreview(version) {
    return !!(version?.previews?.length || version?.preview);
  }

  function previewVersion(item) {
    return (item?.versions || []).find(versionHasPreview) || firstVersion(item);
  }

  function previewList(item, version = firstVersion(item)) {
    const previews = version?.previews?.length ? version.previews : [];
    if (previews.length) return previews;
    const fallbackVersion = versionHasPreview(version) ? null : (item?.versions || []).find(versionHasPreview);
    if (fallbackVersion) return previewList(item, fallbackVersion);
    const preview = version?.preview || item?.preview;
    return preview ? [{ url: preview, type: mediaType(preview), nsfw: item?.nsfw }] : [];
  }

  function showMatureEnabled() {
    return !!el("civitaiNsfwToggle")?.checked;
  }

  function matureFlag(value) {
    if (value === undefined || value === null || value === "") return false;
    if (Array.isArray(value)) return value.some(matureFlag);
    if (value === true) return true;
    if (value === false) return false;
    if (typeof value === "number") return value > 2;
    const text = String(value).trim().toLowerCase();
    if (!text || ["0", "1", "false", "none", "safe", "sfw"].includes(text)) return false;
    return ["mature", "nsfw", "racy", "xxx", "adult", "explicit"].some((token) => text.includes(token)) || text === "x";
  }

  function matureText(value) {
    const values = Array.isArray(value) ? value : [value];
    return values.some((entry) => {
      const text = String(entry || "").trim().toLowerCase();
      if (!text) return false;
      const normalized = text.replace(/[^a-z0-9+]+/g, " ").trim();
      return matureTokens.has(text) || matureTokens.has(normalized) || normalized.split(/\s+/).some((token) => matureTokens.has(token));
    });
  }

  function itemHasExplicitMatureSignal(item) {
    return matureFlag(item?.nsfw)
      || matureText(item?.tags || [])
      || matureText(item?.name || item?.model_name || "")
      || matureText(item?.description || "");
  }

  function mediaType(urlOrPreview) {
    const url = typeof urlOrPreview === "string" ? urlOrPreview : urlOrPreview?.url || "";
    return /\.(mp4|webm|mov)(\?|$)/i.test(url) ? "video" : "image";
  }

  function normalizedUrl(value) {
    const url = String(value || "").trim();
    if (!url) return "";
    return url.startsWith("//") ? `${window.location.protocol}${url}` : url;
  }

  function previewCandidates(preview) {
    if (typeof preview === "string") return [preview];
    return [
      preview?.lowResVideoUrl,
      preview?.lowresVideoUrl,
      preview?.video?.lowResUrl,
      preview?.video?.lowresUrl,
      preview?.videoUrl,
      preview?.video_url,
      preview?.video?.url,
      preview?.video?.src,
      preview?.originalVideoUrl,
      ...(Array.isArray(preview?.sources) ? preview.sources.map((source) => source?.url || source?.src) : []),
      ...(Array.isArray(preview?.variants) ? preview.variants.map((variant) => variant?.url || variant?.src) : []),
      preview?.lowResUrl,
      preview?.lowresUrl,
      preview?.mobileUrl,
      preview?.previewUrl,
      preview?.url,
    ].filter(Boolean);
  }

  function videoUrl(preview) {
    return previewCandidates(preview).map(normalizedUrl).find((candidate) => mediaType(candidate) === "video") || "";
  }

  function posterUrl(preview) {
    if (typeof preview === "string") return mediaType(preview) === "video" ? "" : normalizedUrl(preview);
    const candidates = [
      preview?.posterUrl,
      preview?.poster_url,
      preview?.thumbnailUrl,
      preview?.thumbUrl,
      preview?.imageUrl,
      preview?.image_url,
      preview?.lowResUrl,
      preview?.lowresUrl,
      preview?.mobileUrl,
      preview?.previewUrl,
      preview?.url,
    ].filter(Boolean);
    return candidates.map(normalizedUrl).find((candidate) => mediaType(candidate) !== "video") || "";
  }

  function mediaUrl(preview) {
    const url = typeof preview === "string" ? preview : preview?.url;
    if (!url) return posterUrl(preview) || videoUrl(preview);
    return normalizedUrl((preview?.type || mediaType(url)) === "video" ? (posterUrl(preview) || videoUrl(preview)) : url);
  }

  function previewKind(preview) {
    const url = mediaUrl(preview);
    if ((preview?.type || mediaType(url)) !== "video") return videoUrl(preview) ? "video" : "image";
    return videoUrl(preview) ? "video" : "image";
  }

  function mediaHtml(preview, className, alt, options = {}) {
    const url = mediaUrl(preview);
    if (!url) return `<div class="${className} flex items-center justify-center text-nexus-muted"><i class="fa-regular fa-image text-2xl"></i></div>`;
    const fallback = `<div class="absolute inset-0 flex items-center justify-center text-nexus-muted bg-nexus-bg"><i class="fa-regular fa-image text-2xl"></i></div>`;
    if (previewKind(preview) === "video") {
      const poster = posterUrl(preview);
      const src = videoUrl(preview) || url;
      const playable = !!options.playVideo && !!src;
      return `<div class="${className} relative bg-black">${fallback}${playable ? `<video data-src="${html(src)}" muted loop playsinline preload="none" ${poster ? `poster="${html(poster)}"` : ""} aria-label="${html(alt)}" onerror="this.remove()" class="nexus-civitai-tile-video relative z-10 w-full h-full object-cover"></video>` : poster ? `<img src="${html(poster)}" loading="lazy" decoding="async" alt="${html(alt)}" onerror="this.remove()" class="relative z-10 w-full h-full object-cover">` : `<div class="relative z-10 w-full h-full flex items-center justify-center text-nexus-muted"><i class="fa-solid fa-play text-2xl"></i></div>`}<span class="absolute bottom-2 left-2 z-20 bg-black/80 text-white text-[8px] font-bold px-1.5 py-0.5 uppercase"><i class="fa-solid fa-play mr-1"></i>Video</span></div>`;
    }
    return `<div class="${className} relative bg-nexus-bg overflow-hidden">${fallback}<img src="${html(url)}" loading="lazy" decoding="async" alt="${html(alt)}" onerror="this.remove()" class="relative z-10 w-full h-full object-cover"></div>`;
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

  function markInstalled(data) {
    if (!data) return;
    const versionId = String(data.version_id || data.id || "");
    const path = installedPath(data);
    currentItems = currentItems.map((item) => {
      const versions = (item.versions || []).map((version) => {
        if (versionId && String(version.id || "") !== versionId) return version;
        return {
          ...version,
          installed: true,
          downloaded: true,
          already_downloaded: true,
          exists: true,
          relative_path: data.relative_path || version.relative_path || path,
          path: data.path || version.path || "",
        };
      });
      const matched = versions.some((version) => version.installed && (!versionId || String(version.id || "") === versionId));
      return matched ? { ...item, versions, installed: true, downloaded: true, relative_path: data.relative_path || item.relative_path || path, path: data.path || item.path || "" } : { ...item, versions };
    });
  }

  function wireModalVideoPlayback() {
    if (modalVideoObserver) modalVideoObserver.disconnect();
    const modal = el("civitaiMediaModal");
    const video = el("civitaiMediaStage")?.querySelector("video");
    if (!modal || !video) return;
    if (!video.src) video.src = video.dataset.src || "";
    video.load();
    modalVideoObserver = new IntersectionObserver((entries) => {
      const visible = entries.some((entry) => entry.isIntersecting && entry.intersectionRatio >= 0.6);
      if (!visible || modal.classList.contains("hidden")) {
        video.pause();
      }
    }, { threshold: [0, 0.6, 1] });
    modalVideoObserver.observe(video);
  }

  function wireTileVideoLazyload() {
    if (tileVideoObserver) tileVideoObserver.disconnect();
    const videos = [...document.querySelectorAll(".nexus-civitai-tile-video[data-src]")];
    if (!videos.length) return;
    const loadVideo = (video) => {
      if (!video.src) {
        video.src = video.dataset.src || "";
        video.load();
      }
    };
    const playVideo = (video) => {
      loadVideo(video);
      video.play?.().catch(() => {});
    };
    if (!("IntersectionObserver" in window)) {
      videos.slice(0, 8).forEach(playVideo);
      return;
    }
    tileVideoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting && entry.intersectionRatio >= 0.25) {
          playVideo(video);
        } else {
          video.pause?.();
        }
      });
    }, { root: explorerScroller(), rootMargin: "360px 0px", threshold: [0, 0.25, 0.75] });
    videos.forEach((video) => tileVideoObserver.observe(video));
  }

  function mediaIsMature(preview, owner) {
    if (itemHasExplicitMatureSignal(owner)) return true;
    const previewFlag = preview?.nsfw ?? preview?.needsReview ?? preview?.nsfwLevel ?? preview?.nsfw_level;
    if (previewFlag !== undefined && previewFlag !== null && previewFlag !== "") return matureFlag(previewFlag);
    return false;
  }

  function itemVisibleWithCurrentMatureFilter(item) {
    if (showMatureEnabled()) return true;
    if (itemHasExplicitMatureSignal(item)) return false;
    const version = firstVersion(item) || {};
    const previews = previewList(item, version);
    if (!previews.length) return true;
    return previews.some((preview) => !mediaIsMature(preview, item));
  }

  function installedOnlyEnabled() {
    return !!el("civitaiInstalledToggle")?.checked;
  }

  function setInstalledOnly(value) {
    const toggle = el("civitaiInstalledToggle");
    if (toggle) toggle.checked = !!value;
  }

  function itemInstalled(item) {
    return isInstalled(item) || (item?.versions || []).some((version) => isInstalled(version));
  }

  function itemVisibleWithCurrentFilters(item) {
    if (installedOnlyEnabled() && !itemInstalled(item)) return false;
    return itemVisibleWithCurrentMatureFilter(item);
  }

  function visibleSearchEntries() {
    return currentItems
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => itemVisibleWithCurrentFilters(item));
  }

  function currentFilterSummary() {
    const filters = [];
    const query = activeSearchQuery();
    if (query) filters.push(`search: ${query}`);
    if (installedOnlyEnabled()) filters.push("Installed");
    if (!showMatureEnabled()) filters.push("Show mature off");
    const type = el("civitaiTypeFilter")?.value || "";
    const base = el("civitaiBaseModelFilter")?.value || "";
    const tag = selectedTag();
    if (type) filters.push(type);
    if (base) filters.push(base);
    if (tag) filters.push(`#${tag}`);
    return filters.join(" + ") || "none";
  }

  function clearSearchFilters() {
    setInstalledOnly(false);
    const type = el("civitaiTypeFilter");
    const base = el("civitaiBaseModelFilter");
    const tag = el("civitaiTagFilter");
    const searchInput = el("civitaiUrlInput");
    if (type) type.value = "";
    if (base) base.value = "";
    if (tag) tag.value = "";
    if (searchInput && !isCivitaiUrl(searchInput.value)) searchInput.value = "";
    updateTypeButtons();
    renderTagChips();
    startFreshSearch().catch(() => status("Search failed."));
  }

  function visiblePreviews(data) {
    const source = data.previews?.length
      ? data.previews
      : data.preview
        ? [{ url: data.preview, type: mediaType(data.preview), nsfw: data.nsfw }]
        : [];
    return showMatureEnabled() ? source : source.filter((preview) => !mediaIsMature(preview, data));
  }

  function noPreviewMessage(data) {
    if (showMatureEnabled()) {
      return data?.nsfw || data?.nsfw_level || data?.nsfwLevel
        ? "Civitai returned no preview media for this mature model."
        : "Civitai returned no preview media for this model.";
    }
    return "No visible preview with current mature filter.";
  }

  function previewTile(preview, index, data, shape = "aspect-square") {
    const blurMature = !!el("civitaiBlurMatureToggle")?.checked;
    const mature = mediaIsMature(preview, data);
    const blurClass = mature && blurMature ? "blur-xl scale-110" : "";
    const playVideo = showMatureEnabled();
    return `
      <button type="button" onclick="window.NexusCivitai?.openMedia(${index})" class="${shape} border border-nexus-border overflow-hidden bg-black relative group">
        <div class="w-full h-full transition-transform duration-300 group-hover:scale-[1.03] ${blurClass}">
          ${mediaHtml(preview, "w-full h-full", data.model_name, { playVideo })}
        </div>
        ${mature ? `<span class="absolute z-30 top-1 right-1 bg-yellow-400 text-black text-[7px] font-bold px-1 py-0.5 uppercase">Mature</span>` : ""}
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
    const isVideo = previewKind(preview) === "video";
    const poster = posterUrl(preview);
    const fallback = `<div class="absolute inset-0 flex items-center justify-center text-nexus-muted bg-nexus-bg"><i class="fa-regular fa-image text-3xl"></i></div>`;
    stage.innerHTML = `
      <div class="relative max-w-full max-h-full overflow-hidden border border-nexus-border bg-black">
        ${isVideo
          ? `<div class="relative max-w-full max-h-[76vh]">${fallback}<video data-src="${html(videoUrl(preview))}" controls muted playsinline preload="metadata" poster="${html(poster)}" onerror="this.remove()" class="relative z-10 max-w-full max-h-[76vh] object-contain ${blurClass}">Your browser cannot play this Civitai video.</video></div>`
          : `${fallback}<img src="${html(url)}" alt="Civitai preview" onerror="this.remove()" class="relative z-10 max-w-full max-h-[76vh] object-contain ${blurClass}">`}
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
    const statusText = latest.status === "completed" || latest.status === "downloaded"
      ? "Complete"
      : latest.status === "failed"
        ? "Failed"
        : `${progress.toFixed(progress < 10 && progress > 0 ? 1 : 0)}%`;
    if (summary) summary.innerText = `${statusText}: ${label}`;
    const latestStatus = latest.status === "downloaded" ? "installed" : (latest.status || "download");
    if (name) name.innerText = `${latestStatus} - ${label}`;
    if (percent) percent.innerText = `${progress.toFixed(progress < 10 && progress > 0 ? 1 : 0)}%`;
    if (speed) {
      const total = latest.bytes_total ? ` / ${formatBytes(latest.bytes_total)}` : "";
      speed.innerText = `${formatBytes(latest.bytes_downloaded)}${total} - ${formatBytes(latest.speed_bps)}/s`;
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
        if (job.result) {
          markInstalled(job.result);
          renderResult({ ...job.result, installed: true, downloaded: true, already_downloaded: true, exists: true }, true);
        }
        status("Installed.");
        window.refreshModelCatalog?.();
        window.refreshLoraLibrary?.();
        window.showToast?.("Civitai Installed", job.file_name || job.model_name || "Model saved.");
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
          ${previews[0] ? previewTile(previews[0], 0, data, "aspect-[3/4]") : `<div class="aspect-[3/4] bg-nexus-bg border border-nexus-border flex items-center justify-center text-center text-nexus-muted p-4">${html(noPreviewMessage(data))}</div>`}
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
          ${previews.length > 1 ? `<section><span class="mini-label">Model gallery</span><div class="grid grid-cols-4 lg:grid-cols-6 gap-2 mt-2">${previews.slice(0, galleryPreviewLimit).map((preview, index) => previewTile(preview, index, data)).join("")}</div></section>` : ""}
          ${installed && localPath ? `<div class="border border-nexus-red bg-nexus-bg p-2 text-[10px] text-white font-mono break-all">${html(localPath)}</div>` : ""}
        </div>
      </div>
    `;
    wireTileVideoLazyload();
  }

  function searchCardHtml(item, index, blurMature) {
    const version = firstVersion(item) || {};
    const displayVersion = previewVersion(item) || version;
    const preview = previewList(item, displayVersion)[0];
    const mature = mediaIsMature(preview, item);
    const matureClass = mature && blurMature ? "blur-xl scale-110" : "";
    const installed = isInstalled(version, isInstalled(item));
    const playVideo = showMatureEnabled();
    return `
      <article class="group bg-nexus-panel border border-nexus-border hover:border-nexus-red rounded-sm overflow-hidden">
        <button class="block w-full text-left" onclick="window.NexusCivitai?.selectSearchResult(${index})">
          <div class="aspect-[3/4] bg-nexus-bg border-b border-nexus-border overflow-hidden relative">
            <div class="w-full h-full transition-transform duration-300 group-hover:scale-[1.03] ${matureClass}">
              ${mediaHtml(preview || "", "w-full h-full", item.name, { playVideo })}
            </div>
            <span class="absolute z-30 top-2 left-2 bg-nexus-red text-white text-[7px] font-mono font-bold px-1.5 py-0.5 uppercase">${html(item.type)}</span>
            ${mature ? `<span class="absolute z-30 top-2 right-2 bg-yellow-400 text-black text-[7px] font-bold px-1.5 py-0.5 uppercase">Mature</span>` : ""}
            ${installed ? `<span class="absolute z-30 bottom-7 left-2 bg-emerald-500 text-black text-[7px] font-bold px-1.5 py-0.5 uppercase"><i class="fa-solid fa-circle-check mr-1"></i>Installed</span>` : ""}
            <span class="absolute z-30 bottom-2 right-2 bg-black/80 text-zinc-300 text-[8px] font-mono px-1.5 py-0.5 rounded-sm">${html(version.base_model || "Base")}</span>
          </div>
          <div class="p-2 space-y-1">
            <h4 class="text-xs font-bold text-white truncate group-hover:text-nexus-red">${html(item.name)}</h4>
            <p class="text-[10px] text-nexus-muted truncate">by ${html(item.creator || "Unknown creator")}</p>
            <p class="text-[10px] text-nexus-muted font-mono"><i class="fa-solid fa-cloud-arrow-down text-nexus-red"></i> ${downloadCount(item)}</p>
          </div>
        </button>
      </article>
    `;
  }

  function searchLoadingHtml(label = "Searching Civitai...") {
    return `
      <div class="h-full min-h-[360px] flex flex-col items-center justify-center text-center text-nexus-muted gap-3">
        <i class="fa-solid fa-spinner fa-spin text-nexus-red text-xl"></i>
        <p class="font-mono text-[11px]">${html(label)}</p>
      </div>
    `;
  }

  function searchErrorHtml(message = "Civitai search failed.") {
    return `
      <div class="h-full min-h-[360px] flex flex-col items-center justify-center text-center text-nexus-muted gap-3">
        <i class="fa-solid fa-triangle-exclamation text-nexus-red text-xl"></i>
        <p class="font-mono text-[11px]">${html(message)}</p>
        <button type="button" onclick="window.NexusCivitai?.search()" class="flat-button px-3 py-2 text-xs font-bold uppercase">
          <i class="fa-solid fa-rotate-right text-nexus-red mr-1"></i>Retry
        </button>
      </div>
    `;
  }

  function resetSearchState({ clearPanel = true } = {}) {
    clearTimeout(searchDebounceTimer);
    clearTimeout(tokenRefreshTimer);
    if (activeSearchController) activeSearchController.abort();
    searchRunId += 1;
    activeSearchController = null;
    loadingMore = false;
    searchCursor = null;
    currentItems = [];
    autoLoadMoreCount = 0;
    selectedIndex = -1;
    viewMode = "explorer";
    currentDetail = null;
    mediaItems = [];
    mediaIndex = 0;
    explorerScrollTop = 0;
    lastSearchSignature = "";
    if (tileVideoObserver) {
      tileVideoObserver.disconnect();
      tileVideoObserver = null;
    }
    if (clearPanel) {
      const panel = el("civitaiResultPanel");
      if (panel) panel.innerHTML = searchLoadingHtml("Ready to search Civitai...");
    }
    const scroller = explorerScroller();
    if (scroller) scroller.scrollTop = 0;
  }

  function lazyStatusHtml() {
    return searchCursor
      ? `<div id="civitaiLazyStatus" class="py-6 flex items-center justify-center gap-2 text-nexus-muted font-mono text-[10px]"><i class="fa-solid fa-spinner text-nexus-red"></i> Lazy loading more models as you scroll...</div>`
      : `<div id="civitaiLazyStatus" class="hidden"></div>`;
  }

  function renderSearch(items, append) {
    const panel = el("civitaiResultPanel");
    if (!panel) return;
    const scrollTopBeforeRender = explorerScroller()?.scrollTop || explorerScrollTop;
    viewMode = "explorer";
    currentDetail = null;
    const startIndex = append ? currentItems.length : 0;
    currentItems = append ? [...currentItems, ...items] : items;
    const visibleEntries = visibleSearchEntries();
    if (!visibleEntries.length) {
      const message = currentItems.length
        ? "No models match the current Civitai filters."
        : "No Civitai models found.";
      panel.innerHTML = `
        <div class="h-full min-h-[300px] flex flex-col items-center justify-center text-center text-nexus-muted gap-3">
          <p>${message}</p>
          ${currentItems.length ? `<p class="text-[10px] font-mono">Active filters: ${html(currentFilterSummary())}</p><button type="button" onclick="window.NexusCivitai?.clearFilters()" class="flat-button px-3 py-2 text-xs font-bold uppercase"><i class="fa-solid fa-filter-circle-xmark text-nexus-red mr-1"></i>Clear filters</button>` : ""}
          ${searchCursor ? `<button type="button" onclick="window.NexusCivitai?.loadMore()" class="bg-nexus-red hover:bg-nexus-darkRed text-white px-3 py-2 rounded-sm text-xs font-bold uppercase"><i class="fa-solid fa-angles-down mr-1"></i>Load more</button>` : ""}
        </div>
      `;
      if (searchCursor && autoLoadMoreCount < autoLoadMoreMaxPages) {
        autoLoadMoreCount += 1;
        setTimeout(() => window.NexusCivitai?.loadMore?.(), 0);
      }
      return;
    }
    const blurMature = !!el("civitaiBlurMatureToggle")?.checked;
    if (append) {
      const grid = el("civitaiSearchGrid");
      const lazyStatus = el("civitaiLazyStatus");
      const appendedEntries = items
        .map((item, offset) => ({ item, index: startIndex + offset }))
        .filter(({ item }) => itemVisibleWithCurrentFilters(item));
      if (grid && appendedEntries.length) {
        grid.insertAdjacentHTML("beforeend", appendedEntries.map(({ item, index }) => searchCardHtml(item, index, blurMature)).join(""));
        if (lazyStatus) lazyStatus.outerHTML = lazyStatusHtml();
        scrollExplorerTo(scrollTopBeforeRender);
        wireTileVideoLazyload();
        return;
      }
      if (grid) {
        if (lazyStatus) lazyStatus.outerHTML = lazyStatusHtml();
        return;
      }
    }
    panel.innerHTML = `
      <div id="civitaiSearchGrid" class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-3">
        ${visibleEntries.map(({ item, index }) => searchCardHtml(item, index, blurMature)).join("")}
      </div>
      ${lazyStatusHtml()}
    `;
    if (append) scrollExplorerTo(scrollTopBeforeRender);
    wireTileVideoLazyload();
  }

  function currentSearchSignature() {
    const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
    const tagFromInput = rawInput.startsWith("#") ? normalizeTagValue(rawInput) : "";
    const query = isCivitaiUrl(rawInput) || tagFromInput ? "" : normalizedCivitaiQuery(rawInput);
    return JSON.stringify({
      query,
      tag: tagFromInput || selectedTag(),
      types: el("civitaiTypeFilter")?.value || "",
      base_model: el("civitaiBaseModelFilter")?.value || "",
      sort: el("civitaiSortFilter")?.value || "Newest",
      nsfw: !!el("civitaiNsfwToggle")?.checked,
      installed: installedOnlyEnabled(),
    });
  }

  async function startFreshSearch() {
    clearTimeout(searchDebounceTimer);
    const panel = el("civitaiResultPanel");
    if (panel) panel.innerHTML = searchLoadingHtml("Searching Civitai...");
    viewMode = "explorer";
    currentDetail = null;
    searchCursor = null;
    currentItems = [];
    autoLoadMoreCount = 0;
    explorerScrollTop = 0;
    return search(false);
  }

  async function resolve() {
    const payload = formPayload();
    if (!payload.url) {
      return startFreshSearch();
    }
    if (activeSearchController) activeSearchController.abort();
    const runId = ++searchRunId;
    activeSearchController = new AbortController();
    try {
      status("Resolving...");
      const panel = el("civitaiResultPanel");
      if (panel) panel.innerHTML = searchLoadingHtml("Resolving Civitai model...");
      resolved = await post("/civitai/resolve", payload, activeSearchController.signal);
      if (runId !== searchRunId) return;
      renderResult(resolved, false);
      status("Resolved.");
    } finally {
      if (runId === searchRunId) activeSearchController = null;
    }
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
    const signature = currentSearchSignature();
    lastSearchSignature = signature;
    activeSearchController = new AbortController();
    loadingMore = true;
    status(append ? "Loading more..." : "Searching Civitai...");
    const rawInput = el("civitaiUrlInput")?.value?.trim() || "";
    const tagFromInput = rawInput.startsWith("#") ? normalizeTagValue(rawInput) : "";
    const query = isCivitaiUrl(rawInput) || tagFromInput ? "" : normalizedCivitaiQuery(rawInput);
    const tag = tagFromInput || selectedTag();
    const payload = {
      ...basePayload(),
      query,
      tag,
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
      if (runId !== searchRunId || (!append && signature !== lastSearchSignature)) return;
      searchCursor = result.metadata?.nextCursor || null;
      renderSearch(result.items || [], append);
      if (!append) {
        explorerScrollTop = 0;
        scrollExplorerTo(0);
      }
      status(`${currentItems.length} model(s) loaded.`);
    } catch (error) {
      if (error?.name === "AbortError") {
        if (runId === searchRunId) {
          const panel = el("civitaiResultPanel");
          if (panel && viewMode === "explorer" && !currentItems.length) panel.innerHTML = searchLoadingHtml("Updating Civitai filters...");
        }
        return;
      }
      if (runId === searchRunId) {
        const panel = el("civitaiResultPanel");
        const message = String(error?.message || "Civitai search failed. Try again or adjust filters.").slice(0, 180);
        if (panel) panel.innerHTML = searchErrorHtml(message);
      }
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
      setInstalledOnly(false);
      resetSearchState({ clearPanel: true });
      updateTypeButtons();
      renderTagChips();
      syncDownloadJobs();
      status("Idle");
      startFreshSearch().catch(() => status("Browse unavailable."));
    },
    close() {
      setInstalledOnly(false);
      resetSearchState({ clearPanel: true });
      status("Idle");
      el("civitaiModal")?.classList.add("hidden");
      closeMediaModal();
    },
    resolve() {
      resolve().catch((error) => {
        status("Resolve failed.");
        window.showToast?.("Civitai Resolve Failed", String(error.message || error).slice(0, 180));
      });
    },
    submit() {
      submitSearchInput().catch((error) => {
        status("Submit failed.");
        window.showToast?.("Civitai Submit Failed", String(error.message || error).slice(0, 180));
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
      return startFreshSearch().catch((error) => {
        status("Search failed.");
        window.showToast?.("Civitai Search Failed", String(error.message || error).slice(0, 180));
      });
    },
    loadMore() {
      if (!searchCursor) {
        status("No more results.");
        return Promise.resolve();
      }
      return search(true).catch((error) => {
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
        nsfw_level: item.nsfw_level ?? version.nsfw_level,
        nsfwLevel: item.nsfwLevel ?? item.nsfw_level ?? version.nsfwLevel ?? version.nsfw_level,
        tags: item.tags || [],
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
    setTag(value) {
      setTagFilter(value);
    },
    clearFilters() {
      clearSearchFilters();
    },
    clearToken() {
      localStorage.removeItem(tokenStorageKey);
      const token = el("civitaiTokenInput");
      if (token) token.value = "";
      window.showToast?.("Civitai API Key Cleared", "Saved token removed from this device.");
    },
  };
})();
