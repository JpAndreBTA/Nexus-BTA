# Nexus-BTA — UI, backend, workflows e arquitetura: auditoria de 12/11/2026

> Baseline: main @ 89e8da08bb115fc2cbaa89635837c99ea50b05ca (commit de 30/08/2026). Este documento registra inspeção estática do repositório; NÃO é laudo de execução no Windows do usuário.
> Projeto independente de VEYLUM, Game3 e outras ferramentas MCP.

## Evidências verificadas

- 183 arquivos no tree; ~32,7 MB de blobs contabilizados (inclui dependências empacotadas).
- `index.html` = 2.455.626 bytes; `backend/nexus_backend/main.py` = 597.409; `backend/nexus_backend/workflows.py` = 469.996; `frontend/src/views/home/HomePage.tsx` = 120.919; `frontend/src/styles.css` = 59.117.
- `assets/vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm` = 11.453.626 bytes; `assets/vendor/spark/spark.module.min.js` = 5.018.702. Esses arquivos grandes não devem ser interpretados como vazamento de memória por si só.
- Há duas UIs: FastAPI entrega `/ui` diretamente de `index.html`; React/Rsbuild é entregue em `/app`. `frontend/MIGRATION.md` confirma ausência de paridade.
- Na baseline, `main.py` tinha `allow_origins=["*"]`, `allow_credentials=True`, API que altera caminhos, instala extensões e controla processos, além de `/inputs` e `/model-assets` montados publicamente via StaticFiles. Risco de exposição caso servidor/túnel seja acessível remotamente. CORS NÃO é autenticação.
- `frontend/src/shared/router.tsx` tinha filtros Civitai sem onClick e exibiu exemplos fictícios na grade de resultados; botão para download não existia no modal React.
- `frontend/src/views/settings/SettingsPage.tsx` era informativa, sem efetiva personalização.
- `frontend/src/views/home/InpaintCanvas.tsx` usa canvas máximo de 768 px, `toDataURL` em snapshots e estado com até 25 snapshots. Isso merece perfil de uso de memória e QA de alinhamento/máscara antes de otimização.
- `/api/gallery` faz `rglob` e `stat` de todos os caminhos, ordenando por mtime antes de limitar a 200 entradas. Complexidade ruim para acervos grandes; priorizar paginação e índice persistente.
- `frontend/package.json` contém dependências diretas `latest` (reprodutibilidade depende do lockfile), e não havia workflow de GitHub Actions na baseline.

## Matriz de criticidade e situação

| Área | Evidência | Situação | Saída |
| --- | --- | --- | --- |
| Distribuição frontend | Monolito legado 2,45 MB e React Home 121 KB | Confirmado | Carregamento lazy de 6 rotas no M1; decompor por função no M2 |
| CORS | Wildcard + credentials | Confirmado | Origens explícitas M1; autenticação remota pendente |
| Exposição em rede | StartLAN/StartTunnel expõem rotas administrativas | Confirmado risco | Bloqueio por reconhecimento explícito M1; gateway autenticado M3 |
| Dados em estáticos | /inputs e /model-assets servem caminhos | Confirmado | Segmentar downloads/preview com autorização, não habilitar remoto |
| Civitai modal | filtros sem ação / exemplos fictícios | Confirmado | Filtro e download no M1; testes em Civitai real pendentes |
| Settings React | somente status e checklist | Confirmado | Aparência persistente M1; editor de layout M2 |
| Inpaint | limite 768 e histórico base64 | Potencial gargalo | Perf, zoom/pan e coordenadas nativas M2 |
| Workflow editor | UI modelado em uma página grande | Potencial gargalo | Grafo incremental + validação de tipos e diffs M3 |
| Gallery | varredura integral + ordenação síncrona | Confirmado | Índice SQLite, thumbnails, cursores e observador M2 |
| ComfyUI | integrações e templates de versão fixa | Compatibilidade a verificar | Adapter por capabilities do /object_info M3 |
| 3D | README descreve experimental e bugs conhecidos | Confirmado pelo projeto | QA visual, GLB/mesh/texture pipeline M4 |
| CPU/RAM/VRAM | Sem números de perfil nesta auditoria | Não medido | Telemetria opt-in local e benchmarking M2 |

## Mudanças implementadas em audit/nexus-modernization-m1-2026-11

1. React routes são lazy; mantém `/ui` como fallback sem exclusão.
2. Zustand mantém chave `nexus-ui-state`, preservando dados antigos e adicionando temas, acentos, densidade, redução de movimento e reset.
3. CSS novo isolado em `frontend/src/styles/appearance.css` sem reescrever 59 KB legados.
4. Civitai: filtros acionáveis, sem conteúdo fictício, download explícito e polling com limpeza do timer.
5. Helper de segurança restringe origens cross-origin; modo LAN/tunnel demanda `NEXUS_UNSAFE_NETWORK_ACK=I_UNDERSTAND_THE_RISK` no ambiente. Isso NÃO torna remoto seguro. Não habilitar internet sem autenticação real.
6. Scaffold opcional Rust/WASM para operações determinísticas de máscara (não conectado ao caminho de produção ainda).

## Sequência de produção e critérios de aceite (pacotes grandes, sem microgates)

### M1 — Base de modernização (este PR)
- Rotas lazy, personalização persistente, correções Civitai, advertência de rede e docs.
- Gate: TypeScript + build Rsbuild, compilação Python, cargo tests da crate isolada, smoke de navegação e console, verificação LAN/tunnel bloqueado por padrão.
- Não migrar `/ui` ainda; nenhum modelo/tensor precisa ser executado nos testes de interface.

### M2 — Paridade UI e performance real
- Extrair `HomePage` em: Shell, ModelPicker, GenerationForm, InpaintEditor, VideoDirector, WorkflowPane, OutputPreview, JobMonitor, ModalHost. Estilos por componente + design tokens.
- Galeria com thumbnails criadas no background, índice paginado e descarga de recursos de vídeo/GLTF fora da viewport.
- Máscaras com coordenadas da imagem original, zoom/pan, undo/redo com orçamento explícito de RAM, payloads binários ou Blob URLs, revogação de object URLs.
- Modais acessíveis: Esc, foco restaurado, focus trap, portal root, scroll lock, estados loading/empty/error, cancelamento/retry e abort controllers.
- Preservar preset, LoRA, workflow, valores de CFG, steps, seeds, nodes e layout; migração reversível.
- Gate: regressão em fluxos `txt2img`, `img2img`, `inpaint`, `ControlNet`, `Gallery -> Extras`, carga de 1k/10k outputs; medições comparadas à baseline.

### M3 — Segurança e workflow engine
- API por routers versionados (`/api/v1/...`), serviços de job queue e contratos tipados; separar `main.py` e `workflows.py` sem trocar semântica.
- Acesso remoto autenticado via gateway/OIDC/token escopado, TLS e CSRF/Origin; não servir input, modelo ou configs arbitrariamente.
- Validar caminhos com raízes permitidas; tamanho/quantidade de upload; downloads por allowlist de hosts e hashes; sandbox de custom nodes; logs sem tokens, credenciais, prompts sensíveis ou dados pessoais.
- ComfyUI por adaptadores versionados: `/object_info`, `/prompt`, `/history/{id}`, `/view`, WebSocket; ids, replays e idempotência, cancelamento de job individual, reconnect e retentativas com backoff.
- Canonicalizar workflow API/UI de modo não destrutivo, reportando nós ausentes, mudanças de nome, versões, caminhos, modelos, LoRA e compatibilidade de GPU/VRAM.
- Gate: contrato de workflows multi-modelo, fuzz import JSON, rede/CSRF, instalação bloqueada sem permissão.

### M4 — Compatibilidade 2026: imagem, vídeo, áudio, 3D
- Catalogar por `model family + capability + installed dependencies + VRAM` em vez de listas fixas.
- Atualizações de ComfyUI e custom nodes em snapshot/branch de compatibilidade com rollback; nunca auto-merge de upstream.
- Revalidar SD/SDXL, Flux, Qwen, Z-Image, Anima, WAN, LTX, MiniMax H3/Music, inpaint/outpaint, image edit, ControlNet, video, 3D e máscaras; os nomes dos modelos não garantem workflows funcionais.
- 3D: revisão de TRELLIS2, TripoSplat, GLB, UV, texture-paint, previews e isolamento de pipelines de dependências.
- Gate: pacote de workflows por família com requisitos, faltantes, resultado visual de referência e rastreio de versão.

### M5 — Rust/WASM + desktop opcional
- Manter Python/ComfyUI/PyTorch para inferência e drivers; Rust para scheduler/orquestração, indexação, hashing, watcher e operações CPU-bound com profiling.
- Rust -> WebAssembly para transformações puras (máscara, recorte, histograma, thumbnails/tiling quando fizer sentido); fallback JS.
- Avaliar Tauri 2 estável em branch independente, IPC por capacidades mínimas; não migrar para Tauri 3 alpha em produção. Não enviar paths/credenciais do sistema ao frontend.
- Gate: benchmark A/B CPU/RAM/bundle, testes cross-platform e fallback completo.

### M6 — Release e migração oficial
- Feature flags e coexistência /app + /ui; importação/exportação de layouts versionada; backup de presets, workflows e storage.
- Apenas após paridade: /ui redireciona para React modernizado, com opt-in de rollback e compatibilidade de URLs.
- Gate: QA manual no Windows, build offline e pacote assinado com histórico de alterações.

## Atualizações upstream verificadas em 12/11/2026

- ComfyUI v0.45.0 publicado em 10/11/2026: novas integrações de imagem/vídeo/áudio e otimizações de memória, não instalar automaticamente sem validar custom nodes.
- ComfyUI v0.44.0 (04/11/2026): MiniMax-H3 e ProRes, mais controles de VAE.
- APIs oficiais ComfyUI: `POST /prompt`, `GET /history/{prompt_id}`, `GET /view`, `WebSocket /ws`; preferir public contract, não `/internal/*`.
- Tauri 2: capacidades por janela/webview e escopo explícito. Tauri 3 alpha existe, mas não é base para release.
- Monitorar ComfyUI-Manager, ComfyUI_frontend e modelos conforme dependências efetivamente instaladas. Novidade de GitHub é hipótese de integração até teste.

## Regedit e Windows

Não adicionar modificações automáticas e globais de registro (TDR, cache, memória, Defender, segurança/telemetria, HAGS, redes ou energia). Priorizar opções oficiais do Windows (Sistema > Tela > Gráficos, preferência de GPU por app), CUDA/ROCm/DirectML adequados ao dispositivo e perfis internos. Qualquer associação de arquivo no registry deve ser per-user, explícita, com export/rollback `.reg` e jamais como acelerador genérico de IA.

## Arquitetura alvo

```text
frontend/src/
  app/           # rotas, shell, atalhos, layout e themes
  features/      # image, video, audio, 3d, inpaint, models, gallery, workflows
  components/    # modal, preview, fields, graph canvas, toast, dialogs
  entities/      # presets, jobs, outputs, models, workflow capabilities
  shared/        # HTTP/WS contracts, i18n, design tokens, error boundary

backend/nexus_backend/
  api/v1/        # routers finos: jobs, models, workflows, files, settings
  services/      # Comfy adapter, model catalog, workflow validation
  security/      # auth, origin, paths, download integrity, capabilities
  persistence/   # SQLite jobs, gallery and user preferences (no secrets in API)
  workers/       # CPU-bound jobs/thumbnail/index, bounded queues
crates/nexus-image-core/ # pure CPU operations; WASM optional
desktop/         # future Tauri 2 shell (only after parity/security gates)
```

## Medições e continuidade

Medir startup (cold/warm), heap JS em 20min, memória e GPU em idle/gerando, arquivos e recursos por rota, latência de canvas, TTFB de galeria 200/10k, tempo de reabertura, preview GLTF, listeners/timers após navegar, cancel/retry e perda de estado. Registar valores de referência, regressões e ambiente (SO/GPU/Comfy release). Este checkpoint não afirma que houve testes com GPU, Windows, navegador real ou modelos instalados.
