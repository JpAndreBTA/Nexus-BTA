/// <reference types="@rsbuild/core/types" />

interface ImportMetaEnv {
  readonly NEXUS_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
