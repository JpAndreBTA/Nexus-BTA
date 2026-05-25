# Nexus BTA Frontend

New responsive web app track for Nexus BTA.

## Development

```powershell
cd D:\NexusBTA\frontend
npm install
npm run dev
```

The dev server runs at `http://127.0.0.1:3000` and proxies backend calls to `http://127.0.0.1:7861`.

## Validation

```powershell
npm run check
```

This runs TypeScript checking and a production build.

## Migration Rule

The legacy UI remains at `/ui`. React becomes official only after the migrated pages are feature-complete and responsive.
