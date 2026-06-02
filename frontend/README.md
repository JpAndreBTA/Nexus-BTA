# Nexus BTA Frontend

New responsive web app track for Nexus BTA.

## Development

```powershell
cd path\to\NexusBTA\frontend
npm install
npm run dev
```

The dev server runs at `http://127.0.0.1:3000` and proxies backend calls to `http://127.0.0.1:7861`.

## Validation

```powershell
npm run check
```

This runs TypeScript checking and a production build.

With the FastAPI backend running and the React build served at `/app`:

```powershell
cd path\to\NexusBTA
.\runtime\.venv\Scripts\python.exe .\frontend\scripts\smoke_app.py
```

## Migration Rule

The legacy UI remains at `/ui`. React becomes official only after the migrated pages are feature-complete and responsive.
