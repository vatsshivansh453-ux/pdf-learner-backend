# PDF-LEARNER — OAuth2 setup

## What changed in the backend

- `utils/auth.py` (new) — Google + GitHub OAuth2 via Authlib, JWT cookie sessions.
- `utils/memory.py` — added a `users` table, and `user_id` on `chat_sessions`. Old DBs are auto-migrated on startup (`ALTER TABLE` runs once, harmlessly, if the column is missing).
- `utils/vector_store.py` — search results now carry `user_id` so answers can be scoped per person.
- `utils/rag.py` — `retrieve_context` / `generate_answer` / `stream_answer` accept a `user_id` and only pull chunks that user uploaded.
- `main.py` — every route now requires login (`Depends(get_current_user)`), plus new routes:
  - `GET /auth/login/google`, `GET /auth/login/github` — kick off OAuth
  - `GET /auth/callback/google`, `GET /auth/callback/github` — provider redirects here
  - `GET /auth/me` — who's logged in
  - `POST /auth/logout`
  - `GET /sessions/{id}/messages` — load a past chat to continue it
  - `DELETE /sessions/{id}` — delete a chat
  - Uploads/documents/asks are all filtered to the logged-in user's own files.

**Note on the shared FAISS index:** the vector index stays global (rebuilding a
separate index per user would be wasteful at this scale), but every chunk is
tagged with `user_id` and results are filtered before they're ever used to
answer a question — one user can never see another's PDF content. If you expect
many users with large numbers of documents, migrating to a proper multi-tenant
vector DB (e.g. Qdrant with per-user filters) would scale better than this
filter-after-search approach.

## 1. Install the new dependencies

```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Create OAuth apps

**Google** — https://console.cloud.google.com/apis/credentials
1. Create an OAuth 2.0 Client ID (type: Web application)
2. Authorized redirect URI: `http://localhost:8000/auth/callback/google`
3. Copy the Client ID and Client Secret into `.env`

**GitHub** — https://github.com/settings/developers → New OAuth App
1. Homepage URL: `http://localhost:5173`
2. Authorization callback URL: `http://localhost:8000/auth/callback/github`
3. Copy the Client ID and Client Secret into `.env`

Your `.env` already has `JWT_SECRET_KEY` / `SESSION_SECRET_KEY` filled in with
random values and `FRONTEND_URL` / `BACKEND_URL` set to the localhost defaults —
just paste in the four OAuth values.

## 3. Run it

```bash
# backend
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000

# frontend — any static file server works, e.g.:
cd frontend
python -m http.server 5173
```

Open `http://localhost:5173`. Don't open `index.html` directly via `file://` —
the OAuth redirect and cookie need a real origin that matches `FRONTEND_URL`.

If you deploy this somewhere other than localhost, update `FRONTEND_URL` /
`BACKEND_URL` in `.env`, the redirect URIs in the Google/GitHub app settings,
and `window.__PDF_LEARNER_API__` at the top of `frontend/index.html`'s
`<script>` (or just edit the `API_BASE` line directly) to point at your
backend's real URL.
