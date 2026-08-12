# Subscription-CLI Mode (local) — Architecture

> **Status: planned.** None of the subscription-CLI runtime described here has shipped. This
> document records the fork's intended direction and its current baseline relative to
> upstream. The only code-generation path that works today is the upstream API-key mode.

## 1. Purpose

Adapt this fork (`salemaziel/screenshot-to-code`) so a single screenshot-to-code service can
be driven two ways **without storing developer API keys**:

- by a human through the existing browser WebUI, and
- by AI agents through a local HTTP job API and an MCP server.

Generation is planned to be backed by a **locally installed, subscription-authenticated coding
CLI** — starting with **Codex CLI** — rather than by developer API keys for
OpenAI/Anthropic/Gemini.

## 2. Chosen architecture

The existing React/Vite WebUI and FastAPI backend are preserved. A **runner adapter** boundary
replaces the API-key-specific model selection and in-process provider agent:

```text
Browser WebUI (React/Vite) ── WebSocket ──┐
                                          ├──► FastAPI backend (local control plane)
Local HTTP job API (agents/scripts) ──────┤         │
MCP server (agent-native) ────────────────┘         ▼
                                             Runner adapters
                                                └── Codex CLI (planned first runner)
```

Key decisions:

- **Keep the existing WebUI and backend.** The React/Vite frontend and FastAPI backend remain
  the human-facing experience and the canonical local control plane.
- **Runner adapters are subprocess adapters.** The planned `CodexCliRunner` launches the
  locally authenticated `codex exec` binary in a tight, ephemeral workspace, attaches the
  screenshot, requests schema-validated source output, and maps its event stream into the
  existing WebSocket protocol.
- **Codex CLI is the first (planned) runner.** Claude Code and Antigravity are planned
  follow-on adapters behind the same contract; no second adapter ships before Codex is stable.
- **The HTTP job API and the MCP server are consumers of the same backend.** Neither owns
  provider credentials nor invokes a vendor CLI itself; both call the loopback FastAPI service.
- **Not an OpenAI-compatible proxy.** Screenshot-to-code is a job/workflow API, not a
  chat-completions API; modeling it as a proxy would misrepresent the service and lose useful
  semantics.
- **Hermes plugin deferred.** A future Hermes desktop plugin may package/configure the MCP
  client or add setup/status/history UI, but it is not the runtime core.

## 3. Status: what exists vs. what is planned

### Exists today (shipped upstream)

- React/Vite frontend + FastAPI backend (`frontend/`, `backend/`).
- API-key generation via `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`
  (plus `REPLICATE_API_KEY` for image editing/generation).
- WebSocket generation pipeline: `backend/routes/generate_code.py` /
  `frontend/src/generateCode.ts`.
- Settings UI for API keys: `frontend/src/components/settings/SettingsTab.tsx`.
- `GET /api/capabilities`, which currently reports only screenshot-preview availability
  (`backend/routes/capabilities.py`).
- Provider session protocol (`backend/agent/providers/base.py`) and agent/file-state lifecycle
  (`backend/agent/engine.py`).

### Planned (not yet implemented)

- **Runner layer** (`backend/runners/`): a provider-neutral request/result contract, a runner
  registry, capability probing, and the `CodexCliRunner` subprocess adapter with an ephemeral
  workspace and a read-only sandbox. This directory does not exist yet.
- **WebSocket path through the runner**, preserving the existing client event names so the
  first UI change is minimal.
- **Local HTTP job API** (`POST /v1/jobs`, status/result reads, idempotent cancel) plus a
  provider-agnostic client CLI. Neither exists yet.
- **MCP server** exposing `screenshot_to_code_generate`, `screenshot_to_code_status`, and
  `screenshot_to_code_cancel`. It is a client of the local service, not a launcher of coding
  CLIs. Does not exist yet.
- **WebUI simplification**: replace API-key fields and the legacy model dropdown with runner
  availability and a "sign in to Codex CLI on this host" prerequisite; hide/disable video and
  image-editing controls that v0.1 does not support.
- **Operations and threat-model docs** (`docs/operations/local-runner.md`,
  `docs/security/threat-model.md`).

## 4. Hard boundaries

- **No developer API keys.** The service never asks for, stores, proxies, or displays
  OpenAI/Anthropic/Gemini developer API keys.
- **No vendor credential access.** The runner only executes the vendor-supported local CLI and
  relies on its existing credential store. It never reads, copies, exports, scrapes, or
  forwards CLI OAuth tokens.
- **Local/loopback by default.** The service binds to loopback. Exposing it through a reverse
  proxy or Tailscale is deferred and would require upstream authentication and an explicit
  allowlist; a local subscription executor is not exposed anonymously to a LAN/WAN.
- **User-controlled host.** This mode is intended for a single user on a host they control. It
  is not a multi-tenant or hosted product, and no such claim is made.
- **Respect provider terms and rate limits.** This is not a way to evade subscription limits,
  vendor policies, or rate limits. At most one active run per provider is planned by default,
  and provider errors (including quota/rate-limit) are surfaced plainly.
- **No undocumented provider protocols.** No direct use of a vendor's undocumented OAuth/API
  internals.

## 5. Upstream baseline and divergence

- This repository is a fork of
  [`abi/screenshot-to-code`](https://github.com/abi/screenshot-to-code) (MIT).
- As of this work session, the fork (`origin/main`) and upstream (`upstream/main`) share commit
  `d026163f586dfa8c5c10d28c36edd59a9d3b0e88` — the fork has not yet diverged.
- That is a point-in-time snapshot, not a claim of continuing synchronization. Divergence is
  introduced only by the planned changes above.

## 6. Out of scope for v0.1

- Multi-provider or multi-variant comparison; 2–4 concurrent variants.
- Video input, generated-image editing, background removal, provider-native asset extraction.
- Remote multi-tenant hosting.
- A Hermes desktop plugin as a runtime dependency.
