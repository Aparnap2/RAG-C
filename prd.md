# MCP‑First, Type‑C RAG Platform: Final PRD, Workflows, Design, and Requirements

Note: This framework is API‑agnostic by design. All external connectivity (data sources, operational tools, actions) occurs exclusively through MCP servers discovered and invoked at runtime over JSON‑RPC 2.0 via stdio or HTTP+SSE, not through hard‑coded vendor SDKs or REST clients. Hybrid retrieval uses reciprocal rank fusion (RRF) and a cross‑encoder reranker as opinionated defaults to raise accuracy and stability.[1][2][3][4][5][6]

## 1) Product Requirements Document (PRD)

### 1.1 Vision
Deliver a cloud‑agnostic, connector‑first RAG platform where ingestion, lookups, and actions are performed via MCP tools/resources discovered at runtime, enabling continuous, governed data access without embedding vendor APIs in the engine; retrieval is hybrid (BM25+vector via RRF) with a cross‑encoder reranker; answers are grounded with citations, time‑aware graph facts, and strict ACLs.[2][4][1]

### 1.2 Goals and KPIs
- Accuracy: ≥80% precision@5; ≥70% recall@20 on golden sets; reranker lift ≥15% vs vector‑only baseline.[7][8][9]
- Latency: p95 ≤3s end‑to‑end with streaming tokens; p50 ≤1.8s.[4][10][11]
- Freshness: <5min ingestion→availability for sources supporting MCP streaming or incremental cursors.[12][1]
- Reliability: 99.9% query path availability; DLQ coverage and replay for all ingestion workers.[1][12]
- Portability: add/remove a source by registering/unregistering an MCP server; swap vector/text/graph stores via config only.[3][1]
- Security: 100% ACL enforcement in tests; auditable MCP call logs and permission scopes.[13][12]

### 1.3 Scope
- Phase 1: MCP host/registry, queue‑first ingestion, dual sinks (graph+text), hybrid retrieval (RRF) + cross‑encoder rerank, grounded generation, K8s/Helm deploy.[10][4][1]
- Phase 2: Temporal GraphRAG (validity windows + provenance), ACL passthrough/enforcement, expanded eval harness, multi‑tenant isolation.[5][4]
- Phase 3: Autoscaling, cost‑aware model routing, canary+rollback, hardened MCP adapters for enterprise data (e.g., CRM/Dataverse) and memory tools.[12][13]

### 1.4 Assumptions
- MCP is the only integration path; transports are stdio or HTTP+SSE with JSON‑RPC 2.0.[3][1][12]
- Vendor auth (OAuth/SSO) is handled by MCP servers; the host enforces tool allow‑lists and scopes.[13][12]
- Hybrid retrieval needs RRF for stability across scoring scales and sources.[6][14][4]

### 1.5 Functional Requirements
- R‑1 MCP connectivity: runtime discovery (tools/resources/prompts), schema validation, secure invocation, session handling, and logging over stdio or HTTP+SSE using JSON‑RPC 2.0.[1][3][12]
- R‑2 Ingestion: full and incremental sync via MCP tools; checkpointing, retries/backoff, DLQ; idempotent upserts.[12][1]
- R‑3 Indexing: chunking; embeddings; vector DB + BM25 index; reciprocal rank fusion and shard‑aware filtering.[4][5][6]
- R‑4 Graph: entity/relation extraction with provenance and temporal validity; semantic graph search; optional graph‑then‑text scoping.[15][5]
- R‑5 Retrieval: parallel vector and BM25; RRF fusion; cross‑encoder rerank; strict ACL/time filters; citation construction.[5][10][4]
- R‑6 Generation: grounding‑only responses with citations; refusal when evidence is insufficient; SSE streaming; model routing.[10][4]
- R‑7 Observability & Eval: OpenTelemetry traces, Prometheus metrics; golden‑set evaluation with regression gates (precision/recall, reranker lift, grounding coverage).[8][9][7]
- R‑8 Security & Compliance: per‑tenant isolation; encryption; MCP tool permission scopes; auditable invocation logs and lineage.[13][12]

### 1.6 Non‑Functional Requirements
- Performance: parallel candidate generation; batch/cached embeddings; rerank budget tunable (e.g., rerank top‑50).[9][10]
- Scalability: horizontal scale by queue depth; sharded indices; stateless workers.[6][4]
- Reliability: exactly‑once upserts via keys/checksums; durable queues; DLQ alerts.[1][12]
- Compliance: retained MCP logs; citations with validity windows; configurable retention.[12][13]

### 1.7 Acceptance Criteria
- Adding/removing a data source requires only MCP server registration changes; no engine code edits.[3][1]
- RRF+rerank improves precision ≥15% vs vector‑only on golden sets; CI gate enforces threshold.[7][4]
- ACL/time filters block violations pre‑retrieval and pre‑generation in automated tests.[13][12]
- Store swap (e.g., Qdrant→pgvector; OpenSearch→Elasticsearch) via config only; smoke tests pass.[14][4]

## 2) System Design (MCP‑first, Type‑C)

### 2.1 Architecture Overview
- MCP Layer: Host connects to multiple MCP servers; discovers tools/resources; invokes via JSON‑RPC 2.0 over stdio or HTTP+SSE; session management and event streams per spec.[3][1][12]
- Ingestion: MCP tools stream or page items; normalize→queue; retries/backoff; DLQ; idempotency.[1][12]
- Processing:
  - Normalizer: dedupe, PII scrub, ACL mapping from MCP scopes.
  - Dual Sinks: graph sink (temporal/provenance) and text sink (chunk→embed→index).
- Index Tier: pluggable vector store; BM25 text index; graph store; reciprocal rank fusion to merge vector+BM25.[4][5][6]
- Retrieval: parallel vector and BM25; RRF fusion; optional graph pre‑scope; cross‑encoder reranker; context packer.[5][10][4]
- Generation: model gateway; grounding‑only; citations; refusal on low evidence; SSE streaming.[10][4]
- Observability: traces across MCP→ingest→index→retrieve→generate; metrics; eval service with golden sets and regression gates.[9][7]
- Security: per‑tenant isolation; encryption; MCP tool scopes and allow‑lists; auditable logs of all tool invocations.[12][13]

### 2.2 Key Design Choices
- Single integration surface via MCP: no embedded vendor SDKs; stdio for local, HTTP+SSE for remote/multi‑tenant servers.[3][1][12]
- Hybrid retrieval with RRF: stable fusion across diverse scoring distributions; avoids calibration pitfalls; widely adopted in search engines.[14][6][4]
- Cross‑encoder rerank: highest accuracy for top‑N candidates; tunable for latency/quality tradeoffs.[11][8][10]
- Temporal GraphRAG: validity windows and provenance for time‑aware answers and auditability.[15][5]

### 2.3 Data Models (Canonical)
- Document: id, tenant_id, source_tool, source_id, content, metadata, acl, ts_source, ts_ingested, checksum, schema_version.[1][12]
- Chunk: chunk_id, doc_id, text, headers, tokens, embedding_version, metadata.[4][5]
- Graph:
  - Node: node_id, type, labels, summary/text, embeddings, provenance, tenant_id.
  - Edge: edge_id, src, dst, relation, t_valid_start, t_valid_end, confidence, provenance, tenant_id.[15][5]
- Citation: ref_type(chunk|edge), ref_id, spans/offsets or validity window, source_tool, timestamp[5][4].

### 2.4 APIs (Engine‑side, not vendors)
- POST /ingest/events: accepts normalized items (often from MCP ingestion workers) → enqueues; returns trace_id.[12][1]
- POST /query: {query, tenant_id, user_ctx, filters, use_graph, top_k, stream} → streams grounded answer + citations.[10][4]
- GET /metrics, /healthz, /readyz; Admin: /reindex, /promote_embedder, /switch_index (feature‑flagged).[4]

## 3) End‑to‑End Workflows

### 3.1 Continuous Ingestion (MCP‑Driven)
1) Discover tools/resources: enumerate connected MCP servers; fetch schemas/capabilities; register tool → collection mappings.[1][12]
2) Full then incremental sync: use MCP tool cursors, webhooks, or SSE resources; auth handled by MCP server; apply backoff on rate limits.[12][1]
3) Normalize: map to canonical Document/Episode; attach ACLs from tool scopes; PII scrub; compute checksum; produce to queue with key=(tenant_id, source_id).[1][12]
4) Dual fan‑out:
   - Graph sink: NER/RE → nodes/edges with t_valid/t_invalid + provenance; semantic graph embeddings.[5][15]
   - Text sink: structural chunking (200–400 tokens, overlap), batch embeddings, upsert to vector DB; update BM25 index; write chunk manifest.[5][10][4]
5) Error handling: retries with jitter; DLQ and alerts; replay from checkpoints.[12][1]

### 3.2 Query Serving
1) Pre‑filters: tenant + ACL + optional time window based on metadata (and, if needed, MCP permission lookups).[13][12]
2) Candidate generation: parallel vector top‑K and BM25 top‑K; metadata filters; reciprocal rank fusion; dedupe.[6][4][5]
3) Optional graph pre‑scope: entity link; 1–2 hop expansion to constrain candidate space; include graph snippets.[15][5]
4) Rerank: cross‑encoder over top‑N (e.g., 50) → top‑k (e.g., 5); features include recency and entity overlap.[8][9][10]
5) Compose and generate: pack contexts; enforce citations; refuse if evidence < threshold; stream tokens; log trace/metrics.[10][4]

### 3.3 Evaluation & Release
- Nightly eval: golden sets compute precision/recall, reranker lift, grounding coverage; store time series; compare variants.[7][8][9]
- CI gate: block merges on regression; connector smoke tests; shadow indexes for embedder/chunker upgrades.[14][4]
- Canary: 10% traffic with rollback on degradation; log MCP version/capability changes for audit.[13][12]

## 4) Detailed Requirements

### 4.1 MCP Connectivity
- Transport: stdio for local, HTTP+SSE for remote; JSON‑RPC 2.0 framing; init/session per spec; SSE event IDs for resume.[1][12]
- Discovery: list tools/resources/prompts; validate schemas; allow‑list tools per tenant; deny by default.[3][12][1]
- Security: per‑server auth; host‑enforced permission scopes; full invocation logs; isolate servers and sessions.[13][12]

### 4.2 Ingestion & Streaming
- Full and incremental modes using MCP cursors/webhooks; rate‑limit aware; retries/backoff; checkpoints; DLQ.[12][1]
- Idempotency: upsert keyed by (tenant_id, source_id, checksum); deterministic chunk manifests.[4][5]

### 4.3 Retrieval
- Hybrid search: parallel vector and BM25; reciprocal rank fusion; shard‑aware top‑K merge; tunable weights.[6][5][4]
- Reranker: cross‑encoder baseline; rerank budget configurable (top‑25/50); cache features and embeddings.[11][9][10]
- Filters: tenant, ACL, timestamps; “as‑of” queries leverage graph validity windows.[5][4]

### 4.4 Graph
- Entity linking and relation extraction; nodes/edges carry provenance and temporal validity; conflict resolution policy; graph embeddings for semantic search.[15][5]

### 4.5 Generation
- Grounding‑only; mandatory citations to chunks/edges; refusal on low evidence; SSE streaming; model routing by complexity/budget.[10][4]

### 4.6 Observability & Admin
- Tracing across MCP→ingest→index→retrieve→generate; correlation IDs; Prometheus metrics (p50/p95 latencies, queue lag, recall@k/precision@k, reranker lift, DLQ, freshness).[9][7][4]
- Admin: reindex jobs; promote/demote embedder/chunker versions; switch active index; feature flags.[14][4]

## 5) Technical Foundations and Rationale

- MCP transports and primitives: JSON‑RPC over stdio or HTTP+SSE; servers expose tools/resources/prompts; strict isolation and consent model for security.[3][1][12]
- RRF for hybrid: rank‑based fusion stabilizes across disparate scoring distributions and avoids score calibration pitfalls; used across major search stacks.[16][17][6][14][4][5]
- Cross‑encoder reranking: top accuracy in two‑stage pipelines; typical lifts of 15–30% relevance with manageable compute when capped to top‑N.[8][11][9][10]

This PRD, workflows, and design provide an API‑agnostic, MCP‑first RAG platform that can add or remove enterprise sources through protocol registration alone while delivering production retrieval quality via RRF and cross‑encoder reranking, time‑aware graph grounding, and rigorous evaluation gates.[4][10][1]

