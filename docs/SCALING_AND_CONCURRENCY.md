# Scaling & Concurrency Architecture

> **Target:** 2,000–3,000 concurrent organizational users on a single GCP **e2-standard-4** VM (4 vCPU, 16 GB RAM, ~$100–140/month).
>
> This document **supplements** [ARCHITECTURE.md](./ARCHITECTURE.md). It does not replace it.

---

## Table of Contents

1. [Infrastructure Sizing](#1-infrastructure-sizing)
2. [Concurrent File Upload Handling](#2-concurrent-file-upload-handling)
3. [Concurrent Chat Handling](#3-concurrent-chat-handling)
4. [Database Connection Pooling](#4-database-connection-pooling)
5. [Caching Strategy](#5-caching-strategy)
6. [Rate Limiting](#6-rate-limiting)
7. [WebSocket at Scale](#7-websocket-at-scale)
8. [Monitoring & Observability](#8-monitoring--observability)
9. [Error Handling & Resilience](#9-error-handling--resilience)
10. [Security Hardening](#10-security-hardening)
11. [Docker Compose Resource Limits](#11-docker-compose-resource-limits)
12. [New Files, Models & Dependencies](#12-new-files-models--dependencies)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Infrastructure Sizing

### VM Specification

| Component | Value |
|-----------|-------|
| **Machine type** | e2-standard-4 |
| **vCPU** | 4 |
| **RAM** | 16 GB |
| **Boot disk** | 100 GB SSD (pd-ssd) |
| **OS** | Ubuntu 22.04 LTS |
| **Region** | asia-south1 (Mumbai) |
| **Monthly cost** | ~$100–140 |

### Memory Budget (16 GB Total)

| Service | Allocation | Notes |
|---------|-----------|-------|
| **OS + Docker overhead** | 1.0 GB | Kernel, journald, containerd |
| **PostgreSQL** | 1.5 GB | `shared_buffers=512MB` + work_mem |
| **Redis** | 768 MB | `maxmemory=512MB` + overhead |
| **Qdrant** | 3.0 GB | Vector index + HNSW graphs |
| **FastAPI (3 workers)** | 1.5 GB | 500 MB per Uvicorn worker |
| **Celery prefork (3 workers)** | 2.4 GB | 800 MB per worker (Unstructured, EasyOCR) |
| **Celery gevent (1 worker)** | 0.5 GB | 100 greenlets for IO-bound embedding |
| **Nginx** | 0.1 GB | Static files + reverse proxy |
| **Flower + Prometheus exporters** | 0.2 GB | Monitoring |
| **Headroom** | 5.0 GB | Spike absorption, OS cache, temp processing |
| **Total** | **16.0 GB** | |

### CPU Budget (4 vCPU)

| Process | Typical CPU | Peak CPU |
|---------|------------|----------|
| Uvicorn (3 workers) | 0.5 cores | 1.5 cores |
| Celery prefork (3) | 0.3 cores idle, 3.0 peak | 3.0 cores |
| PostgreSQL | 0.2 cores | 0.5 cores |
| Redis | 0.1 cores | 0.3 cores |
| Qdrant | 0.2 cores | 0.8 cores |

> **Key constraint:** CPU-heavy document processing (OCR, PDF extraction) competes with API serving. The two-pool Celery strategy (Section 2) is critical to avoid starving HTTP requests during upload spikes.

### Realistic Capacity

| Scenario | Concurrent users | Notes |
|----------|-----------------|-------|
| **Comfortable** | ~1,500 | Low queue backlog, <200ms p95 API latency |
| **Stretched** | ~2,000 | Aggressive caching, some queue delay |
| **Maximum** | ~2,500–3,000 | Requires rate limiting + graceful degradation |
| **Beyond single VM** | 3,000+ | Horizontal scaling required (see Section 1.1) |

### 1.1 When to Scale Horizontally

If any of these thresholds are sustained for >5 minutes, consider a second VM:

- CPU utilization >85% across all cores
- Redis memory >90% of maxmemory
- Celery queue depth >500 tasks
- API p95 latency >2s
- PostgreSQL active connections >80% of `max_connections`

**Horizontal scaling path:**
```
Single VM (current)     →    Two VMs + Cloud SQL
e2-standard-4              VM1: API + WebSocket + Redis
~$140/mo                   VM2: Celery workers + Qdrant
                           Cloud SQL: managed PostgreSQL
                           ~$250-300/mo
```

---

## 2. Concurrent File Upload Handling

### 2.1 Two-Pool Celery Strategy

Document processing has two distinct phases with different concurrency profiles:

```
Upload → [prefork pool: CPU-bound] → [gevent pool: IO-bound] → Done
          Text extraction, OCR         Embedding API calls,
          PDF parsing                  Qdrant upserts
```

**Prefork pool** (CPU-bound work):
```python
# celery_app.py - CPU-heavy tasks
celery_app.conf.update(
    worker_pool="prefork",
    worker_concurrency=3,          # Match vCPU count minus 1 for API
    worker_max_tasks_per_child=50,  # Restart after 50 tasks (memory leak guard)
    worker_prefetch_multiplier=1,   # Don't prefetch beyond current capacity
    task_soft_time_limit=300,       # 5 min soft limit
    task_time_limit=600,            # 10 min hard kill
)
```

**Gevent pool** (IO-bound work):
```python
# Second worker process
# celery -A app.celery_app worker --pool=gevent --concurrency=100 -Q embeddings-queue
celery_app.conf.update(
    # Gevent worker handles embedding API calls and vector store writes
    # 100 greenlets can await Gemini embedding API concurrently
)
```

### 2.2 Queue Architecture

```
                    ┌─────────────────┐
  Upload API ──────►│ documents-queue  │──► Prefork Worker (x3 processes)
                    └─────────────────┘         │
                                                │ on_success
                    ┌─────────────────┐         ▼
                    │ embeddings-queue │◄── chain to embedding task
                    └─────────────────┘
                            │
                            ▼
                    Gevent Worker (100 greenlets)
                            │
                    ┌───────┴───────┐
                    ▼               ▼
              Gemini Embed    Qdrant Upsert
                    │               │
                    └───────┬───────┘
                            ▼
                    ┌─────────────────┐
                    │  dead-letter-q   │◄── Failed after 3 retries
                    └─────────────────┘
```

**Task routing:**
```python
celery_app.conf.task_routes = {
    "app.worker.extract_document":    {"queue": "documents-queue"},
    "app.worker.chunk_document":      {"queue": "documents-queue"},
    "app.worker.generate_embeddings": {"queue": "embeddings-queue"},
    "app.worker.upsert_vectors":      {"queue": "embeddings-queue"},
}
```

### 2.3 Backpressure

Prevent the system from accepting more uploads than it can process:

```python
# In upload endpoint
async def check_queue_depth(redis: Redis) -> bool:
    """Reject uploads if queue is saturated."""
    queue_len = await redis.llen("documents-queue")
    if queue_len > 200:  # Max pending tasks
        raise HTTPException(
            status_code=503,
            detail="System busy. Please retry in a few minutes.",
            headers={"Retry-After": "120"}
        )
```

**Backpressure thresholds:**

| Queue | Soft limit (warn) | Hard limit (reject) |
|-------|-------------------|---------------------|
| `documents-queue` | 100 | 200 |
| `embeddings-queue` | 300 | 500 |

### 2.4 Progress Tracking (Redis Pub/Sub → WebSocket Bridge)

Real-time upload progress via Redis pub/sub bridged to WebSocket:

```python
# Worker side: publish progress
def publish_progress(document_id: str, stage: str, percent: int):
    redis_client.publish(
        f"doc_progress:{tenant_id}",
        json.dumps({
            "document_id": document_id,
            "stage": stage,       # "extracting", "chunking", "embedding", "indexing"
            "percent": percent,
            "timestamp": time.time()
        })
    )

# API side: bridge pub/sub → WebSocket
async def progress_listener(tenant_id: str):
    """Subscribe to Redis pub/sub and forward to WebSocket connections."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"doc_progress:{tenant_id}")
    async for message in pubsub.listen():
        if message["type"] == "message":
            await manager.broadcast_to_tenant(
                json.loads(message["data"]),
                tenant_id
            )
```

**Progress stages:**

| Stage | Percent range | Description |
|-------|--------------|-------------|
| `uploading` | 0–10% | File received by API |
| `extracting` | 10–40% | Text extraction (CPU-heavy) |
| `chunking` | 40–60% | Semantic chunking |
| `embedding` | 60–90% | Gemini embedding API calls |
| `indexing` | 90–99% | Qdrant vector upsert |
| `completed` | 100% | Ready for search |

### 2.5 Retry & Dead Letter Strategy

```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,      # 30s, 60s, 120s (exponential)
    retry_backoff=True,
    retry_backoff_max=300,       # Max 5 min between retries
    reject_on_worker_lost=True,  # Re-queue if worker crashes
    acks_late=True,              # Acknowledge after completion
)
def extract_document(self, document_id: str):
    try:
        # ... extraction logic
    except TransientError as exc:
        # Retry on transient errors (API timeouts, temp disk full)
        self.retry(exc=exc)
    except PermanentError as exc:
        # Send to dead letter queue for manual review
        send_to_dead_letter(document_id, str(exc))
        update_document_status(document_id, "failed", error_message=str(exc))
```

**Dead letter queue handling:**
```python
celery_app.conf.task_routes.update({
    "app.worker.dead_letter_handler": {"queue": "dead-letter-q"},
})

@celery_app.task
def dead_letter_handler(document_id: str, error: str):
    """Log failed task for admin review. Notify tenant admins."""
    logger.error(f"Dead letter: doc={document_id}, error={error}")
    # Store in DB for admin dashboard
    # Optionally notify via WebSocket
```

---

## 3. Concurrent Chat Handling

### 3.1 Gemini API Rate Limit Mitigation

Google Gemini has per-minute rate limits. With 2,000+ users, concurrent chat requests will hit these limits without mitigation.

**LLM Request Pool (Token Bucket + Semaphore):**

```python
import asyncio
from dataclasses import dataclass, field
from time import monotonic

@dataclass
class TokenBucket:
    """Rate limiter for Gemini API calls."""
    rate: float = 60          # Requests per minute (adjust to your Gemini quota)
    max_tokens: float = 60
    tokens: float = field(default=60)
    last_refill: float = field(default_factory=monotonic)

    def consume(self) -> float:
        """Try to consume a token. Returns wait time if empty."""
        now = monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * (self.rate / 60))
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / (self.rate / 60)

class LLMRequestPool:
    """Manages concurrent access to Gemini API."""

    def __init__(self, max_concurrent: int = 10, rpm: int = 60):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.bucket = TokenBucket(rate=rpm, max_tokens=rpm)

    async def execute(self, coro):
        """Execute an LLM call with rate limiting and concurrency control."""
        async with self.semaphore:
            wait = self.bucket.consume()
            if wait > 0:
                await asyncio.sleep(wait)
            return await coro

# Global pool instance
llm_pool = LLMRequestPool(max_concurrent=10, rpm=60)
```

**Usage in chat endpoint:**
```python
@router.post("/chat")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    result = await llm_pool.execute(
        agent.ainvoke(initial_state)
    )
    return ChatResponse(message=result["messages"][-1].content)
```

### 3.2 SSE Streaming for Chat

Server-Sent Events for real-time streaming of agent responses:

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user)):
    async def event_generator():
        async for event in agent.astream_events(initial_state, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            elif event["event"] == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
            elif event["event"] == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name']})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### 3.3 Conversation Context Storage

**Hot/cold storage pattern:**

| Layer | Store | TTL | Use case |
|-------|-------|-----|----------|
| **Hot** | Redis hash | 30 min | Active conversation context (last 10 messages) |
| **Cold** | PostgreSQL | Permanent | Full conversation history |

```python
# Redis keys for conversation context
# conv:{user_id}:messages  → List of last 10 messages (JSON)
# conv:{user_id}:metadata  → Hash with tenant_id, last_active, message_count

async def get_conversation_context(user_id: str, redis: Redis, db: AsyncSession):
    """Load conversation context, falling back to DB if Redis misses."""
    # Try hot cache first
    cached = await redis.lrange(f"conv:{user_id}:messages", 0, 9)
    if cached:
        return [json.loads(m) for m in cached]

    # Fall back to cold storage
    messages = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
    )
    context = [msg.to_dict() for msg in messages.scalars()]

    # Warm the cache
    pipe = redis.pipeline()
    for msg in reversed(context):
        pipe.rpush(f"conv:{user_id}:messages", json.dumps(msg))
    pipe.expire(f"conv:{user_id}:messages", 1800)  # 30 min TTL
    await pipe.execute()

    return context
```

### 3.4 Circuit Breaker for Gemini API

Prevent cascading failures when Gemini is down or degraded:

```python
import time

class CircuitBreaker:
    """Circuit breaker for external API calls."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject immediately
    HALF_OPEN = "half_open" # Testing if service recovered

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                self.state = self.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow limited requests
        return True

    def record_success(self):
        if self.state == self.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = self.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        self.success_count = 0
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN

# Global circuit breaker for Gemini
gemini_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    success_threshold=2,
)
```

---

## 4. Database Connection Pooling

### 4.1 SQLAlchemy Async Pool Tuning

```python
# database.py
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,

    # Pool settings tuned for e2-standard-4
    pool_size=10,            # Base connections per worker
    max_overflow=5,          # Burst capacity per worker
    pool_timeout=30,         # Wait max 30s for a connection
    pool_recycle=1800,       # Recycle connections every 30 min
    pool_pre_ping=True,      # Verify connections before use

    # Connection args
    connect_args={
        "server_settings": {
            "statement_timeout": "30000",        # 30s query timeout
            "idle_in_transaction_session_timeout": "60000",  # 60s idle txn timeout
        }
    },
)
```

**Connection math (3 Uvicorn workers):**
```
Base:      3 workers × 10 pool_size  = 30 connections
Burst:     3 workers ×  5 overflow   = 15 connections
Max total:                           = 45 connections

PostgreSQL max_connections should be ≥ 60
(45 API + 10 Celery + 5 monitoring/admin headroom)
```

### 4.2 PostgreSQL Server Tuning

```ini
# postgresql.conf overrides (via Docker environment or config mount)

# Memory (constrained to 1.5GB allocation)
shared_buffers = 512MB
effective_cache_size = 1GB
work_mem = 4MB                  # Per-operation sort memory
maintenance_work_mem = 128MB    # VACUUM, CREATE INDEX

# Connections
max_connections = 60            # Tight for 16GB VM
idle_in_transaction_session_timeout = 60s

# WAL
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 512MB

# Query planner
random_page_cost = 1.1          # SSD-optimized
effective_io_concurrency = 200  # SSD-optimized

# Logging
log_min_duration_statement = 500  # Log queries slower than 500ms
log_connections = on
log_disconnections = on
```

### 4.3 PgBouncer Recommendation

For deployments approaching 2,000+ users, add PgBouncer as a connection pooler in front of PostgreSQL:

```yaml
# docker-compose addition
pgbouncer:
  image: edoburu/pgbouncer:1.21.0
  environment:
    DATABASE_URL: postgres://postgres:postgres@postgres:5432/chatbot
    POOL_MODE: transaction
    MAX_CLIENT_CONN: 200
    DEFAULT_POOL_SIZE: 20
    MIN_POOL_SIZE: 5
    RESERVE_POOL_SIZE: 5
    RESERVE_POOL_TIMEOUT: 3
    SERVER_IDLE_TIMEOUT: 300
    SERVER_LIFETIME: 3600
  ports:
    - "6432:6432"
  depends_on:
    postgres:
      condition: service_healthy
```

When PgBouncer is enabled, update `DATABASE_URL` to point to PgBouncer (port 6432) and increase SQLAlchemy `pool_size` since PgBouncer handles the actual connection multiplexing.

### 4.4 Redis Async Connection Pool

```python
# redis_pool.py
import redis.asyncio as aioredis

redis_pool = aioredis.ConnectionPool.from_url(
    settings.redis_url,
    max_connections=50,          # Shared across all async contexts
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30,
)

redis_client = aioredis.Redis(connection_pool=redis_pool)
```

### 4.5 Qdrant gRPC Connection

Use gRPC (port 6334) instead of REST for lower latency and better throughput:

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    grpc_port=6334,
    prefer_grpc=True,          # Use gRPC for all operations
    timeout=10,                # 10s timeout
)
```

---

## 5. Caching Strategy

### 5.1 Redis LRU Cache Layers

```python
# Redis maxmemory policy
# Set in redis.conf or via Docker command:
# redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

| Cache key pattern | TTL | Eviction | Use case |
|-------------------|-----|----------|----------|
| `session:{user_id}` | 30 min | LRU | JWT session data, user profile |
| `search:{tenant_id}:{query_hash}` | 10 min | LRU | Vector search results |
| `doc_meta:{document_id}` | 60 min | LRU | Document metadata (filename, status, chunk_count) |
| `conv:{user_id}:messages` | 30 min | LRU | Recent conversation messages |
| `tenant:{tenant_id}:version` | None | Never | Cache invalidation counter |
| `rate:{user_id}:chat` | 60 sec | TTL | Rate limit sliding window |
| `rate:{tenant_id}:upload` | 60 sec | TTL | Tenant upload rate limit |

### 5.2 Cache Invalidation via Tenant Version Counters

Avoid stale data when documents are added or deleted:

```python
async def invalidate_tenant_cache(tenant_id: str, redis: Redis):
    """Increment tenant version to invalidate all search caches."""
    await redis.incr(f"tenant:{tenant_id}:version")

async def get_cached_search(tenant_id: str, query: str, redis: Redis):
    """Get cached search result, respecting tenant version."""
    version = await redis.get(f"tenant:{tenant_id}:version") or "0"
    cache_key = f"search:{tenant_id}:v{version}:{hashlib.md5(query.encode()).hexdigest()}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

async def set_cached_search(tenant_id: str, query: str, results: list, redis: Redis):
    """Cache search results with tenant version."""
    version = await redis.get(f"tenant:{tenant_id}:version") or "0"
    cache_key = f"search:{tenant_id}:v{version}:{hashlib.md5(query.encode()).hexdigest()}"
    await redis.setex(cache_key, 600, json.dumps(results))  # 10 min TTL
```

**Invalidation triggers:**
- Document upload completed → `invalidate_tenant_cache(tenant_id)`
- Document deleted → `invalidate_tenant_cache(tenant_id)`
- Document re-indexed → `invalidate_tenant_cache(tenant_id)`

### 5.3 Estimated Cache Hit Ratios

| Cache layer | Expected hit ratio | Impact |
|-------------|-------------------|--------|
| Session cache | 95%+ | Avoids DB lookup on every authenticated request |
| Search results | 30–50% | Many users in same org ask similar questions |
| Document metadata | 80%+ | Metadata is read-heavy, write-rare |
| Conversation context | 70%+ | Active users send multiple messages in a session |

---

## 6. Rate Limiting

### 6.1 Sliding Window Rate Limiter

```python
import time
import redis.asyncio as aioredis

async def sliding_window_rate_limit(
    redis: aioredis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Redis-based sliding window rate limiter.

    Returns:
        (allowed: bool, remaining: int)
    """
    now = time.time()
    window_start = now - window_seconds

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)   # Remove expired entries
    pipe.zadd(key, {str(now): now})                # Add current request
    pipe.zcard(key)                                # Count requests in window
    pipe.expire(key, window_seconds + 1)           # Auto-cleanup
    results = await pipe.execute()

    current_count = results[2]
    allowed = current_count <= limit
    remaining = max(0, limit - current_count)

    if not allowed:
        # Remove the request we just added since it's denied
        await redis.zrem(key, str(now))

    return allowed, remaining
```

### 6.2 Rate Limit Tiers

| Endpoint | Scope | Limit | Window | HTTP header |
|----------|-------|-------|--------|-------------|
| `POST /api/chat/` | Per user | 20 req | 1 min | `X-RateLimit-Limit-Chat` |
| `POST /api/chat/` | Per tenant | 200 req | 1 min | `X-RateLimit-Limit-Tenant-Chat` |
| `POST /api/documents/upload` | Per user | 10 req | 1 min | `X-RateLimit-Limit-Upload` |
| `POST /api/documents/upload` | Per tenant | 50 req | 1 min | `X-RateLimit-Limit-Tenant-Upload` |
| `POST /api/auth/login` | Per IP | 5 req | 1 min | `X-RateLimit-Limit-Auth` |
| All endpoints | Global | 5,000 req | 1 min | `X-RateLimit-Limit-Global` |

### 6.3 Rate Limit Middleware

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract identifiers
        user_id = getattr(request.state, "user_id", None)
        tenant_id = getattr(request.state, "tenant_id", None)
        client_ip = request.client.host

        # Check global limit
        allowed, remaining = await sliding_window_rate_limit(
            redis_client, "rate:global", 5000, 60
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Global rate limit exceeded"},
                headers={"Retry-After": "60"}
            )

        # Check endpoint-specific limits
        if request.url.path.startswith("/api/chat") and user_id:
            allowed, remaining = await sliding_window_rate_limit(
                redis_client, f"rate:{user_id}:chat", 20, 60
            )
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Chat rate limit exceeded"},
                    headers={"Retry-After": "30"}
                )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
```

---

## 7. WebSocket at Scale

### 7.1 Redis Pub/Sub-Backed ConnectionManager

The current `ConnectionManager` in `websockets.py` keeps connections in a per-process dict. This breaks when running multiple Uvicorn workers because each worker has its own memory space. Solution: back the manager with Redis pub/sub.

```python
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket
import redis.asyncio as aioredis

class ScalableConnectionManager:
    """
    WebSocket connection manager backed by Redis pub/sub.
    Enables cross-worker message delivery.
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
        self.local_connections: Dict[str, Set[WebSocket]] = {}
        self._pubsub_task: asyncio.Task | None = None

    async def start(self):
        """Start the Redis pub/sub listener."""
        self._pubsub_task = asyncio.create_task(self._listen())

    async def _listen(self):
        """Listen for messages from other workers via Redis pub/sub."""
        pubsub = self.redis.pubsub()
        await pubsub.psubscribe("ws:tenant:*")
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                tenant_id = message["channel"].split(":")[-1]
                data = json.loads(message["data"])
                await self._deliver_local(tenant_id, data)

    async def _deliver_local(self, tenant_id: str, data: dict):
        """Deliver message to locally-connected WebSockets."""
        if tenant_id in self.local_connections:
            dead = set()
            for ws in self.local_connections[tenant_id]:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.add(ws)
            self.local_connections[tenant_id] -= dead

    async def connect(self, websocket: WebSocket, tenant_id: str, user_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if tenant_id not in self.local_connections:
            self.local_connections[tenant_id] = set()
        self.local_connections[tenant_id].add(websocket)

        # Track in Redis for connection counting
        await self.redis.hincrby("ws:connections", tenant_id, 1)

    async def disconnect(self, websocket: WebSocket, tenant_id: str):
        """Remove a WebSocket connection."""
        if tenant_id in self.local_connections:
            self.local_connections[tenant_id].discard(websocket)
            if not self.local_connections[tenant_id]:
                del self.local_connections[tenant_id]
        await self.redis.hincrby("ws:connections", tenant_id, -1)

    async def broadcast_to_tenant(self, message: dict, tenant_id: str):
        """Publish message to all workers serving this tenant."""
        await self.redis.publish(
            f"ws:tenant:{tenant_id}",
            json.dumps(message)
        )
```

### 7.2 WebSocket Authentication

```python
from fastapi import WebSocket, WebSocketException, status
from jose import jwt, JWTError

async def authenticate_websocket(websocket: WebSocket) -> dict:
    """
    Authenticate WebSocket via query param token.
    Usage: ws://host/ws?token=<jwt_token>
    """
    token = websocket.query_params.get("token")
    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return {
            "user_id": payload["user_id"],
            "tenant_id": payload["tenant_id"],
        }
    except JWTError:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
```

### 7.3 Connection Limits & Heartbeat

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max connections per user | 3 | Browser tabs |
| Max connections per tenant | 500 | Organizational limit |
| Heartbeat interval | 30 sec | Detect dead connections |
| Heartbeat timeout | 90 sec | 3 missed heartbeats → disconnect |
| Message size limit | 64 KB | Prevent memory abuse |

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    auth = await authenticate_websocket(websocket)
    tenant_id = auth["tenant_id"]
    user_id = auth["user_id"]

    # Check connection limits
    tenant_conns = int(await redis_client.hget("ws:connections", tenant_id) or 0)
    if tenant_conns >= 500:
        await websocket.close(code=1013, reason="Too many connections")
        return

    await manager.connect(websocket, tenant_id, user_id)
    try:
        while True:
            # Heartbeat with 90s timeout
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=90.0
            )
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except (asyncio.TimeoutError, Exception):
        await manager.disconnect(websocket, tenant_id)
```

---

## 8. Monitoring & Observability

### 8.1 Monitoring Stack

```
┌────────────┐    ┌───────────────┐    ┌──────────┐
│ Prometheus │◄───│ Exporters     │    │ Grafana  │
│            │    │  - FastAPI    │    │          │
│  (scrape)  │    │  - Celery     │    │ (dashb.) │
│            │    │  - PostgreSQL │    │          │
│            │    │  - Redis      │    │          │
│            │    │  - Node       │    │          │
└─────┬──────┘    └───────────────┘    └────┬─────┘
      │                                      │
      └──────────── datasource ──────────────┘
```

**Key exporters:**

| Exporter | Port | Metrics |
|----------|------|---------|
| `prometheus-fastapi-instrumentator` | 8000 `/metrics` | Request latency, status codes, in-flight |
| `flower` (Celery) | 5555 `/metrics` | Task success/fail rates, queue lengths, worker utilization |
| `postgres_exporter` | 9187 | Connection count, query duration, cache hit ratio |
| `redis_exporter` | 9121 | Memory usage, connected clients, command latency |
| `node_exporter` | 9100 | CPU, memory, disk, network |

### 8.2 Structured JSON Logging

```python
import logging
import json
import sys
from uuid import uuid4

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "tenant_id": getattr(record, "tenant_id", None),
            "user_id": getattr(record, "user_id", None),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# Configure root logger
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
```

### 8.3 Request ID Tracing

```python
from fastapi import Request
from uuid import uuid4

class RequestIDMiddleware:
    """Inject a unique request ID into every request for tracing."""

    async def __call__(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        # Inject into logging context
        logger = logging.getLogger("app")
        logger = logging.LoggerAdapter(logger, {"request_id": request_id})

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

### 8.4 Health Checks

```python
@app.get("/health")
async def health_check():
    """Basic liveness probe."""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Deep readiness probe - checks all dependencies."""
    checks = {}

    # PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)}"

    # Redis
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    # Qdrant
    try:
        qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)}"

    # Celery
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active()
        checks["celery"] = "ok" if active else "no workers"
    except Exception as e:
        checks["celery"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks}
    )
```

### 8.5 Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU usage | >70% 5min | >90% 5min | Scale up or reduce Celery concurrency |
| Memory usage | >80% | >90% | Investigate leaks, reduce cache TTLs |
| API p95 latency | >1s | >3s | Check DB queries, enable caching |
| API error rate (5xx) | >1% | >5% | Check logs, circuit breaker state |
| Celery queue depth | >100 | >300 | Add workers or increase concurrency |
| PostgreSQL connections | >50 | >55 | Connection leak, check PgBouncer |
| Redis memory | >400MB | >480MB | Reduce TTLs, check eviction policy |
| Qdrant latency p95 | >200ms | >500ms | Check collection size, optimize HNSW |
| Disk usage | >70% | >85% | Clean uploads, archive old data |
| WebSocket connections | >1,500 | >2,000 | Check for connection leaks |
| Gemini circuit breaker | HALF_OPEN | OPEN | Check Gemini API status page |

---

## 9. Error Handling & Resilience

### 9.1 Circuit Breaker for Gemini

See Section 3.4 for implementation. State transitions:

```
         5 failures                2 successes
CLOSED ──────────► OPEN ──30s──► HALF_OPEN ──────────► CLOSED
  ▲                  │                │
  │                  │                │ failure
  │                  └────────────────┘
  │
  └──── success resets failure count
```

**Behavior in each state:**

| State | Behavior | User experience |
|-------|----------|-----------------|
| CLOSED | Normal operation | Full chat functionality |
| OPEN | Reject immediately | "AI service temporarily unavailable. Please try again shortly." |
| HALF_OPEN | Allow 1 request through | Normal for lucky request; others still get error |

### 9.2 Retry with Exponential Backoff

All external API calls use retry with exponential backoff + jitter:

```python
import asyncio
import random

async def retry_with_backoff(
    coro_factory,          # Callable that returns a coroutine
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError),
):
    """Retry an async operation with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable_exceptions as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay + jitter)
```

### 9.3 Dead Letter Queue

Failed tasks after all retries go to a dead letter queue:

```python
# Dead letter queue flow:
#
# 1. Task fails after max_retries
# 2. on_failure callback sends to dead-letter-q
# 3. Admin can inspect via Flower or API endpoint
# 4. Admin can retry or discard

@celery_app.task(bind=True)
def on_task_failure(self, exc, task_id, args, kwargs, einfo):
    """Callback when a task permanently fails."""
    dead_letter = {
        "task_id": task_id,
        "task_name": self.name,
        "args": args,
        "kwargs": kwargs,
        "exception": str(exc),
        "traceback": str(einfo),
        "failed_at": datetime.utcnow().isoformat(),
    }
    redis_client.lpush("dead-letter-q", json.dumps(dead_letter))
    redis_client.ltrim("dead-letter-q", 0, 999)  # Keep last 1000

# Admin endpoint to view dead letters
@router.get("/admin/dead-letters")
async def list_dead_letters(admin: User = Depends(require_admin)):
    items = await redis_client.lrange("dead-letter-q", 0, 49)
    return [json.loads(item) for item in items]
```

### 9.4 Graceful Degradation

When under heavy load, the system should degrade gracefully rather than crash:

| Condition | Degradation | User experience |
|-----------|------------|-----------------|
| Gemini API down (circuit open) | Disable chat, uploads still work | "Chat temporarily unavailable" banner |
| Celery queue >200 | Reject new uploads, chat still works | "Upload queue full, try again later" |
| Redis down | Fall back to DB for auth, disable caching | Slower responses but functional |
| Qdrant down | Disable search-dependent chat | "Search unavailable, try again shortly" |
| High CPU (>90%) | Reduce Celery concurrency dynamically | Slower document processing |

---

## 10. Security Hardening

### 10.1 File Upload Validation

Never trust the file extension or Content-Type header. Validate at the byte level:

```python
import magic  # python-magic library

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

async def validate_upload(file: UploadFile) -> None:
    """Validate uploaded file before processing."""
    # 1. Size check (read in chunks to avoid memory issues)
    size = 0
    chunk = await file.read(8192)
    header_bytes = chunk  # Save first bytes for MIME detection
    while chunk:
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise HTTPException(413, "File exceeds 100MB limit")
        chunk = await file.read(8192)
    await file.seek(0)

    # 2. MIME type detection from file bytes (not extension)
    detected_mime = magic.from_buffer(header_bytes, mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            415,
            f"Unsupported file type: {detected_mime}. "
            f"Allowed: PDF, DOCX, PPTX, XLSX, PNG, JPEG"
        )

    # 3. Filename sanitization
    safe_name = re.sub(r'[^\w\-.]', '_', file.filename)
    if '..' in safe_name or safe_name.startswith('.'):
        raise HTTPException(400, "Invalid filename")
```

### 10.2 WebSocket Security

See Section 7.2 for JWT-based WebSocket authentication. Additional measures:

- **Origin validation:** Check the `Origin` header against allowed domains
- **Message size limit:** 64 KB max per message (prevents memory abuse)
- **Rate limiting:** Max 30 messages per minute per connection
- **Connection timeout:** Close idle connections after 90 seconds of no heartbeat

### 10.3 API Key Rotation for Gemini

```python
# Support multiple API keys for rotation and load distribution
class GeminiKeyRotator:
    """Rotate between multiple Gemini API keys."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_key(self) -> str:
        async with self._lock:
            key = self.keys[self._index]
            self._index = (self._index + 1) % len(self.keys)
            return key

# Config: GOOGLE_API_KEYS=key1,key2,key3
key_rotator = GeminiKeyRotator(settings.google_api_keys.split(","))
```

**Key rotation strategy:**
- Maintain 2–3 active API keys
- Rotate keys quarterly
- If a key is rate-limited, the rotator naturally distributes load to others
- Monitor per-key usage in Gemini dashboard

---

## 11. Docker Compose Resource Limits

Production-tuned resource limits for 16 GB VM:

```yaml
# docker-compose.prod.yml
services:
  postgres:
    image: postgres:15-alpine
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1536M
        reservations:
          cpus: "0.25"
          memory: 512M
    command: >
      postgres
        -c shared_buffers=512MB
        -c effective_cache_size=1GB
        -c work_mem=4MB
        -c maintenance_work_mem=128MB
        -c max_connections=60
        -c max_wal_size=512MB
        -c random_page_cost=1.1
        -c effective_io_concurrency=200
        -c log_min_duration_statement=500
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 3072M
        reservations:
          cpus: "0.2"
          memory: 1024M
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS: 2
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 768M
        reservations:
          cpus: "0.1"
          memory: 256M
    command: >
      redis-server
        --maxmemory 512mb
        --maxmemory-policy allkeys-lru
        --save 60 1000
        --appendonly yes
        --appendfsync everysec
        --tcp-backlog 511
        --timeout 300
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1536M
        reservations:
          cpus: "0.5"
          memory: 512M
    command: >
      uvicorn app.main:app
        --host 0.0.0.0
        --port 8000
        --workers 3
        --loop uvloop
        --http httptools
        --limit-concurrency 200
        --limit-max-requests 10000
        --timeout-keep-alive 30
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/chatbot
      - REDIS_URL=redis://redis:6379
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env
    volumes:
      - ./uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  celery_worker_cpu:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2560M
        reservations:
          cpus: "0.5"
          memory: 1024M
    command: >
      celery -A app.celery_app worker
        --pool=prefork
        --concurrency=3
        --queues=documents-queue
        --max-tasks-per-child=50
        --loglevel=info
        --without-heartbeat
        --without-mingle
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/chatbot
      - REDIS_URL=redis://redis:6379
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env
    volumes:
      - ./uploads:/app/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  celery_worker_io:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.1"
          memory: 256M
    command: >
      celery -A app.celery_app worker
        --pool=gevent
        --concurrency=100
        --queues=embeddings-queue
        --loglevel=info
        --without-heartbeat
        --without-mingle
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/chatbot
      - REDIS_URL=redis://redis:6379
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 256M
    command: celery -A app.celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 128M
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    depends_on:
      - backend

volumes:
  postgres_data:
    driver: local
  qdrant_data:
    driver: local
  redis_data:
    driver: local
```

### Resource Summary

| Container | CPU limit | Memory limit | Memory reservation |
|-----------|-----------|-------------|-------------------|
| postgres | 1.0 | 1,536 MB | 512 MB |
| qdrant | 1.0 | 3,072 MB | 1,024 MB |
| redis | 0.5 | 768 MB | 256 MB |
| backend (3 workers) | 2.0 | 1,536 MB | 512 MB |
| celery_worker_cpu (prefork) | 2.0 | 2,560 MB | 1,024 MB |
| celery_worker_io (gevent) | 0.5 | 512 MB | 256 MB |
| flower | 0.25 | 256 MB | — |
| nginx | 0.25 | 128 MB | — |
| **Total limits** | **7.5** | **10,368 MB** | — |

> **Note:** CPU limits intentionally exceed 4.0 because not all containers peak simultaneously. Docker CPU limits cap individual containers, not total host usage. Memory limits total ~10 GB, leaving ~6 GB for OS, buffers, and spike absorption.

---

## 12. New Files, Models & Dependencies

### 12.1 New Files to Create

```
backend/app/
├── middleware/
│   ├── __init__.py
│   ├── rate_limit.py          # Sliding window rate limiter middleware
│   └── request_id.py          # Request ID injection middleware
├── resilience/
│   ├── __init__.py
│   ├── circuit_breaker.py     # Circuit breaker for external APIs
│   ├── retry.py               # Exponential backoff retry utility
│   └── llm_pool.py            # Token bucket + semaphore for Gemini
├── cache/
│   ├── __init__.py
│   ├── redis_pool.py          # Async Redis connection pool
│   └── cache_manager.py       # Tenant-versioned cache get/set/invalidate
├── monitoring/
│   ├── __init__.py
│   ├── health.py              # Health check endpoints (liveness + readiness)
│   ├── metrics.py             # Prometheus metrics setup
│   └── logging_config.py      # Structured JSON logging configuration
├── websockets_v2.py           # Redis pub/sub-backed ConnectionManager
├── security/
│   ├── __init__.py
│   └── file_validator.py      # MIME-based file validation
└── worker.py                  # Updated with two-pool task routing
```

### 12.2 New Database Models

```python
# models/chat_message.py
class ChatMessage(Base):
    """Persisted chat message for conversation history (cold storage)."""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    role = Column(String(20), nullable=False)     # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)         # Source citations for assistant messages
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("ix_chat_messages_user_created", "user_id", "created_at"),
        Index("ix_chat_messages_tenant", "tenant_id"),
    )


# models/rate_limit_event.py (optional, for analytics)
class RateLimitEvent(Base):
    """Track rate limit violations for monitoring."""
    __tablename__ = "rate_limit_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    tenant_id = Column(UUID(as_uuid=True), nullable=True)
    endpoint = Column(String(255), nullable=False)
    ip_address = Column(String(45), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 12.3 New Dependencies

Add to `backend/requirements.txt`:

```
# Concurrency & Performance
uvloop>=0.19.0                  # Fast asyncio event loop
httptools>=0.6.1                # Fast HTTP parsing for uvicorn
gevent>=24.2.0                  # Green threads for IO-bound Celery pool

# Monitoring
prometheus-fastapi-instrumentator>=6.1.0  # FastAPI metrics exporter
flower>=2.0.1                   # Celery monitoring dashboard

# Connection Pooling
redis[hiredis]>=5.0.1           # Redis with C parser (faster)

# Security
python-magic>=0.4.27            # MIME type detection from file bytes

# Resilience (no extra deps - pure Python implementations)
# circuit_breaker.py, retry.py, llm_pool.py are custom modules
```

### 12.4 New Configuration Files

```
project_root/
├── nginx/
│   └── nginx.conf              # Reverse proxy, WebSocket upgrade, rate limiting
├── prometheus/
│   └── prometheus.yml          # Scrape config for all exporters
├── grafana/
│   └── dashboards/
│       └── overview.json       # Pre-built dashboard (optional)
└── docker-compose.prod.yml     # Production overrides with resource limits
```

---

## 13. Configuration Reference

All concrete configuration values in one place. Every value is tuned for e2-standard-4 (4 vCPU, 16 GB).

### Application Server

| Parameter | Value | Notes |
|-----------|-------|-------|
| Uvicorn workers | 3 | `2 × CPU + 1` minus 1 for Celery headroom |
| Uvicorn loop | `uvloop` | ~2x faster than default asyncio loop |
| Uvicorn HTTP parser | `httptools` | ~2x faster than h11 |
| Uvicorn concurrency limit | 200 | Max concurrent connections per worker |
| Uvicorn max requests | 10,000 | Worker restart after 10k requests (memory leak guard) |
| Uvicorn keep-alive timeout | 30s | Match Nginx `proxy_read_timeout` |

### Celery Workers

| Parameter | Value | Notes |
|-----------|-------|-------|
| Prefork concurrency | 3 | CPU-bound tasks (extraction, OCR) |
| Prefork max tasks per child | 50 | Memory leak prevention |
| Prefork prefetch multiplier | 1 | Don't hoard tasks |
| Gevent concurrency | 100 | IO-bound tasks (embedding API, vector upsert) |
| Soft time limit | 300s (5 min) | Raises `SoftTimeLimitExceeded` |
| Hard time limit | 600s (10 min) | `SIGKILL` the worker process |
| Max retries | 3 | Per task |
| Retry backoff base | 30s | 30s → 60s → 120s |
| Retry backoff max | 300s (5 min) | Cap on retry delay |

### PostgreSQL

| Parameter | Value | Notes |
|-----------|-------|-------|
| `shared_buffers` | 512 MB | ~25% of PostgreSQL allocation |
| `effective_cache_size` | 1 GB | OS cache hint |
| `work_mem` | 4 MB | Per-sort operation |
| `maintenance_work_mem` | 128 MB | VACUUM, CREATE INDEX |
| `max_connections` | 60 | 45 API + 10 Celery + 5 admin |
| `max_wal_size` | 512 MB | WAL rotation threshold |
| `random_page_cost` | 1.1 | SSD-tuned |
| `log_min_duration_statement` | 500 ms | Slow query logging |

### SQLAlchemy Connection Pool

| Parameter | Value | Notes |
|-----------|-------|-------|
| `pool_size` | 10 | Per Uvicorn worker |
| `max_overflow` | 5 | Burst capacity per worker |
| `pool_timeout` | 30s | Wait for connection from pool |
| `pool_recycle` | 1800s (30 min) | Prevent stale connections |
| `pool_pre_ping` | `True` | Connection health check before use |
| Total max connections | 45 | 3 workers × (10 + 5) |

### Redis

| Parameter | Value | Notes |
|-----------|-------|-------|
| `maxmemory` | 512 MB | LRU eviction beyond this |
| `maxmemory-policy` | `allkeys-lru` | Evict least-recently-used keys |
| Max pool connections | 50 | Shared across async contexts |
| `save` | `60 1000` | Snapshot every 60s if 1000+ writes |
| `appendonly` | `yes` | AOF persistence enabled |
| `appendfsync` | `everysec` | Fsync every second |
| Socket timeout | 5s | Fail fast on unresponsive Redis |
| Health check interval | 30s | Pool-level connection health |

### Qdrant

| Parameter | Value | Notes |
|-----------|-------|-------|
| Memory limit | 3 GB | Docker container limit |
| gRPC port | 6334 | Lower latency than REST |
| `prefer_grpc` | `True` | Client-side setting |
| Max search threads | 2 | Cap CPU usage |
| Client timeout | 10s | Per-request timeout |

### Rate Limits

| Scope | Limit | Window |
|-------|-------|--------|
| Chat per user | 20 req | 1 min |
| Chat per tenant | 200 req | 1 min |
| Upload per user | 10 req | 1 min |
| Upload per tenant | 50 req | 1 min |
| Auth per IP | 5 req | 1 min |
| Global | 5,000 req | 1 min |

### WebSocket

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max connections per user | 3 | Browser tab limit |
| Max connections per tenant | 500 | Organization limit |
| Heartbeat interval | 30s | Client sends ping |
| Heartbeat timeout | 90s | Server disconnects if no ping |
| Message size limit | 64 KB | Prevents memory abuse |

### Caching TTLs

| Cache layer | TTL | Notes |
|-------------|-----|-------|
| Session | 30 min | JWT user profile cache |
| Search results | 10 min | Invalidated by tenant version counter |
| Document metadata | 60 min | Read-heavy, write-rare |
| Conversation context | 30 min | Active chat session |
| Rate limit windows | 60 sec | Auto-expire |

### Backpressure Thresholds

| Queue | Soft limit (log warning) | Hard limit (reject 503) |
|-------|--------------------------|-------------------------|
| `documents-queue` | 100 pending tasks | 200 pending tasks |
| `embeddings-queue` | 300 pending tasks | 500 pending tasks |

### Circuit Breaker (Gemini)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Failure threshold | 5 | Consecutive failures to open |
| Recovery timeout | 30s | Time before testing recovery |
| Success threshold | 2 | Successes in half-open to close |

### Monitoring

| Metric endpoint | Port | Path |
|-----------------|------|------|
| FastAPI metrics | 8000 | `/metrics` |
| Flower (Celery) | 5555 | `/metrics` |
| PostgreSQL exporter | 9187 | `/metrics` |
| Redis exporter | 9121 | `/metrics` |
| Node exporter | 9100 | `/metrics` |
| Health (liveness) | 8000 | `/health` |
| Health (readiness) | 8000 | `/health/ready` |

---

## Summary: What Changes from Current Codebase

| Current state | After this plan |
|---------------|-----------------|
| Single Celery worker, default pool | Two workers: prefork (CPU) + gevent (IO) |
| In-memory `ConnectionManager` | Redis pub/sub-backed `ScalableConnectionManager` |
| No connection pooling tuning | SQLAlchemy pool + PgBouncer ready |
| No caching | Redis LRU with tenant version counters |
| No rate limiting | Sliding window per-user/tenant/global |
| No circuit breaker | Circuit breaker on Gemini + retry with backoff |
| No monitoring | Prometheus + Grafana + Flower + structured logging |
| No file validation | MIME-type detection from bytes |
| No resource limits in Docker | Full resource limits for 16 GB VM |
| `uvicorn` with defaults | `uvicorn` with uvloop, httptools, 3 workers |
| Single Celery queue | Three queues: documents, embeddings, dead-letter |
