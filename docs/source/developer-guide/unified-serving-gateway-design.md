# Unified Serving Gateway: Multi-Rank Single-Endpoint Design

## Problem Statement

In large-scale LLM inference clusters (GB300/GB200 NVL72, B200/300 HGX), each model
rank — whether in-flight batching (IFB), disaggregated context, or disaggregated
generation — runs as a separate HTTP endpoint. This creates significant operational
complexity:

| Cluster | GPUs | Typical Model Config | Resulting Endpoints |
|---------|------|----------------------|---------------------|
| GB200 NVL72 | 72 | 4×ctx (TP=8) + 5×gen (TP=4) + orchestrator | **10 HTTP servers** |
| B200 HGX 8-node | 64 | 4×ctx (TP=8) + 4×gen (TP=4) + orchestrator | **9 HTTP servers** |
| B300 HGX 16-node | 128 | 8×ctx (TP=8) + 8×gen (TP=4) + orchestrator | **17 HTTP servers** |

Each endpoint is a separate `srun` step or process, each requiring:
- Unique hostname:port allocation and tracking
- Individual health monitoring
- Manual URL enumeration in disaggregated config YAML
- External load balancer configuration
- Slurm job script boilerplate per endpoint

The current disaggregated orchestrator (`trtllm-serve disaggregated`) reduces the
client-facing surface to one endpoint, but the backend complexity remains: every
context and generation instance is still an independent HTTP server that must be
pre-configured by URL.

**Goal:** Deploy multiple TensorRT-LLM ranks behind a single HTTP endpoint, with
internal load balancing, eliminating per-instance endpoint management.

---

## Current Architecture

```
                  External Client
                       │
                       ▼
            ┌────────────────────┐
            │  Disagg Orchestrator│   (trtllm-serve disaggregated)
            │  OpenAIDisaggServer │   FastAPI :8000
            │  ┌───────────────┐ │
            │  │  ctx_router   │ │   RoundRobin / LoadBalancing / KvCacheAware
            │  │  gen_router   │ │
            │  └───────┬───────┘ │
            └──────────┼─────────┘
              HTTP      │     HTTP
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ ctx:8001  │  │ gen:8002  │  │ gen:8003  │     Each is a separate
   │ TP=8      │  │ TP=4      │  │ TP=4      │     srun step with its
   │ (8 MPI    │  │ (4 MPI    │  │ (4 MPI    │     own HTTP server
   │  ranks)   │  │  ranks)   │  │  ranks)   │
   └───────────┘  └───────────┘  └───────────┘
```

### Pain Points

1. **Endpoint proliferation**: N context + M generation instances = N+M+1 processes
   to manage, each with its own port.

2. **Static URL configuration**: Disagg config YAML requires explicit
   `hostname:port` for every instance. Adding/removing capacity means editing YAML
   and restarting the orchestrator.

3. **HTTP overhead between co-located processes**: On NVL72, all GPUs are in one
   NVLink domain, yet the orchestrator talks to workers via HTTP/TCP — serialization,
   TCP stack, and context switches add unnecessary latency.

4. **Slurm complexity**: Each worker is a separate `srun` step. The launch script
   must coordinate port allocation, wait for readiness, and wire URLs into the
   disagg config.

5. **No topology awareness**: The router treats all servers as equivalent URLs with
   no knowledge of which workers share a node, NVLink domain, or IB fabric.

### Existing Building Blocks (to reuse)

| Component | File | Reuse |
|-----------|------|-------|
| Router abstraction | `serve/router.py:146` | Extend for direct dispatch |
| MPI comm split | `llmapi/disagg_utils.py:290` | Reuse for embedded mode |
| ZMQ IPC proxy | `executor/proxy.py` | Reuse for same-node transport |
| gRPC servicer | `grpc/grpc_servicer.py` | Reuse for internal protocol |
| Metadata server | `serve/metadata_server.py` | Reuse for dynamic discovery |
| Cluster storage | `serve/cluster_storage.py` | Reuse for worker registration |
| DisaggClusterManager | `serve/disagg_auto_scaling.py` | Reuse for watch/heartbeat |
| OpenAIDisaggService | `serve/openai_disagg_service.py` | Reuse ctx→gen orchestration |

---

## Proposed Design: Unified Serving Gateway

Two deployment modes sharing a common gateway architecture:

### Mode 1: Embedded Gateway (Single MPI Program)

All ranks launched in one `srun`. Rank 0 runs the HTTP gateway; other ranks are
organized into worker pools by role. Communication between gateway and worker
leaders uses MPI or IPC — no HTTP between co-located processes.

```
srun --ntasks=72 trtllm-serve gateway --config gateway.yaml
```

```
 ┌─────────────────── MPI_COMM_WORLD (72 ranks) ───────────────────┐
 │                                                                  │
 │  Rank 0: Gateway                                                 │
 │  ┌───────────────────────────────────────────────────────┐       │
 │  │  HTTP Server (:8000)                                  │       │
 │  │  ┌─────────────┐  ┌──────────────────────────────┐   │       │
 │  │  │ OpenAI API  │  │  Unified Router              │   │       │
 │  │  │ /v1/chat    │→ │  (role-aware, topology-aware) │   │       │
 │  │  │ /v1/complete│  │  + disagg orchestration       │   │       │
 │  │  └─────────────┘  └──────────┬───────────────────┘   │       │
 │  └──────────────────────────────┼────────────────────────┘       │
 │                    MPI Send/Recv │ or ZMQ IPC                    │
 │       ┌─────────────────────────┼──────────────────────┐        │
 │       ▼                         ▼                      ▼        │
 │  ┌──────────┐            ┌──────────┐           ┌──────────┐   │
 │  │Pool: ctx │            │Pool: gen │           │Pool: gen │   │
 │  │Leader: R1│            │Leader: R9│           │Leader:R13│   │
 │  │Ranks 1-8 │            │Ranks 9-12│           │Ranks13-16│   │
 │  │(TP=8)    │            │(TP=4)    │           │(TP=4)    │   │
 │  └──────────┘            └──────────┘           └──────────┘   │
 └─────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Single srun, single endpoint, zero HTTP overhead internally
- MPI collectives available for coordination and (optionally) data transfer
- Ideal for NVL72 where everything is in one NVLink domain
- Slurm job script is one line

**How it works:**
1. Gateway reads config, computes rank assignments per pool
   (reuses `split_world_comm` logic from `disagg_utils.py:290`)
2. Rank 0 spawns the HTTP server (FastAPI/uvicorn)
3. Pool leader ranks (first rank per pool) enter a **worker loop** that:
   - Receives serialized requests from the gateway via MPI `Recv` or ZMQ IPC
   - Submits to the local `LLM` / executor instance
   - Sends serialized responses back
4. Non-leader ranks enter the standard `MPICommExecutor` for TP/PP compute
5. The gateway's router selects a pool based on role + load, dispatches via the
   transport layer, and streams the response back to the HTTP client

### Mode 2: External Gateway (Multi-srun / Heterogeneous)

Workers are independent srun steps or processes. They register with a central
gateway via a lightweight internal protocol. The gateway is a standalone process
(CPU-only, no GPU required).

```
# Gateway (no GPU)
trtllm-serve gateway --config gateway.yaml --mode external

# Workers register themselves on startup
srun --ntasks=8 trtllm-serve worker --gateway gateway:9090 \
    --role context --model meta-llama/Llama-3.1-70B ...

srun --ntasks=4 trtllm-serve worker --gateway gateway:9090 \
    --role generation --model meta-llama/Llama-3.1-70B ...
```

```
     External Client
          │
          ▼
 ┌──────────────────────┐
 │  Gateway (:8000)     │
 │  ┌────────────────┐  │
 │  │  OpenAI API    │  │
 │  │  Unified Router│  │
 │  └───────┬────────┘  │
 │          │            │     Internal gRPC (:9090)
 │  ┌───────▼────────┐  │◄──── Worker registration
 │  │ Worker Pool    │  │      + heartbeat
 │  │ Manager        │  │
 │  └───────┬────────┘  │
 └──────────┼───────────┘
            │
  ┌─────────┼──────────────────────┐
  │  gRPC   │  gRPC                │  gRPC
  ▼         ▼                      ▼
┌──────┐  ┌──────┐              ┌──────┐
│Wkr 0 │  │Wkr 1 │  ...        │Wkr N │    Each is an independent
│ctx   │  │gen   │              │gen   │    srun step; only rank 0
│TP=8  │  │TP=4  │              │TP=4  │    talks to gateway
└──────┘  └──────┘              └──────┘
```

**Advantages:**
- Workers can be added/removed without restarting the gateway (elastic scaling)
- Heterogeneous configs: different models, different TP sizes, mixed GPU types
- Cross-node clusters (HGX over IB) where a single srun isn't practical
- Gateway can run on a CPU node

**How it works:**
1. Gateway starts with minimal config (port, routing strategy, optional static worker list)
2. Workers start normally (existing `trtllm-serve` + model), but additionally:
   - Connect to gateway's internal gRPC port on startup
   - Register: `{worker_id, role, hostname, port, model, tp_size, capacity_info}`
   - Send periodic heartbeats
3. Gateway maintains a `WorkerPool` with live workers, grouped by role
4. On each request, the router picks a worker. Two forwarding strategies:
   - **Proxy mode** (default): Gateway forwards the OpenAI HTTP request to the
     worker's existing HTTP server. Same as today's disagg server, but with
     auto-discovery instead of static URLs.
   - **Direct gRPC mode** (optional, lower latency): Gateway sends a serialized
     request over the internal gRPC channel; worker responds over the same channel.
     Avoids HTTP parsing overhead on the worker side.
5. Workers deregister on shutdown or are removed after heartbeat timeout.

---

## Component Design

### 1. Gateway Configuration (`gateway.yaml`)

Unified config that replaces the current split between `trtllm-serve` args and
`disagg_config.yaml`:

```yaml
# Unified Gateway Configuration
gateway:
  hostname: 0.0.0.0
  port: 8000
  mode: embedded           # or "external"
  internal_port: 9090      # for worker registration (external mode)

# Worker pool definitions (embedded mode — ranks assigned automatically)
# In external mode, these serve as expected-topology hints; actual pools
# are built from worker registrations.
worker_pools:
  - name: ctx_pool
    role: context
    num_instances: 2
    model: meta-llama/Llama-3.1-70B-Instruct
    tensor_parallel_size: 8
    pipeline_parallel_size: 1
    kv_cache_config:
      free_gpu_memory_fraction: 0.85
    cache_transceiver_config:
      backend: DEFAULT
    disable_overlap_scheduler: true

  - name: gen_pool
    role: generation
    num_instances: 4
    model: meta-llama/Llama-3.1-70B-Instruct
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    cache_transceiver_config:
      backend: DEFAULT

# Routing configuration
routing:
  context_strategy: load_balancing    # round_robin | load_balancing | kv_cache_aware
  generation_strategy: load_balancing
  context_strategy_args:
    use_tokens: true
  disaggregated: true                 # enable ctx→gen orchestration
  schedule_style: context_first       # context_first | generation_first
  conditional_disagg:
    max_local_prefill_length: 512

# Topology hints (for topology-aware routing)
topology:
  cluster_type: nvl72                 # nvl72 | hgx_ib
  # Optional: explicit node→GPU mapping for placement-aware routing
  # If omitted, auto-detected from MPI topology
```

### 2. Worker Pool Manager

Central component that tracks all backend workers, their health, load, and
topology position. Replaces the current pattern of static URL lists in
`DisaggServerConfig.server_configs`.

```python
class WorkerPool:
    """Manages a set of workers grouped by role."""

    def __init__(self, role: ServerRole):
        self.role = role
        self.workers: Dict[str, WorkerHandle] = {}  # worker_id → handle

    def add_worker(self, handle: WorkerHandle): ...
    def remove_worker(self, worker_id: str): ...
    def get_healthy_workers(self) -> List[WorkerHandle]: ...


class WorkerHandle:
    """Represents a single backend worker (one LLM instance)."""

    worker_id: str
    role: ServerRole              # context | generation | ifb
    transport: WorkerTransport    # how to communicate with this worker
    capacity: WorkerCapacity      # max_batch_size, max_num_tokens, etc.
    load: WorkerLoad              # current active_requests, active_tokens
    topology: WorkerTopology      # node_id, nvlink_domain, ib_subnet
    health: WorkerHealth          # last_heartbeat, consecutive_failures


class WorkerPoolManager:
    """Manages multiple WorkerPools, one per role."""

    def __init__(self):
        self.pools: Dict[ServerRole, WorkerPool] = {}

    def get_pool(self, role: ServerRole) -> WorkerPool: ...
    def register_worker(self, info: WorkerRegistration) -> WorkerHandle: ...
    def deregister_worker(self, worker_id: str): ...

    # For embedded mode: build pools from MPI rank assignments
    def init_from_mpi(self, config: GatewayConfig, comm: MPI.Comm): ...

    # For external mode: accept registrations via gRPC
    def init_from_registrations(self): ...
```

### 3. Transport Layer (Gateway ↔ Worker Communication)

Pluggable transport abstraction. The gateway sends requests and receives responses
through this layer, decoupled from the routing logic.

```python
class WorkerTransport(ABC):
    """Abstract transport for gateway ↔ worker communication."""

    @abstractmethod
    async def send_request(self, request: SerializedRequest) -> AsyncIterator[SerializedResponse]:
        """Send a request and yield streaming response chunks."""

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    async def close(self): ...


class MPITransport(WorkerTransport):
    """MPI-based transport for embedded mode (same MPI_COMM_WORLD).

    Uses non-blocking MPI_Isend/Irecv with asyncio integration.
    For streaming: worker sends token-level response messages with
    a terminal sentinel.
    """

    def __init__(self, comm: MPI.Comm, target_rank: int): ...

    async def send_request(self, request):
        # Serialize request → msgpack/protobuf bytes
        # MPI_Isend to target_rank with REQUEST tag
        # Loop MPI_Irecv with RESPONSE tag until sentinel
        ...


class ZMQIPCTransport(WorkerTransport):
    """ZMQ IPC transport for same-node workers (different processes).

    Reuses the existing ZMQ IPC pattern from executor/proxy.py.
    Lower latency than HTTP for co-located processes.
    """

    def __init__(self, ipc_addr: str): ...


class GRPCTransport(WorkerTransport):
    """gRPC transport for cross-node workers (external mode).

    Uses the internal gRPC service (not the public OpenAI API).
    Supports streaming via gRPC server-streaming RPCs.
    """

    def __init__(self, target_addr: str): ...


class HTTPProxyTransport(WorkerTransport):
    """HTTP proxy transport — forwards to worker's existing OpenAI HTTP server.

    This is the fallback transport, equivalent to what the current disagg
    orchestrator does. Zero code changes required on the worker side.
    """

    def __init__(self, target_url: str, session: aiohttp.ClientSession): ...
```

**Transport selection logic:**

| Scenario | Transport | Latency | Streaming |
|----------|-----------|---------|-----------|
| Embedded, same MPI_COMM_WORLD | `MPITransport` | Lowest (shared memory) | Via repeated MPI messages |
| Same node, different process | `ZMQIPCTransport` | Low (unix socket) | Native ZMQ streaming |
| Cross-node (external mode) | `GRPCTransport` | Medium (IB/ethernet) | gRPC server-streaming |
| Legacy / fallback | `HTTPProxyTransport` | Highest | SSE streaming |

### 4. Unified Router

Extends the existing `Router` abstraction to work with `WorkerHandle` objects
instead of URL strings. Incorporates topology awareness.

```python
class UnifiedRouter(ABC):
    """Routes requests to workers, not URLs."""

    @abstractmethod
    async def select_worker(
        self,
        pool: WorkerPool,
        request: OpenAIRequest,
        exclude: Optional[Set[str]] = None,  # worker_ids to exclude
        affinity: Optional[TopologyAffinity] = None,
    ) -> WorkerHandle: ...


class TopologyAffinity:
    """Hint for topology-aware routing."""
    prefer_node: Optional[str] = None       # prefer workers on this node
    prefer_nvlink_domain: Optional[int] = None
    kv_source_worker: Optional[str] = None  # for gen: prefer same domain as ctx worker
```

**Topology-aware routing examples:**

- **NVL72 (single NVLink domain):** All workers are equidistant. Routing is
  purely load-based. KV cache transfer is fast between any pair.

- **HGX over IB:** After the context phase completes on a specific node, the
  generation router should prefer generation workers on the **same node** or
  same IB rail to minimize KV cache transfer latency. The `TopologyAffinity`
  carries this hint:

  ```python
  # After context completes on worker ctx_3 (node B, NVLink domain 1):
  affinity = TopologyAffinity(
      kv_source_worker="ctx_3",
      prefer_nvlink_domain=1,
  )
  gen_worker = await gen_router.select_worker(gen_pool, gen_req, affinity=affinity)
  ```

### 5. Disaggregated Orchestration (reused)

The existing `OpenAIDisaggregatedService` logic (context-first / generation-first
scheduling, conditional disagg, request conversion) is reused inside the gateway.
The key change: instead of calling `self._ctx_client.send_request(req, server=url)`,
the service calls `self._ctx_transport.send_request(req)` where the transport is
resolved from the worker handle selected by the router.

```python
class GatewayDisaggService:
    """Disaggregated orchestration, adapted for the gateway."""

    def __init__(self, pool_manager: WorkerPoolManager, ctx_router, gen_router):
        self._pool_manager = pool_manager
        self._ctx_router = ctx_router
        self._gen_router = gen_router

    async def handle_request(self, request, hooks=None):
        ctx_pool = self._pool_manager.get_pool(ServerRole.CONTEXT)
        gen_pool = self._pool_manager.get_pool(ServerRole.GENERATION)

        # 1. Select context worker
        ctx_worker = await self._ctx_router.select_worker(ctx_pool, request)

        # 2. Send context-only request via worker's transport
        ctx_response = await ctx_worker.transport.send_request(
            self._make_ctx_request(request)
        )

        # 3. Select generation worker with topology affinity
        affinity = TopologyAffinity(kv_source_worker=ctx_worker.worker_id)
        gen_worker = await self._gen_router.select_worker(
            gen_pool, request, affinity=affinity
        )

        # 4. Send generation-only request
        return await gen_worker.transport.send_request(
            self._make_gen_request(request, ctx_response)
        )
```

---

## Cluster-Specific Considerations

### GB300/GB200 NVL72

- **72 GPUs, single NVLink domain**, all-to-all NVLink bandwidth
- **Recommended mode: Embedded** (single srun, MPI transport)
- Rank assignment: contiguous blocks per pool
- KV cache transfer: NIXL over NVLink (already optimal, no routing needed)
- All pools on same physical "node" — topology routing not needed, pure load balancing
- NUMA binding: gateway rank 0 should be on CPU NUMA node 0; use `numactl -m 0,1`
  for worker ranks (as current `start_worker.sh` already does)

**Example:**
```yaml
gateway:
  mode: embedded
  port: 8000
topology:
  cluster_type: nvl72
worker_pools:
  - name: ctx
    role: context
    num_instances: 4
    tensor_parallel_size: 8       # 4×8 = 32 GPUs for context
  - name: gen
    role: generation
    num_instances: 10
    tensor_parallel_size: 4       # 10×4 = 40 GPUs for generation
# Total: 32 + 40 = 72 GPUs, rank 0 is gateway (shared with ctx pool leader)
```

```bash
srun --ntasks=72 --ntasks-per-node=72 trtllm-serve gateway -c gateway.yaml
```

### B200/300 HGX (Multi-Node, InfiniBand)

- **8 GPUs per node**, NVLink intra-node, InfiniBand inter-node
- **Recommended mode: External** (workers are independent srun steps)
- Workers may span nodes for PP (e.g., TP=8 within one node, PP=2 across two nodes)
- Topology routing matters: prefer gen workers on same node/rail as ctx worker for
  KV cache transfer over NVLink rather than IB
- Gateway runs on a head node or any node with network access (CPU-only)

**Example:**
```yaml
gateway:
  mode: external
  port: 8000
  internal_port: 9090
routing:
  context_strategy: kv_cache_aware
  generation_strategy: load_balancing
  disaggregated: true
topology:
  cluster_type: hgx_ib
```

```bash
# Head node: gateway
trtllm-serve gateway -c gateway.yaml &

# Node A: context server (TP=8, 1 node)
srun -N1 --ntasks=8 trtllm-serve worker \
    --gateway head:9090 --role context \
    --model meta-llama/Llama-3.1-70B --tp 8

# Node B: another context server
srun -N1 --ntasks=8 trtllm-serve worker \
    --gateway head:9090 --role context \
    --model meta-llama/Llama-3.1-70B --tp 8

# Nodes C,D: generation servers (TP=4, 2 per node)
srun -N1 --ntasks=4 trtllm-serve worker \
    --gateway head:9090 --role generation \
    --model meta-llama/Llama-3.1-70B --tp 4
# (repeat for each gen instance)
```

### Mixed / Hybrid Mode

For large HGX clusters where some nodes are co-located (same rack, low-latency IB),
you can use embedded mode per-rack and connect racks via external mode:

```
 Rack 1 (embedded srun)          Rack 2 (embedded srun)
 ┌─────────────────────┐         ┌─────────────────────┐
 │ Gateway + ctx + gen  │◄──IB──►│ Gateway + ctx + gen  │
 │ (single srun)        │         │ (single srun)        │
 └──────────┬──────────┘         └──────────┬──────────┘
            │                               │
            └───────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │ Top-Level Gateway│   (external mode)
              │ routes across    │   aggregates rack-level gateways
              │ racks            │
              └──────────────────┘
```

---

## Streaming Response Design

Streaming (SSE for `/v1/chat/completions`) must work end-to-end through the
gateway. Each transport must support it:

| Transport | Streaming Mechanism |
|-----------|---------------------|
| `MPITransport` | Worker sends token-level `MPI_Send` messages; gateway converts to SSE chunks. Sentinel message indicates completion. |
| `ZMQIPCTransport` | Worker pushes token-level ZMQ messages; gateway reads and converts. Existing pattern from `executor/proxy.py`. |
| `GRPCTransport` | gRPC server-streaming RPC. Worker yields response chunks; gateway converts to SSE. |
| `HTTPProxyTransport` | Gateway proxies the SSE stream from worker's HTTP response. Current disagg server behavior. |

The gateway's HTTP handler looks the same regardless of transport:

```python
async def handle_chat_completion(self, request):
    worker = await self.router.select_worker(pool, request)
    response_stream = worker.transport.send_request(serialize(request))

    if request.stream:
        async def sse_generator():
            async for chunk in response_stream:
                yield f"data: {chunk.to_json()}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        # Collect full response
        full_response = await collect_response(response_stream)
        return JSONResponse(full_response.model_dump())
```

---

## Worker-Side Changes

### Embedded Mode

Workers don't run their own HTTP server. Instead, the pool leader rank runs a
**request loop** that receives from the gateway and feeds the local executor:

```python
def worker_serve_loop(comm, gateway_rank, executor):
    """Run on pool leader ranks in embedded mode."""
    while True:
        # Receive request from gateway
        req_data = comm.recv(source=gateway_rank, tag=Tags.REQUEST)
        if req_data is None:  # shutdown sentinel
            break

        request = deserialize(req_data)
        request_id = request.request_id

        # Submit to executor (existing LLM.generate_async)
        result = executor.generate_async(request.inputs, request.sampling_params)

        # Stream response tokens back to gateway
        async for output in result:
            comm.send(
                serialize(ResponseChunk(request_id, output)),
                dest=gateway_rank,
                tag=Tags.RESPONSE,
            )

        # Send completion sentinel
        comm.send(
            serialize(ResponseDone(request_id)),
            dest=gateway_rank,
            tag=Tags.RESPONSE,
        )
```

### External Mode

Workers run their existing HTTP server (no changes) **plus** a registration sidecar:

```python
class WorkerRegistrar:
    """Registers this worker with the gateway on startup."""

    def __init__(self, gateway_addr, role, worker_info):
        self.gateway_addr = gateway_addr
        self.role = role
        self.worker_info = worker_info

    async def register(self):
        """Called during server lifespan startup."""
        async with grpc.aio.insecure_channel(self.gateway_addr) as channel:
            stub = GatewayServiceStub(channel)
            await stub.RegisterWorker(WorkerRegistration(
                worker_id=self.worker_info.id,
                role=self.role,
                hostname=self.worker_info.hostname,
                port=self.worker_info.port,
                model=self.worker_info.model,
                tp_size=self.worker_info.tp_size,
                capacity=self.worker_info.capacity,
            ))

    async def heartbeat_loop(self):
        """Periodic heartbeat to gateway."""
        while True:
            await self.send_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)
```

In external/proxy mode, the gateway forwards HTTP requests to the worker's
existing OpenAI endpoint — zero worker-side code changes needed for the initial
implementation. The gRPC direct transport is an optimization for later.

---

## Migration Path

### Phase 1: External Gateway with HTTP Proxy (lowest risk)

- New `trtllm-serve gateway` command
- Gateway accepts `--mode external`
- Workers use `--gateway` flag to auto-register (or gateway discovers via etcd)
- Gateway forwards requests via HTTP (reuses `OpenAIHttpClient` from
  `serve/openai_client.py`)
- Disagg orchestration reuses `OpenAIDisaggregatedService`
- **Benefit:** Single endpoint, auto-discovery, no worker changes
- **Delta from current disagg server:** Auto-registration replaces static URLs

### Phase 2: Embedded Gateway with IPC Transport

- Implement `trtllm-serve gateway --mode embedded`
- Reuse `split_world_comm` for rank assignment
- Implement `ZMQIPCTransport` (extend existing proxy.py patterns)
- Worker leader ranks enter request loop instead of starting HTTP servers
- **Benefit:** Eliminates HTTP overhead for co-located deployments (NVL72)

### Phase 3: Topology-Aware Routing

- Extend `WorkerHandle` with topology metadata (node, NVLink domain, IB rail)
- Implement `TopologyAffinity` in router selection
- Auto-detect topology from MPI communicator / CUDA device properties
- **Benefit:** Smarter gen worker selection after ctx phase, especially on HGX

### Phase 4: MPI Direct Transport

- Implement `MPITransport` for embedded mode
- Lower latency than ZMQ IPC (avoids user-space serialization for same-node)
- Requires careful asyncio ↔ MPI integration (use `mpi4py.futures` or dedicated
  MPI polling thread)
- **Benefit:** Minimal-overhead transport for single-srun deployments

### Phase 5: gRPC Direct Transport for External Mode

- Implement `GRPCTransport` for cross-node workers
- Define internal proto service (lighter than OpenAI HTTP)
- Workers expose internal gRPC port alongside HTTP
- **Benefit:** Lower latency than HTTP proxy for cross-node

---

## Internal gRPC Service Definition (Phases 2+5)

```protobuf
syntax = "proto3";

package trtllm.gateway;

service GatewayService {
    // Worker → Gateway: registration and heartbeat
    rpc RegisterWorker(WorkerRegistration) returns (RegistrationResponse);
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
    rpc DeregisterWorker(DeregisterRequest) returns (DeregisterResponse);
}

service WorkerService {
    // Gateway → Worker: request dispatch
    rpc Generate(GenerateRequest) returns (stream GenerateResponse);
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
    rpc GetStats(StatsRequest) returns (StatsResponse);
}

message WorkerRegistration {
    string worker_id = 1;
    string role = 2;          // "context" | "generation" | "ifb"
    string hostname = 3;
    int32 port = 4;
    int32 internal_grpc_port = 5;
    string model = 6;
    int32 tp_size = 7;
    int32 pp_size = 8;
    WorkerCapacity capacity = 9;
    TopologyInfo topology = 10;
}

message TopologyInfo {
    string node_id = 1;
    int32 nvlink_domain = 2;
    string ib_subnet = 3;
    repeated int32 gpu_ids = 4;
}

message GenerateRequest {
    int64 request_id = 1;
    bytes serialized_request = 2;  // msgpack or protobuf-serialized OpenAI request
}

message GenerateResponse {
    int64 request_id = 1;
    bytes serialized_chunk = 2;
    bool is_final = 3;
}
```

---

## Key Design Decisions

### Q: Why not just put nginx/envoy in front of existing servers?

External load balancers solve the "single endpoint" problem but not:
- Disaggregated orchestration (ctx→gen request chaining with opaque state)
- Topology-aware routing (needs application-level knowledge)
- KV cache transfer affinity
- Embedding in a single MPI program for NVL72

The gateway is application-aware, not just a TCP proxy.

### Q: Why keep HTTP proxy transport at all?

Pragmatism. Workers already have battle-tested HTTP servers. Phase 1 reuses
them, enabling a zero-worker-change migration. Advanced transports (IPC, gRPC,
MPI) are phased in as optimizations.

### Q: How does the gateway handle backpressure?

Same as the current architecture:
- The router's load tracking (`ServerState.num_active_requests/tokens`) prevents
  overloading individual workers
- Workers' internal schedulers handle in-flight batching and queueing
- The gateway can add a global admission controller (max concurrent requests)
  as a configurable option

### Q: Single gateway = single point of failure?

Yes, same as today's disagg orchestrator. Mitigations:
- Gateway is stateless (no persistent data) — restart recovers instantly
- Workers continue serving in-flight requests during gateway restart
- For HA: run multiple gateway replicas behind a standard L4 load balancer
  (each gateway discovers the same set of workers)
- External mode: workers re-register with a new gateway on reconnect

### Q: How does this interact with the existing `trtllm-serve disaggregated`?

The gateway subsumes the disaggregated command:
- `trtllm-serve gateway --mode external` replaces `trtllm-serve disaggregated`
  with auto-discovery
- `trtllm-serve gateway --mode embedded` replaces the
  `trtllm-serve disaggregated_mpi_worker` pattern with a simpler single-srun model
- The existing commands remain for backward compatibility but can be deprecated

---

## File Structure (Proposed)

```
tensorrt_llm/
├── serve/
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── gateway_server.py       # FastAPI app, HTTP endpoints
│   │   ├── gateway_config.py       # GatewayConfig, pool definitions
│   │   ├── worker_pool.py          # WorkerPool, WorkerPoolManager, WorkerHandle
│   │   ├── gateway_service.py      # Disagg orchestration adapted for gateway
│   │   └── topology.py             # Topology detection and affinity
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── base.py                 # WorkerTransport ABC
│   │   ├── http_proxy.py           # HTTPProxyTransport (Phase 1)
│   │   ├── zmq_ipc.py             # ZMQIPCTransport (Phase 2)
│   │   ├── grpc_transport.py      # GRPCTransport (Phase 5)
│   │   └── mpi_transport.py       # MPITransport (Phase 4)
│   ├── router.py                   # Extended with UnifiedRouter (Phase 3)
│   └── ...existing files...
├── commands/
│   └── serve.py                    # + gateway() and worker() commands
└── proto/
    └── gateway.proto               # Internal gRPC definitions
```

---

## Summary

| Aspect | Current (disagg server) | Gateway (proposed) |
|--------|------------------------|--------------------|
| Client-facing endpoints | 1 (orchestrator) | 1 (gateway) |
| Backend endpoints | N ctx + M gen HTTP servers | 0 HTTP (embedded) or N+M auto-discovered |
| Worker configuration | Static URLs in YAML | Auto-registration or MPI rank assignment |
| Gateway↔Worker transport | HTTP | MPI / IPC / gRPC / HTTP (pluggable) |
| Topology awareness | None | Node / NVLink domain / IB rail |
| Single srun support | Yes (disagg_mpi_worker) but each instance starts HTTP server | Yes (embedded), no per-instance HTTP servers |
| Elastic scaling | Via etcd + cluster manager | Same + simpler worker registration |
| Worker-side changes | None | None (Phase 1), request loop (Phase 2+) |
