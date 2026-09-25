# CPU Workload Policy

The canonical CPU contract is `config/cpu_workload_policy_v1.json`, with validation and resolution in `core/cpu_workload_policy.py`. It separates execution, market decisions, collection, research, storage, and operator views before the runtime applies worker caps or priority hints.

## Apple Silicon Contract

macOS schedules work dynamically across performance and efficiency cores. The system therefore does not claim hard core affinity. Critical live execution, paper execution, and market-decision processes start outside Darwin background policy, use a bounded shared performance-core budget, and cannot spill into the service-worker budget by policy. Collection, research, storage maintenance, and livefeed work receive lower-priority classes that the scheduler may place on efficiency cores.

On the current 8-P-core / 2-E-core host, one P core remains reserved for the foreground and operating system, seven form the bounded critical shared budget, and at most two E-core service workers handle eligible support work. Resource governors may shrink these budgets under pressure but may not widen them.

`nice` is one-way for an unprivileged running process. A child may lower its own priority, but it cannot safely raise itself after inheriting a worse value. Any critical process already above its class ceiling reports `managed_restart_required`; the policy does not restart it and does not interrupt the soak. The next supervised natural restart launches it under the locked class.

## Stronger Isolation Elsewhere

Linux deployments can replace scheduler intent with cgroup v2 cpuset partitions or Kubernetes CPU Manager static-policy exclusive CPUs. Native packet-processing deployments can further separate polling and service work with DPDK lcores. Those are documented upgrade paths, not capabilities attributed to this macOS host.

## MLX Coexistence And Measured Optimization

MLX uses Apple Silicon unified memory, so CPU and GPU operations can share arrays without an explicit host/device copy. That is an opportunity, not a blanket instruction to run everything on the GPU. Dense, sufficiently large tensor work is a GPU candidate; tiny feature transforms may remain faster on CPU. Any CPU/GPU split must be benchmarked on the actual tensor shapes and must retain deterministic output parity.

The MLX router already enforces pressure-aware batch caps, bounded concurrency, compile canaries, shared-weight reuse, warm-lane limits, hysteresis, and a single-flight mode while backlog workers own the critical budget. The next optimization gate is deliberately evidence based:

1. Compile only stable, reused outer graphs and warm them before measurement; changing shapes or dtypes can retrace or recompile.
2. Batch compatible symbols or sleeves instead of dispatching one tiny GPU operation per decision.
3. Use explicit evaluation boundaries for latency measurements and avoid accidental synchronization in the hot path.
4. Prefetch training samples with deterministic ordering when labels or point-in-time replay require it.
5. Set cache and memory limits from the pressure controller; clear caches only at safe lifecycle boundaries.
6. Admit an optimization only when baseline-versus-candidate output parity passes and p95 latency or peak memory improves on this host.

Faster research can test more valid hypotheses and faster inference can reduce signal staleness. Neither creates alpha. Profitability remains a separate post-cost, out-of-sample, candidate-bound result and cannot be inferred from MLX utilization or throughput.

## Authority Boundary

This policy may classify work, cap configured worker counts, remove Darwin background state for a critical process, and publish evidence. It cannot pin cores on macOS, restart processes, widen workers, pause paper execution, alter order quantity, authorize promotion, or submit live orders.
