// ═══════════════════════════════════════════════════════════════════════
// QOMN — Runtime Engine
//
// Componentes:
//   1. AsyncOracleEngine  — pool de threads para ORACLE_CALL async
//   2. CrystalCache       — mmap lazy-load + cache por crystal_id
//   3. Profiler           — latencia por instrucción, cache misses, throughput
//   4. MemoryPool         — pool de buffers f32/i8 para zero-copy
//
// Flujo de ejecución (v1.4):
//   BytecodeVm::ORACLE_CALL → AsyncOracleEngine::submit(oid, args) → ticket
//   BytecodeVm::ORACLE_WAIT → AsyncOracleEngine::wait(ticket) → result
//   BytecodeVm::LOAD_CRYS   → CrystalCache::load(cid, mode) → &[u8]
//   BytecodeVm::MM_TERN     → backend_cpu::tgemv_ternary(...)
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

// ── 1. Async Oracle Engine ────────────────────────────────────────────

pub type OracleTicket = u64;
pub type OracleResult = f64;

/// A submitted oracle job.
struct OracleJob {
    ticket:    OracleTicket,
    oracle_id: usize,
    args:      Vec<f64>,
    /// Function to evaluate (closure over oracle body)
    eval_fn:   Box<dyn Fn(&[f64]) -> f64 + Send + 'static>,
}

/// Async oracle execution engine.
/// ORACLE_CALL submits a job and returns a ticket immediately.
/// ORACLE_WAIT blocks until the ticket's result is ready.
pub struct AsyncOracleEngine {
    /// Send jobs to worker threads
    job_tx:  Sender<OracleJob>,
    /// Completed results (ticket → result)
    results: Arc<Mutex<HashMap<OracleTicket, f64>>>,
    /// Monotonic ticket counter
    next_ticket: Arc<Mutex<OracleTicket>>,
}

impl AsyncOracleEngine {
    /// Create engine with `n_workers` threads.
    pub fn new(n_workers: usize) -> Self {
        let (job_tx, job_rx) = channel::<OracleJob>();
        let results: Arc<Mutex<HashMap<OracleTicket, f64>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let job_rx = Arc::new(Mutex::new(job_rx));

        for _ in 0..n_workers {
            let rx  = Arc::clone(&job_rx);
            let res = Arc::clone(&results);
            thread::spawn(move || {
                loop {
                    let job = {
                        let lock = rx.lock().unwrap();
                        lock.recv()
                    };
                    match job {
                        Ok(j) => {
                            let value = (j.eval_fn)(&j.args);
                            res.lock().unwrap().insert(j.ticket, value);
                        }
                        Err(_) => break,  // sender dropped → shutdown
                    }
                }
            });
        }

        Self {
            job_tx,
            results,
            next_ticket: Arc::new(Mutex::new(0)),
        }
    }

    /// Submit an oracle computation. Returns immediately with a ticket.
    pub fn submit<F>(&self, oracle_id: usize, args: Vec<f64>, f: F) -> OracleTicket
    where F: Fn(&[f64]) -> f64 + Send + 'static
    {
        let ticket = {
            let mut t = self.next_ticket.lock().unwrap();
            let id = *t;
            *t += 1;
            id
        };
        let _ = self.job_tx.send(OracleJob {
            ticket,
            oracle_id,
            args,
            eval_fn: Box::new(f),
        });
        ticket
    }

    /// Block until ticket result is ready, then return it.
    pub fn wait(&self, ticket: OracleTicket) -> f64 {
        // Spin wait (low-latency for fast oracle evals; use condvar for slow ones)
        loop {
            let r = self.results.lock().unwrap().remove(&ticket);
            if let Some(v) = r { return v; }
            // Yield to avoid busy-spin eating CPU during slow oracle evals
            std::thread::yield_now();
        }
    }

    /// Submit + wait in one call (synchronous path — used by ORACLE_FUSED).
    pub fn call_sync<F>(&self, oracle_id: usize, args: Vec<f64>, f: F) -> f64
    where F: Fn(&[f64]) -> f64 + Send + 'static
    {
        let ticket = self.submit(oracle_id, args, f);
        self.wait(ticket)
    }
}

// ── 2. Crystal Cache (mmap lazy loader) ───────────────────────────────

use crate::bytecode::QomnoadMode;

/// Loaded crystal data (immutable after load).
pub struct CrystalData {
    /// 2-bit packed ternary weights
    pub packed:  Vec<u8>,
    /// Per-row scales
    pub scales:  Vec<f32>,
    pub rows:    usize,
    pub cols:    usize,
    pub name:    String,
}

pub struct CrystalCache {
    loaded: HashMap<usize, Arc<CrystalData>>,
}

impl CrystalCache {
    pub fn new() -> Self {
        Self { loaded: HashMap::new() }
    }

    /// Load crystal by id, applying the appropriate cache mode.
    /// Returns `Arc<CrystalData>` — zero-copy shared between VM frames.
    pub fn load(
        &mut self,
        crystal_id: usize,
        path: &str,
        mode: QomnoadMode,
    ) -> Result<Arc<CrystalData>, String> {
        if let Some(cached) = self.loaded.get(&crystal_id) {
            return Ok(Arc::clone(cached));
        }

        let data = Self::read_crystal_file(path, mode)?;
        let arc  = Arc::new(data);
        self.loaded.insert(crystal_id, Arc::clone(&arc));
        Ok(arc)
    }

    fn read_crystal_file(path: &str, mode: QomnoadMode) -> Result<CrystalData, String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open crystal '{}': {}", path, e))?;

        // Read header (64 bytes)
        let mut header = [0u8; 64];
        f.read_exact(&mut header).map_err(|e| e.to_string())?;

        // Validate magic
        if &header[0..4] != b"CRYS" {
            return Err(format!("Not a .crystal file: {}", path));
        }

        let n_layers = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

        // Read arch string
        let arch_end = header[12..60].iter().position(|&b| b == 0).unwrap_or(48);
        let name = String::from_utf8_lossy(&header[12..12 + arch_end]).to_string();

        // Read layer index (32 bytes per layer)
        let mut layer_idx = vec![0u8; 32 * n_layers];
        f.read_exact(&mut layer_idx).map_err(|e| e.to_string())?;

        let rows = u32::from_le_bytes(layer_idx[8..12].try_into().unwrap()) as usize;
        let cols = u32::from_le_bytes(layer_idx[12..16].try_into().unwrap()) as usize;

        // Read payload (2-bit packed weights)
        let n_bytes = (rows * cols + 3) / 4;
        let mut packed = vec![0u8; n_bytes];

        match mode {
            QomnoadMode::Stream | QomnoadMode::Prefetch => {
                // Stream: read in 64KB chunks to avoid cache pollution
                let chunk = 65536;
                let mut off = 0;
                while off < n_bytes {
                    let end = (off + chunk).min(n_bytes);
                    f.read_exact(&mut packed[off..end]).map_err(|e| e.to_string())?;
                    off = end;
                }
            }
            QomnoadMode::L1Pin => {
                f.read_exact(&mut packed).map_err(|e| e.to_string())?;
            }
        }

        // Generate default scales (1.0 per row — real scales in extended format)
        let scales = vec![1.0f32; rows];

        Ok(CrystalData { packed, scales, rows, cols, name })
    }

    pub fn is_loaded(&self, crystal_id: usize) -> bool {
        self.loaded.contains_key(&crystal_id)
    }

    pub fn evict(&mut self, crystal_id: usize) {
        self.loaded.remove(&crystal_id);
    }
}

// ── 3. Memory Pool (zero-copy buffers) ───────────────────────────────

/// Slab allocator for temporary float/i8 buffers.
/// Reuses buffers across VM frames to avoid repeated allocation.
pub struct MemoryPool {
    f32_pool: Vec<Vec<f32>>,
    i8_pool:  Vec<Vec<i8>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            f32_pool: Vec::with_capacity(16),
            i8_pool:  Vec::with_capacity(16),
        }
    }

    /// Borrow or allocate a f32 buffer of at least `len` elements.
    pub fn get_f32(&mut self, len: usize) -> Vec<f32> {
        if let Some(mut v) = self.f32_pool.pop() {
            v.clear();
            v.resize(len, 0.0);
            v
        } else {
            vec![0.0f32; len]
        }
    }

    /// Return a f32 buffer to the pool.
    pub fn return_f32(&mut self, v: Vec<f32>) {
        if self.f32_pool.len() < 32 { self.f32_pool.push(v); }
    }

    pub fn get_i8(&mut self, len: usize) -> Vec<i8> {
        if let Some(mut v) = self.i8_pool.pop() {
            v.clear();
            v.resize(len, 0);
            v
        } else {
            vec![0i8; len]
        }
    }

    pub fn return_i8(&mut self, v: Vec<i8>) {
        if self.i8_pool.len() < 32 { self.i8_pool.push(v); }
    }
}

// ── 4. Profiler ───────────────────────────────────────────────────────

/// Per-instruction timing accumulator.
#[derive(Debug, Default, Clone)]
pub struct OpStat {
    pub calls:      u64,
    pub total_ns:   u64,
    pub min_ns:     u64,
    pub max_ns:     u64,
}

impl OpStat {
    pub fn record(&mut self, ns: u64) {
        self.calls    += 1;
        self.total_ns += ns;
        if self.min_ns == 0 || ns < self.min_ns { self.min_ns = ns; }
        if ns > self.max_ns { self.max_ns = ns; }
    }
    pub fn avg_ns(&self) -> u64 { if self.calls == 0 { 0 } else { self.total_ns / self.calls } }
}

pub struct Profiler {
    pub stats: HashMap<String, OpStat>,
}

impl Profiler {
    pub fn new() -> Self {
        Self { stats: HashMap::new() }
    }

    pub fn record(&mut self, op: &str, ns: u64) {
        self.stats.entry(op.to_string()).or_default().record(ns);
    }

    pub fn report(&self) -> String {
        let mut lines: Vec<_> = self.stats.iter()
            .map(|(op, s)| {
                format!("  {:<20}  calls={:>6}  avg={:>7}ns  min={:>7}ns  max={:>7}ns",
                    op, s.calls, s.avg_ns(), s.min_ns, s.max_ns)
            })
            .collect();
        lines.sort();

        let mut out = String::from("═══ QOMN Profiler ═══\n");
        for l in &lines { out.push_str(l); out.push('\n'); }

        // Summary: total throughput
        let total_ns: u64 = self.stats.values().map(|s| s.total_ns).sum();
        let total_calls: u64 = self.stats.values().map(|s| s.calls).sum();
        out.push_str(&format!("\n  Total: {} calls in {}ms\n",
            total_calls, total_ns / 1_000_000));
        out
    }
}

// ── 5. Runtime context (ties everything together) ─────────────────────

pub struct CrysRuntime {
    pub oracle_engine: AsyncOracleEngine,
    pub crystal_cache: CrystalCache,
    pub mem_pool:      MemoryPool,
    pub profiler:      Profiler,
}

impl CrysRuntime {
    /// Create runtime with `n_oracle_workers` async oracle threads.
    /// Tune for EPYC: each NUMA node has 6 cores → 6 workers per node.
    pub fn new(n_oracle_workers: usize) -> Self {
        Self {
            oracle_engine: AsyncOracleEngine::new(n_oracle_workers),
            crystal_cache: CrystalCache::new(),
            mem_pool:      MemoryPool::new(),
            profiler:      Profiler::new(),
        }
    }

    pub fn default_epyc() -> Self {
        // EPYC 3rd gen: 12 cores → 6 oracle workers (leave 6 for MM_TERN)
        Self::new(6)
    }
}
