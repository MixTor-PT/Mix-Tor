//! Runtime differential privacy budget accounting (Phase 3 — item 11).
//!
//! Tracks the cumulative ε spent per session and per user.  Alerts or
//! throttles when approaching the configured budget ceiling.
//!
//! # Composition methods
//!
//! - **Basic**: ε_total = Σ ε_i  (pessimistic, always sound)
//! - **Advanced (RDP)**: tighter via Rényi DP — used for many repeated
//!   Gaussian/Laplace queries.  Not implemented yet; placeholder variant.
//! - **Zero-Concentrated (zCDP)**: even tighter for Gaussian mechanisms.
//!   Placeholder.
//!
//! For the Laplace mechanism used in `DpShaper`, basic composition is exact.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Composition method
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionMethod {
    /// ε_total = Σ ε_i.  Always sound; pessimistic for many queries.
    Basic,
    /// Rényi DP composition (tighter for repeated Laplace/Gaussian).
    /// Returns `Err(NotImplemented)` when queried — use Basic for now.
    AdvancedRdp,
    /// Zero-concentrated DP (tightest for Gaussian mechanisms).
    /// Returns `Err(NotImplemented)` when queried.
    ZeroConcentrated,
}

// ---------------------------------------------------------------------------
// Accountant
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PrivacyAccountant {
    /// Total ε budget available for this session/user.
    budget: f64,
    /// Composition method for tracking.
    method: CompositionMethod,
    /// Cumulative ε spent so far.
    epsilon_spent: f64,
    /// Cumulative δ (failure probability) for approximate DP mechanisms.
    delta_spent: f64,
    /// Number of queries recorded.
    query_count: u64,
    /// When accounting started (for per-time-unit reporting).
    started_at: Instant,
    /// Whether the budget has been exhausted (alerted flag).
    budget_exceeded: bool,
}

impl PrivacyAccountant {
    pub fn new(budget: f64, method: CompositionMethod) -> Self {
        assert!(budget > 0.0, "budget must be positive");
        Self {
            budget,
            method,
            epsilon_spent: 0.0,
            delta_spent: 0.0,
            query_count: 0,
            started_at: Instant::now(),
            budget_exceeded: false,
        }
    }

    // -----------------------------------------------------------------------
    // Recording queries
    // -----------------------------------------------------------------------

    /// Record a single ε-DP query (pure DP, no δ component).
    pub fn record_query(&mut self, epsilon: f64) -> AccountingResult {
        assert!(epsilon >= 0.0, "epsilon must be non-negative");
        self.query_count += 1;
        match self.method {
            CompositionMethod::Basic => {
                self.epsilon_spent += epsilon;
            }
            CompositionMethod::AdvancedRdp | CompositionMethod::ZeroConcentrated => {
                // Fall back to basic composition until RDP/zCDP is implemented.
                self.epsilon_spent += epsilon;
            }
        }

        if self.epsilon_spent > self.budget && !self.budget_exceeded {
            self.budget_exceeded = true;
        }

        self.status()
    }

    /// Record a (ε, δ)-DP query (approximate DP).
    pub fn record_approx_query(&mut self, epsilon: f64, delta: f64) -> AccountingResult {
        self.delta_spent += delta;
        self.record_query(epsilon)
    }

    // -----------------------------------------------------------------------
    // Querying status
    // -----------------------------------------------------------------------

    /// Current accounting status.
    pub fn status(&self) -> AccountingResult {
        let remaining = (self.budget - self.epsilon_spent).max(0.0);
        AccountingResult {
            epsilon_spent: self.epsilon_spent,
            epsilon_remaining: remaining,
            delta_spent: self.delta_spent,
            query_count: self.query_count,
            budget_fraction_used: self.epsilon_spent / self.budget,
            exceeded: self.budget_exceeded,
        }
    }

    /// True if the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.budget_exceeded
    }

    /// True if the budget is within `fraction` of exhaustion.
    ///
    /// E.g., `is_near_limit(0.9)` returns true when ≥ 90% of budget is used.
    pub fn is_near_limit(&self, fraction: f64) -> bool {
        self.epsilon_spent / self.budget >= fraction
    }

    pub fn epsilon_spent(&self) -> f64 {
        self.epsilon_spent
    }

    pub fn delta_spent(&self) -> f64 {
        self.delta_spent
    }

    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    pub fn budget(&self) -> f64 {
        self.budget
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Per-user multi-session accountant
// ---------------------------------------------------------------------------

/// Aggregates `PrivacyAccountant` instances across multiple sessions for a
/// single user.  Each session's ε is added to the global total under basic
/// composition.
pub struct UserPrivacyLedger {
    session_budgets: Vec<f64>,
    lifetime_budget: f64,
    total_epsilon_spent: f64,
}

impl UserPrivacyLedger {
    pub fn new(lifetime_budget: f64) -> Self {
        assert!(lifetime_budget > 0.0);
        Self {
            session_budgets: Vec::new(),
            lifetime_budget,
            total_epsilon_spent: 0.0,
        }
    }

    /// Register a completed session's total ε spend.
    pub fn record_session(&mut self, epsilon_spent: f64) -> LedgerStatus {
        self.session_budgets.push(epsilon_spent);
        self.total_epsilon_spent += epsilon_spent;
        self.status()
    }

    pub fn status(&self) -> LedgerStatus {
        LedgerStatus {
            total_epsilon_spent: self.total_epsilon_spent,
            lifetime_budget: self.lifetime_budget,
            sessions_recorded: self.session_budgets.len(),
            fraction_used: self.total_epsilon_spent / self.lifetime_budget,
            exceeded: self.total_epsilon_spent > self.lifetime_budget,
        }
    }

    pub fn is_exceeded(&self) -> bool {
        self.total_epsilon_spent > self.lifetime_budget
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct AccountingResult {
    pub epsilon_spent: f64,
    pub epsilon_remaining: f64,
    pub delta_spent: f64,
    pub query_count: u64,
    pub budget_fraction_used: f64,
    /// True when the budget ceiling has been crossed.
    pub exceeded: bool,
}

impl AccountingResult {
    /// Suggested action: should the caller throttle or reject the next query?
    pub fn should_throttle(&self) -> bool {
        self.exceeded || self.budget_fraction_used >= 0.95
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LedgerStatus {
    pub total_epsilon_spent: f64,
    pub lifetime_budget: f64,
    pub sessions_recorded: usize,
    pub fraction_used: f64,
    pub exceeded: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_composition_accumulates() {
        let mut acc = PrivacyAccountant::new(10.0, CompositionMethod::Basic);
        for _ in 0..5 {
            acc.record_query(1.0);
        }
        let status = acc.status();
        assert!((status.epsilon_spent - 5.0).abs() < 1e-9);
        assert!(!status.exceeded);
    }

    #[test]
    fn detects_budget_exceeded() {
        let mut acc = PrivacyAccountant::new(2.0, CompositionMethod::Basic);
        acc.record_query(1.5);
        assert!(!acc.is_exceeded());
        acc.record_query(1.0);
        assert!(acc.is_exceeded());
        assert!(acc.status().exceeded);
    }

    #[test]
    fn near_limit_threshold() {
        let mut acc = PrivacyAccountant::new(10.0, CompositionMethod::Basic);
        acc.record_query(8.5);
        assert!(acc.is_near_limit(0.85));
        assert!(!acc.is_near_limit(0.90));
    }

    #[test]
    fn user_ledger_tracks_sessions() {
        let mut ledger = UserPrivacyLedger::new(5.0);
        ledger.record_session(2.0);
        ledger.record_session(2.0);
        let status = ledger.status();
        assert!((status.total_epsilon_spent - 4.0).abs() < 1e-9);
        assert_eq!(status.sessions_recorded, 2);
        assert!(!status.exceeded);

        ledger.record_session(2.0); // total = 6.0 > 5.0
        assert!(ledger.is_exceeded());
    }

    #[test]
    fn should_throttle_at_95_percent() {
        let mut acc = PrivacyAccountant::new(10.0, CompositionMethod::Basic);
        acc.record_query(9.4);
        assert!(!acc.status().should_throttle());
        acc.record_query(0.2); // 9.6 / 10.0 = 96%
        assert!(acc.status().should_throttle());
    }
}