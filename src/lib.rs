//! Core components for the MixTor pluggable transport.
//!
//! The first security boundary is ephemeral process-local seed material. The
//! seed module intentionally does not provide serialization, persistence, or
//! network-facing representations.

#![forbid(unsafe_code)]

pub mod clumping;
pub mod composition;
pub mod correlation_attack;
pub mod lab;
pub mod protocol;
pub mod seeds;
pub mod socks;
pub mod timing;
pub mod transport;
pub mod dp_shaper;
pub mod optimal_padding;
pub mod privacy_accountant;
pub mod self_test;
pub mod session_bounder;
pub mod spectral;
pub mod timing_correlator;
