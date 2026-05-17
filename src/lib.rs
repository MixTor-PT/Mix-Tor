//! Core components for the MixTor pluggable transport.
//!
//! The first security boundary is ephemeral process-local seed material. The
//! seed module intentionally does not provide serialization, persistence, or
//! network-facing representations.

#![forbid(unsafe_code)]

pub mod clumping;
pub mod composition;
pub mod lab;
pub mod protocol;
pub mod seeds;
pub mod socks;
pub mod timing;
pub mod transport;
