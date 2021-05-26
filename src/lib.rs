//! # Sparse linear assignment
//! Solvers for weighted perfect matching problem ([linear assignment](https://en.wikipedia.org/wiki/Assignment_problem)) for bipartite graphs. Both solvers use variants of auction algorithm and implemented in Rust.
//!
//!    * [KhoslaSolver](KhoslaSolver) is best suited for asymmetric k-regular sparse graphs. The algorithm is presented in [this paper](https://arxiv.org/pdf/2101.07155.pdf). It stops in finite number of iterations.
//!    * [ForwardAuctionSolver](ForwardAuctionSolver) works better for symmetric assignment problems. It uses ε-scaling to speedup the auction algorithm. The implementation is based on [sslap](https://github.com/OllieBoyne/sslap). When there is no perfect matching it enters in endless loop and stops after `max_iterations` number of iterations.

#[cfg(feature = "khosla")]
pub use crate::ksparse::KhoslaSolver;
pub use crate::solution::AuctionSolution;
pub use crate::solver::AuctionSolver;
#[cfg(feature = "forward")]
pub use crate::symmetric::ForwardAuctionSolver;

#[cfg(feature = "khosla")]
pub mod ksparse;
pub mod solution;
pub mod solver;
#[cfg(feature = "forward")]
pub mod symmetric;
