# Sparse linear assignment

Solvers for weighted perfect matching problem ([linear assignment](https://en.wikipedia.org/wiki/Assignment_problem)) for bipartite graphs. Both solvers use variants of auction algorithm and implemented in Rust.

   * **KhoslaSolver** is best suited for asymmetric k-regular sparse graphs. The algorithm is presented in [this paper](https://arxiv.org/pdf/2101.07155.pdf). It stops in finite number of iterations.
   * **ForwardAuctionSolver** works better for symmetric assignment problems. It uses ε-scaling to speedup the auction algorithm. The implementation is based on [sslap](https://github.com/OllieBoyne/sslap). When there is no perfect matching it enters in endless loop and stops after `max_iterations` number of iterations.

## Usage
```rust
use sparse_linear_assignment::{AuctionSolver, KhoslaSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
   // We have 2 people and 4 objects
   // weights between person i and objects j
   let weights = vec![
       // person 0 can connect with all objects
       vec![10, 6, 14, 1],
       // person 1 can connect with first 3 objects
       vec![17, 18, 16]
   ];
   let expected_cost = 1. + 16.;
   let expected_person_to_object = vec![3, 2];
   let expected_object_to_person = vec![u32::MAX, u32::MAX, 1, 0];
   // Create `KhoslaSolver` and `AuctionSolution` instances with expected capacity of rows,
   // columns and arcs. We can reuse them in case there is a need to solve multiple assignment
   // problems.
   let max_rows_capacity = 10;
   let max_columns_capacity = 10;
   let max_arcs_capacity = 100;
   let (mut solver, mut solution) = KhoslaSolver::new(
       max_rows_capacity, max_columns_capacity, max_arcs_capacity);

   // init solver and CSR storage before populating weights for the next problem instance
   let num_rows = weights.len();
   let num_cols = weights[0].len();
   solver.init(num_rows as u32, num_cols as u32)?;
   // populate weights into CSR storage and init the solver
   // row indices are expected to be nondecreasing
   (0..weights.len() as u32)
       .zip(weights.iter())
       .for_each(|(i, row_ref)| {
           let j_indices = (0..row_ref.len() as u32).collect::<Vec<_>>();
           let values = row_ref.iter().map(|v| ((*v) as f64)).collect::<Vec<_>>();
           solver.extend_from_values(i, j_indices.as_slice(), values.as_slice()).unwrap();
   });
   // solve the problem instance. We want to minimize the cost of the assignment.
   let maximize = false;
   solver.solve(&mut solution, maximize, None)?;
   // We found perfect matching and all people are assigned
   assert_eq!(solution.num_unassigned, 0);
   assert_eq!(solver.get_objective(&solution), expected_cost);
   assert_eq!(solution.person_to_object, expected_person_to_object);
   assert_eq!(solution.object_to_person, expected_object_to_person);
   Ok(())
}
```
See [tests](./src/solver.rs#L261) for more examples.
