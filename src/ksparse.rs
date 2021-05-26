use crate::solution::{AuctionSolution, UnsignedInt};
use crate::solver::AuctionSolver;
use anyhow;
use anyhow::Result;
use num_integer::Integer;
use num_iter;
use tracing::trace;

/// Solver for weighted perfect matching problem (also known as linear assignment problem) with tighter runtime complexity bound for k-left regular sparse bipartite graphs.
/// It finds ε-optimal assignment of N people -> M objects (N <= M), by having people 'bid' for objects
/// sequentially
///
/// The algorithm is presented in [the article](https://arxiv.org/pdf/2101.07155.pdf).
///
/// We denote n = max(N, M), w<sub>max</sub> and w<sub>min</sub> - maximum and minimum weights in the graph
/// The worst case runtime of the algorithm for sparse k-regular is O(nk(w<sub>max</sub> - w<sub>min</sub>) / ε) with
/// high probability. For complete bipartite graphs the runtime is O(n<sup>2</sup>(w<sub>max</sub> - w<sub>min</sub>) / ε).
///
/// If there is no perfect matching the algorithm finds good matching in finite number of steps.
///
/// ## Example
/// ```rust
/// use sparse_linear_assignment::{AuctionSolver, KhoslaSolver};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///    // We have 2 people and 4 objects
///    // weights between person i and objects j
///    let weights = vec![
///        // person 0 can connect with all objects
///        vec![10, 6, 14, 1],
///        // person 1 can connect with first 3 objects
///        vec![17, 18, 16]
///    ];
///    let expected_cost = 1. + 16.;
///    let expected_person_to_object = vec![3, 2];
///    // u32::MAX value is used to indicate that the corresponding object is not assigned.
///    // If there is no perfect matching unassigned people in `person_to_object` will be marked by
///    // u32::MAX too
///    let expected_object_to_person = vec![u32::MAX, u32::MAX, 1, 0];
///    // Create [KhoslaSolver] and [AuctionSolution] instances with expected capacity of rows,
///    // columns and arcs. We can reuse them in case there is a need to solve multiple assignment
///    // problems.
///    let max_rows_capacity = 10;
///    let max_columns_capacity = 10;
///    let max_arcs_capacity = 100;
///    let (mut solver, mut solution) = KhoslaSolver::new(
///        max_rows_capacity, max_columns_capacity, max_arcs_capacity);
///
///    // init solver and CSR storage before populating weights for the next problem instance
///    let num_rows = weights.len();
///    let num_cols = weights[0].len();
///    solver.init(num_rows as u32, num_cols as u32)?;
///    // populate weights into CSR storage and init the solver
///    // row indices are expected to be nondecreasing
///    (0..weights.len() as u32)
///        .zip(weights.iter())
///        .for_each(|(i, row_ref)| {
///            let j_indices = (0..row_ref.len() as u32).collect::<Vec<_>>();
///            let values = row_ref.iter().map(|v| ((*v) as f64)).collect::<Vec<_>>();
///            solver.extend_from_values(i, j_indices.as_slice(), values.as_slice()).unwrap();
///    });
///    // solve the problem instance. We want to minimize the cost of the assignment.
///    let maximize = false;
///    solver.solve(&mut solution, maximize, None)?;
///    // We found perfect matching and all people are assigned
///    assert_eq!(solution.num_unassigned, 0);
///    assert_eq!(solver.get_objective(&solution), expected_cost);
///    assert_eq!(solution.person_to_object, expected_person_to_object);
///    assert_eq!(solution.object_to_person, expected_object_to_person);
///    Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct KhoslaSolver<I: UnsignedInt + Integer> {
    num_rows: I,
    num_cols: I,
    prices: Vec<f64>,
    i_starts_stops: Vec<I>,
    j_counts: Vec<I>,
    column_indices: Vec<I>,
    // memory view of all values
    values: Vec<f64>,
    ustack: Vec<I>,
    pub nits: u32,
}

impl<I: UnsignedInt + Integer> AuctionSolver<I, KhoslaSolver<I>> for KhoslaSolver<I> {
    fn new(
        row_capacity: usize,
        column_capacity: usize,
        arcs_capacity: usize,
    ) -> (Self, AuctionSolution<I>) {
        (
            Self {
                num_rows: I::zero(),
                num_cols: I::zero(),
                i_starts_stops: Vec::with_capacity(row_capacity + 1),
                j_counts: Vec::with_capacity(row_capacity),
                prices: Vec::with_capacity(column_capacity),
                column_indices: Vec::with_capacity(arcs_capacity),
                values: Vec::with_capacity(arcs_capacity),
                ustack: Vec::with_capacity(row_capacity),
                nits: 0,
            },
            AuctionSolution::<I>::new(row_capacity, column_capacity),
        )
    }
    fn num_rows(&self) -> I {
        self.num_rows
    }
    fn num_cols(&self) -> I {
        self.num_cols
    }
    fn num_rows_mut(&mut self) -> &mut I {
        &mut self.num_rows
    }
    fn num_cols_mut(&mut self) -> &mut I {
        &mut self.num_cols
    }

    fn prices(&self) -> &Vec<f64> {
        &self.prices
    }
    fn i_starts_stops(&self) -> &Vec<I> {
        &self.i_starts_stops
    }
    fn j_counts(&self) -> &Vec<I> {
        &self.j_counts
    }
    fn column_indices(&self) -> &Vec<I> {
        &self.column_indices
    }
    fn values(&self) -> &Vec<f64> {
        &self.values
    }

    fn prices_mut(&mut self) -> &mut Vec<f64> {
        &mut self.prices
    }
    fn i_starts_stops_mut(&mut self) -> &mut Vec<I> {
        &mut self.i_starts_stops
    }
    fn j_counts_mut(&mut self) -> &mut Vec<I> {
        &mut self.j_counts
    }
    fn column_indices_mut(&mut self) -> &mut Vec<I> {
        &mut self.column_indices
    }
    fn values_mut(&mut self) -> &mut Vec<f64> {
        &mut self.values
    }

    fn solve(
        &mut self,
        solution: &mut AuctionSolution<I>,
        maximize: bool,
        eps: Option<f64>,
    ) -> Result<(), anyhow::Error> {
        self.validate_input()?;
        self.init_solve(solution, maximize);

        let num_cols_f: f64 = self.num_cols.as_();

        let eps = if let Some(eps) = eps {
            eps
        } else {
            1.0 / num_cols_f
        };
        solution.eps = eps;

        let (w_min, w_max) =
            self.values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), el| {
                    (
                        if min < *el { min } else { *el },
                        if max > *el { max } else { *el },
                    )
                });

        let price_threshold = (num_cols_f / 2.) * (w_max - w_min + eps);
        trace!("price threshold: {}", price_threshold);

        self.nits = 0;

        while let Some(u_i) = self.ustack.pop() {
            self.nits += 1;
            let u: usize = u_i.as_();
            trace!("u: {}", u);
            trace!("prices {:?}", self.prices);
            let start: usize = self.i_starts_stops[u].as_();
            let num_of_u_objects: usize = self.j_counts[u].as_();
            let mut max_profit = f64::NEG_INFINITY;
            let mut max_edge_value = f64::NEG_INFINITY;
            let mut matched_v_i: I = I::zero();

            let mut second_max_profit = f64::NEG_INFINITY;

            // choice rule
            for idx in 0..num_of_u_objects {
                let glob_idx = start + idx;
                let j: I = self.column_indices[glob_idx];
                let j_usize: usize = j.as_();
                let edge_value = self.values[glob_idx];
                let profit = edge_value - self.prices[j_usize];
                if profit > max_profit {
                    matched_v_i = j;
                    second_max_profit = max_profit;
                    max_profit = profit;
                    max_edge_value = edge_value;
                } else if profit > second_max_profit {
                    second_max_profit = profit;
                }
            }
            let matched_v: usize = matched_v_i.as_();
            trace!("matched_v: {}, max_profit: {}", matched_v, max_profit);

            if self.prices[matched_v] > price_threshold {
                continue;
            }

            // update rule
            if second_max_profit.is_finite() {
                self.prices[matched_v] = max_edge_value - second_max_profit + eps;
            } else {
                self.prices[matched_v] += eps;
            }

            let moved_out_u_i = solution.object_to_person[matched_v];

            if moved_out_u_i != I::max_value() {
                trace!("{} move out ", moved_out_u_i);
                let moved_out_u: usize = moved_out_u_i.as_();
                debug_assert!(moved_out_u != u);
                debug_assert!(matched_v_i == solution.person_to_object[moved_out_u]);
                // move edge (u, v) out of matching
                solution.person_to_object[moved_out_u] = I::max_value();
                solution.num_unassigned += I::one();
                self.ustack.push(moved_out_u_i);
            }
            // move new age (u, matched_v) to the matching
            solution.person_to_object[u] = matched_v_i;
            solution.object_to_person[matched_v] = u_i;
            solution.num_unassigned -= I::one();
        }
        trace!("OBJECTIVE: {:?}", self.get_objective(solution));
        trace!("person_to_object: {:?}", solution.person_to_object);
        trace!("prices: {:?}", self.prices);

        Ok(())
    }
}
impl<I: UnsignedInt + Integer> KhoslaSolver<I> {
    fn init_solve(&mut self, solution: &mut AuctionSolution<I>, maximize: bool) {
        AuctionSolver::init_solve(self, solution, maximize);
        self.ustack.clear();
        self.ustack
            .extend(num_iter::range(I::zero(), self.num_rows).rev());
    }
}
