use crate::solution::{AuctionSolution, UnsignedInt};
use crate::solver::AuctionSolver;
use anyhow;
use anyhow::Result;
use num_integer::Integer;
use num_iter;
use tracing::trace;

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
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
                nits: 0,
            },
            AuctionSolution::<I>::new(row_capacity, column_capacity),
        )
    }
    fn num_rows(&self) -> I {
        self.num_rows
    }
    fn set_num_rows(&mut self, num_rows: I) {
        self.num_rows = num_rows
    }
    fn num_cols(&self) -> I {
        self.num_cols
    }
    fn set_num_cols(&mut self, num_cols: I) {
        self.num_cols = num_cols
    }

    fn prices_ref(&self) -> &Vec<f64> {
        &self.prices
    }
    fn i_starts_stops_ref(&self) -> &Vec<I> {
        &self.i_starts_stops
    }
    fn j_counts_ref(&self) -> &Vec<I> {
        &self.j_counts
    }
    fn column_indices_ref(&self) -> &Vec<I> {
        &self.column_indices
    }
    fn values_ref(&self) -> &Vec<f64> {
        &self.values
    }

    fn prices_mut_ref(&mut self) -> &mut Vec<f64> {
        &mut self.prices
    }
    fn i_starts_stops_mut_ref(&mut self) -> &mut Vec<I> {
        &mut self.i_starts_stops
    }
    fn j_counts_mut_ref(&mut self) -> &mut Vec<I> {
        &mut self.j_counts
    }
    fn column_indices_mut_ref(&mut self) -> &mut Vec<I> {
        &mut self.column_indices
    }
    fn values_mut_ref(&mut self) -> &mut Vec<f64> {
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

        let mut ustack = Vec::with_capacity(self.num_rows.as_());
        ustack.extend(num_iter::range(I::zero(), self.num_rows).rev());

        while let Some(u_i) = ustack.pop() {
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
                ustack.push(moved_out_u_i);
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
