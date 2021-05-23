use crate::solution::UnsignedInt;
use anyhow;
use anyhow::{anyhow as anyhow_error, ensure, Result};
use num_iter;
use tracing::trace;

pub trait AuctionSolver<I: UnsignedInt>
{
    type AuctionSolution;

    fn j_counts_mut(&mut self) -> Vec<I>

    #[inline]
    fn add_value(&mut self, row: I, column: I, value: f64) -> Result<(), anyhow::Error> {
        let current_row = self.j_counts.len() - 1;
        let row_usize: usize = row.as_();
        ensure!(row_usize == current_row || row_usize == current_row + 1);

        let cumulative_offset = self.i_starts_stops[current_row + 1]
            .checked_add(&I::one())
            .ok_or_else(|| {
                anyhow_error!("i_starts_stops vector is longer then max value of type")
            })?;

        if row_usize > current_row {
            // starting the next row
            // ensure that row has at least one element
            ensure!(self.j_counts[current_row] > I::zero());
            self.i_starts_stops.push(cumulative_offset);
            self.j_counts.push(I::one());
        } else {
            self.i_starts_stops[current_row + 1] = cumulative_offset;
            self.j_counts[current_row] += I::one()
        }

        self.column_indices.push(column);
        self.values.push(value);
        Ok(())
    }

    #[inline]
    fn extend_from_values(
        &mut self,
        row: I,
        columns: &[I],
        values: &[f64],
    ) -> Result<(), anyhow::Error> {
        ensure!(columns.len() == values.len());
        let current_row = self.j_counts.len() - 1;
        let row_usize: usize = row.as_();
        ensure!(row_usize == current_row || row_usize == current_row + 1);

        let length_increment = I::from_usize(columns.len())
            .ok_or_else(|| anyhow_error!(" columns slice is longer then max value of type"))?;
        let cumulative_offset = self.i_starts_stops[current_row + 1]
            .checked_add(&length_increment)
            .ok_or_else(|| {
                anyhow_error!("i_starts_stops vector is longer then max value of type")
            })?;

        if row_usize > current_row {
            // starting the next row
            // ensure that current_row has at least one element
            ensure!(self.j_counts[current_row] > I::zero());
            self.i_starts_stops.push(cumulative_offset);
            self.j_counts.push(length_increment);
        } else {
            self.i_starts_stops[current_row + 1] = cumulative_offset;
            self.j_counts[current_row] += length_increment;
        }
        self.column_indices.extend_from_slice(columns);
        self.values.extend_from_slice(values);
        Ok(())
    }

    #[inline]
    fn num_of_arcs(&self) -> usize {
        self.column_indices.len()
    }

    /// Returns current objective value of assignments.
    /// Checks for the sign of the first element to return positive objective.
    fn get_objective(&self, solution: &Self::AuctionSolution) -> f64 {
        let positive_values = if *self.values.get(0).unwrap_or(&0.0) >= 0. {
            true
        } else {
            false
        };
        let mut obj = 0.;
        for i in num_iter::range(I::zero(), self.num_rows) {
            // due to the way data is stored, need to go do some searching to find the corresponding value
            // to assignment i -> j
            let i_usize: usize = i.as_();
            let j: I = solution.person_to_object[i_usize]; // chosen j
            if j == I::max_value() {
                // skip any unassigned
                continue;
            }

            let num_objects = self.j_counts[i_usize];
            let start: I = self.i_starts_stops[i_usize];
            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let l = self.column_indices[glob_idx];
                if l == j {
                    if positive_values {
                        obj += self.values[glob_idx];
                    } else {
                        obj -= self.values[glob_idx];
                    }
                }
            }
        }
        obj
    }

    fn get_toleration(&self, max_abs_cost: f64) -> f64 {
        1.0 / 2_u64.pow(f64::MANTISSA_DIGITS - (max_abs_cost + 1e-7).log2() as u32) as f64
    }

    /// Checks if current solution is a complete solution that satisfies eps-complementary slackness.
    ///
    /// As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
    /// it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure
    /// Returns True if eps-complementary slackness condition is satisfied
    /// e-CS: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
    fn ecs_satisfied(&self, person_to_object: &[I], eps: f64, toleration: f64) -> bool {
        for i in num_iter::range(I::zero(), self.num_rows) {
            let i_usize: usize = i.as_();
            let num_objects = self.j_counts[i_usize]; // the number of objects this person is able to bid on

            let start = self.i_starts_stops[i_usize]; // in flattened index format, the starting index of this person's objects/values
            let j = person_to_object[i_usize]; // chosen object

            let mut chosen_value = f64::NEG_INFINITY;
            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let l: I = self.column_indices[glob_idx];
                if l == j {
                    chosen_value = self.values[glob_idx];
                }
            }

            //  k are all possible biddable objects.
            // Go through each, asserting that max(a_ik - p_k) - eps <= (a_ij - p_j) + tol for all k.
            // Tolerance is added to deal with floating point precision for eCS, due to eps being stored as float
            let j_usize: usize = j.as_();
            let lhs: f64 = chosen_value - self.prices[j_usize] + toleration; // left hand side of inequality

            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let k: usize = self.column_indices[glob_idx].as_();
                let value: f64 = self.values[glob_idx];
                if lhs < value - self.prices[k] - eps {
                    trace!("ECS CONDITION is not met");
                    return false;
                }
            }
        }
        trace!("ECS CONDITION met");
        true
    }

    fn init_csr_storage(&mut self, num_rows: I, num_cols: I) -> Result<(), anyhow::Error> {
        ensure!(num_rows <= num_cols);
        ensure!(num_rows < I::max_value());
        self.num_rows = num_rows;
        self.num_cols = num_cols;

        self.i_starts_stops.clear();
        self.i_starts_stops.resize(2, I::zero());
        self.j_counts.clear();
        self.j_counts.push(I::zero());

        self.column_indices.clear();
        self.values.clear();
        Ok(())
    }

    fn init_solve(&mut self, solution: &mut Self::AuctionSolution, maximize: bool) {
        let num_cols_usize: usize = self.num_cols.as_();
        let positive_values = if *self.values.get(0).unwrap_or(&0.0) >= 0. {
            true
        } else {
            false
        };
        if maximize ^ positive_values {
            self.values.iter_mut().for_each(|v_ref| *v_ref *= -1.);
        }

        self.prices.clear();
        self.prices.resize(num_cols_usize, 0.);

        solution.person_to_object.clear();
        solution
            .person_to_object
            .resize(self.num_rows.as_(), I::max_value());
        solution.object_to_person.clear();
        solution
            .object_to_person
            .resize(self.num_cols.as_(), I::max_value());
        solution.num_unassigned = self.num_rows;
    }
    fn validate_input(&self) -> Result<(), anyhow::Error> {
        let arcs_count = self.num_of_arcs();
        ensure!(arcs_count > 0);
        ensure!(self.num_rows > I::zero() && self.num_cols > I::zero());
        ensure!(arcs_count < I::max_value().as_());
        ensure!(
            arcs_count == self.column_indices.len()
                && self.column_indices.len() == self.values.len()
        );
        debug_assert!(*self.column_indices.iter().max().unwrap() < self.num_cols);
        Ok(())
    }
}
