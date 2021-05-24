use crate::solution::AuctionSolution;
use crate::solution::UnsignedInt;
use anyhow;
use anyhow::{anyhow as anyhow_error, ensure, Result};
use num_iter;
use tracing::trace;

pub trait AuctionSolver<I: UnsignedInt, T: AuctionSolver<I, T>> {
    fn new(
        row_capacity: usize,
        column_capacity: usize,
        arcs_capacity: usize,
    ) -> (T, AuctionSolution<I>);

    fn solve(
        &mut self,
        solution: &mut AuctionSolution<I>,
        maximize: bool,
        eps: Option<f64>,
    ) -> Result<(), anyhow::Error>;

    // getters/setters/ref and mut ref accessors, used in the default implementation
    fn num_rows(&self) -> I;
    fn set_num_rows(&mut self, num_rows: I);
    fn num_cols(&self) -> I;
    fn set_num_cols(&mut self, num_cols: I);

    fn prices_ref(&self) -> &Vec<f64>;
    fn i_starts_stops_ref(&self) -> &Vec<I>;
    fn j_counts_ref(&self) -> &Vec<I>;
    fn column_indices_ref(&self) -> &Vec<I>;
    fn values_ref(&self) -> &Vec<f64>;

    fn prices_mut_ref(&mut self) -> &mut Vec<f64>;
    fn i_starts_stops_mut_ref(&mut self) -> &mut Vec<I>;
    fn j_counts_mut_ref(&mut self) -> &mut Vec<I>;
    fn column_indices_mut_ref(&mut self) -> &mut Vec<I>;
    fn values_mut_ref(&mut self) -> &mut Vec<f64>;

    #[inline]
    fn add_value(&mut self, row: I, column: I, value: f64) -> Result<(), anyhow::Error> {
        let current_row = self.j_counts_mut_ref().len() - 1;
        let row_usize: usize = row.as_();
        ensure!(row_usize == current_row || row_usize == current_row + 1);

        let cumulative_offset = self.i_starts_stops_mut_ref()[current_row + 1]
            .checked_add(&I::one())
            .ok_or_else(|| {
                anyhow_error!("i_starts_stops vector is longer then max value of type")
            })?;

        if row_usize > current_row {
            // starting the next row
            // ensure that row has at least one element
            ensure!(self.j_counts_mut_ref()[current_row] > I::zero());
            self.i_starts_stops_mut_ref().push(cumulative_offset);
            self.j_counts_mut_ref().push(I::one());
        } else {
            self.i_starts_stops_mut_ref()[current_row + 1] = cumulative_offset;
            self.j_counts_mut_ref()[current_row] += I::one()
        }

        self.column_indices_mut_ref().push(column);
        self.values_mut_ref().push(value);
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
        let current_row = self.j_counts_mut_ref().len() - 1;
        let row_usize: usize = row.as_();
        ensure!(row_usize == current_row || row_usize == current_row + 1);

        let length_increment = I::from_usize(columns.len())
            .ok_or_else(|| anyhow_error!(" columns slice is longer then max value of type"))?;
        let cumulative_offset = self.i_starts_stops_mut_ref()[current_row + 1]
            .checked_add(&length_increment)
            .ok_or_else(|| {
                anyhow_error!("i_starts_stops_mut_ref() vector is longer then max value of type")
            })?;

        if row_usize > current_row {
            // starting the next row
            // ensure that current_row has at least one element
            ensure!(self.j_counts_mut_ref()[current_row] > I::zero());
            self.i_starts_stops_mut_ref().push(cumulative_offset);
            self.j_counts_mut_ref().push(length_increment);
        } else {
            self.i_starts_stops_mut_ref()[current_row + 1] = cumulative_offset;
            self.j_counts_mut_ref()[current_row] += length_increment;
        }
        self.column_indices_mut_ref().extend_from_slice(columns);
        self.values_mut_ref().extend_from_slice(values);
        Ok(())
    }

    #[inline]
    fn num_of_arcs(&self) -> usize {
        self.column_indices_ref().len()
    }

    /// Returns current objective value of assignments.
    /// Checks for the sign of the first element to return positive objective.
    fn get_objective(&self, solution: &AuctionSolution<I>) -> f64 {
        let positive_values = if *self.values_ref().get(0).unwrap_or(&0.0) >= 0. {
            true
        } else {
            false
        };
        let mut obj = 0.;
        for i in num_iter::range(I::zero(), self.num_rows()) {
            // due to the way data is stored, need to go do some searching to find the corresponding value
            // to assignment i -> j
            let i_usize: usize = i.as_();
            let j: I = solution.person_to_object[i_usize]; // chosen j
            if j == I::max_value() {
                // skip any unassigned
                continue;
            }

            let num_objects = self.j_counts_ref()[i_usize];
            let start: I = self.i_starts_stops_ref()[i_usize];
            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let l = self.column_indices_ref()[glob_idx];
                if l == j {
                    if positive_values {
                        obj += self.values_ref()[glob_idx];
                    } else {
                        obj -= self.values_ref()[glob_idx];
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
        for i in num_iter::range(I::zero(), self.num_rows()) {
            let i_usize: usize = i.as_();
            let num_objects = self.j_counts_ref()[i_usize]; // the number of objects this person is able to bid on

            let start = self.i_starts_stops_ref()[i_usize]; // in flattened index format, the starting index of this person's objects/values
            let j = person_to_object[i_usize]; // chosen object

            let mut chosen_value = f64::NEG_INFINITY;
            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let l: I = self.column_indices_ref()[glob_idx];
                if l == j {
                    chosen_value = self.values_ref()[glob_idx];
                }
            }

            //  k are all possible biddable objects.
            // Go through each, asserting that max(a_ik - p_k) - eps <= (a_ij - p_j) + tol for all k.
            // Tolerance is added to deal with floating point precision for eCS, due to eps being stored as float
            let j_usize: usize = j.as_();
            let lhs: f64 = chosen_value - self.prices_ref()[j_usize] + toleration; // left hand side of inequality

            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let k: usize = self.column_indices_ref()[glob_idx].as_();
                let value: f64 = self.values_ref()[glob_idx];
                if lhs < value - self.prices_ref()[k] - eps {
                    trace!("ECS CONDITION is not met");
                    return false;
                }
            }
        }
        trace!("ECS CONDITION met");
        true
    }

    fn init(&mut self, num_rows: I, num_cols: I) -> Result<(), anyhow::Error> {
        ensure!(num_rows <= num_cols);
        ensure!(num_rows < I::max_value());
        self.set_num_rows(num_rows);
        self.set_num_cols(num_cols);

        self.i_starts_stops_mut_ref().clear();
        self.i_starts_stops_mut_ref().resize(2, I::zero());
        self.j_counts_mut_ref().clear();
        self.j_counts_mut_ref().push(I::zero());

        self.column_indices_mut_ref().clear();
        self.values_mut_ref().clear();
        Ok(())
    }

    fn init_solve(&mut self, solution: &mut AuctionSolution<I>, maximize: bool) {
        let num_cols_usize: usize = self.num_cols().as_();
        let positive_values = if *self.values_mut_ref().get(0).unwrap_or(&0.0) >= 0. {
            true
        } else {
            false
        };
        if maximize ^ positive_values {
            self.values_mut_ref()
                .iter_mut()
                .for_each(|v_ref| *v_ref *= -1.);
        }

        self.prices_mut_ref().clear();
        self.prices_mut_ref().resize(num_cols_usize, 0.);

        solution.person_to_object.clear();
        solution
            .person_to_object
            .resize(self.num_rows().as_(), I::max_value());
        solution.object_to_person.clear();
        solution
            .object_to_person
            .resize(self.num_cols().as_(), I::max_value());
        solution.num_unassigned = self.num_rows();
    }

    fn validate_input(&self) -> Result<(), anyhow::Error> {
        let arcs_count = self.num_of_arcs();
        ensure!(arcs_count > 0);
        ensure!(self.num_rows() > I::zero() && self.num_cols() > I::zero());
        ensure!(arcs_count < I::max_value().as_());
        ensure!(
            arcs_count == self.column_indices_ref().len()
                && self.column_indices_ref().len() == self.values_ref().len()
        );
        debug_assert!(*self.column_indices_ref().iter().max().unwrap() < self.num_cols());
        Ok(())
    }
}

#[cfg(test)]
#[generic_tests::define]
mod tests {
    use super::AuctionSolver;
    #[cfg(feature = "khosla")]
    use crate::ksparse::KhoslaSolver;
    #[cfg(feature = "forward")]
    use crate::symmetric::ForwardAuctionSolver;
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;
    use tracing::{debug, subscriber};
    use tracing_subscriber;

    fn populate_with_ksparse_input<Solver: AuctionSolver<u32, Solver>>(
        solver: &mut Solver,
        num_rows: u32,
        num_cols: u32,
        arcs_per_person: usize,
        max_value: f64,
    ) {
        solver.init(num_rows, num_cols).unwrap();
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        let between = Uniform::from(0.0..max_value);

        (0..num_rows)
            .map(|i| {
                let mut j_samples = vec![0; arcs_per_person];
                reservoir_sample(0..num_cols, &mut j_samples, &mut filter_rng);
                j_samples.sort_unstable();
                (i, j_samples)
            })
            .for_each(|(i, j_samples)| {
                let j_values = j_samples
                    .iter()
                    .map(|_| between.sample(&mut val_rng))
                    .collect::<Vec<_>>();
                solver
                    .extend_from_values(i, j_samples.as_slice(), j_values.as_slice())
                    .unwrap();

                debug!("({} -> {:?}: {:?})", i, j_samples, j_values);
            });
    }

    #[test]
    fn test_random_solve_small<Solver: AuctionSolver<u32, Solver>>() {
        let cases = [(false, 19.329346102942907), (true, 26.682897194725648)];
        const NUM_ROWS: u32 = 5;
        const NUM_COLS: u32 = 5;
        const ARCS_PER_PERSON: usize = 2;

        let (mut solver, mut solution) = Solver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        for (maximize, objective) in cases.iter() {
            debug!("maximize {}", *maximize);
            populate_with_ksparse_input(&mut solver, NUM_ROWS, NUM_COLS, ARCS_PER_PERSON, 10.0);
            solver.solve(&mut solution, *maximize, None).unwrap();
            let solution_objective = solver.get_objective(&solution);
            assert_eq!(*objective, solution_objective);
            assert_eq!(solution.num_unassigned, 0);
        }
    }

    #[test]
    fn test_random_no_perfect_matching<Solver: AuctionSolver<u32, Solver>>() {
        const NUM_ROWS: u32 = 9;
        const NUM_COLS: u32 = 9;
        const ARCS_PER_PERSON: usize = 3;
        const MAX_VALUE: f64 = 10.0;

        let (mut solver, mut solution) = Solver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        populate_with_ksparse_input(&mut solver, NUM_ROWS, NUM_COLS, ARCS_PER_PERSON, MAX_VALUE);
        solver.solve(&mut solution, false, None).unwrap();
        assert_eq!(solution.num_unassigned, 1);
        let solution_objective = solver.get_objective(&solution);
        assert!(
            solution_objective == 19.00601422087291 || solution_objective == 27.812843918178544
        );
    }

    #[test]
    fn test_fixed_cases<Solver: AuctionSolver<u32, Solver>>() {
        let _ = subscriber::set_global_default(
            tracing_subscriber::fmt()
                .with_test_writer()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .finish(),
        );
        // taken from https://github.com/gatagat/lap/blob/master/lap/tests/test_lapjv.py
        let cases = [
            (
                false,
                vec![
                    vec![1000, 2, 11, 10, 8, 7, 6, 5],
                    vec![6, 1000, 1, 8, 8, 4, 6, 7],
                    vec![5, 12, 1000, 11, 8, 12, 3, 11],
                    vec![11, 9, 10, 1000, 1, 9, 8, 10],
                    vec![11, 11, 9, 4, 1000, 2, 10, 9],
                    vec![12, 8, 5, 2, 11, 1000, 11, 9],
                    vec![10, 11, 12, 10, 9, 12, 1000, 3],
                    vec![10, 10, 10, 10, 6, 3, 1, 1000],
                ],
                (
                    17.0,
                    vec![1, 2, 0, 4, 5, 3, 7, 6],
                    vec![2, 0, 1, 5, 3, 4, 7, 6],
                ),
            ),
            (
                false,
                vec![vec![10, 10, 13], vec![4, 8, 8], vec![8, 5, 8]],
                (13.0 + 4.0 + 5.0, vec![1, 0, 2], vec![1, 0, 2]),
            ),
            (
                false,
                vec![
                    vec![10, 6, 14, 1],
                    vec![17, 18, 17, 15],
                    vec![14, 17, 15, 8],
                    vec![11, 13, 11, 4],
                ],
                (6. + 17. + 14. + 4., vec![1, 2, 0, 3], vec![2, 0, 1, 3]),
            ),
            // one person
            (
                false,
                vec![vec![10, 6, 14, 1]],
                (1., vec![3], vec![u32::MAX, u32::MAX, u32::MAX, 0]),
            ),
        ];

        let (mut solver, mut solution) = Solver::new(10, 10, 100);

        for (maximize, costs, (optimal_cost, person_to_object, object_to_person)) in cases.iter() {
            let num_rows = costs.len();
            let num_cols = costs[0].len();

            solver.init(num_rows as u32, num_cols as u32).unwrap();
            (0..costs.len() as u32)
                .zip(costs.iter())
                .for_each(|(i, row_ref)| {
                    let j_indices = (0..row_ref.len() as u32).collect::<Vec<_>>();
                    let values = row_ref.iter().map(|v| ((*v) as f64)).collect::<Vec<_>>();
                    solver
                        .extend_from_values(i, j_indices.as_slice(), values.as_slice())
                        .unwrap();
                });
            solver.solve(&mut solution, *maximize, None).unwrap();
            assert_eq!(solution.num_unassigned, 0);
            assert_eq!(solver.get_objective(&solution), *optimal_cost);
            assert_eq!(
                solution.person_to_object, *person_to_object,
                "person_to_object"
            );
            assert_eq!(
                solution.object_to_person, *object_to_person,
                "object_to_person"
            );
        }
    }
    #[test]
    fn test_random_large<Solver: AuctionSolver<u32, Solver>>() {
        const NUM_ROWS: u32 = 90;
        const NUM_COLS: u32 = 900;
        const ARCS_PER_PERSON: usize = 32;
        const MAX_VALUE: f64 = 10.0;

        let (mut solver, mut solution) = Solver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        populate_with_ksparse_input(&mut solver, NUM_ROWS, NUM_COLS, ARCS_PER_PERSON, MAX_VALUE);
        solver.solve(&mut solution, false, None).unwrap();
        let solution_objective = solver.get_objective(&solution);
        assert_eq!(solution_objective, 32.48411883859272);
        assert_eq!(solution.num_unassigned, 0);
    }

    #[cfg(feature = "khosla")]
    #[instantiate_tests(<KhoslaSolver<u32>>)]
    mod khosla {}
    #[cfg(feature = "forward")]
    #[instantiate_tests(<ForwardAuctionSolver<u32>>)]
    mod forwardauction {}
}
