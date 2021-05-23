use crate::solution;
use crate::solver::AuctionSolver;
use anyhow;
use anyhow::Result;
use num_iter;
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, PrimInt, Unsigned};
use tracing::{info, trace};

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
#[derive(Clone)]
struct ForwardAuctionSolver<I: solution::UnsignedInt> {
    num_rows: I,
    num_cols: I,
    prices: Vec<f64>,
    i_starts_stops: Vec<I>,
    j_counts: Vec<I>,
    column_indices: Vec<I>,
    // memory view of all values
    values: Vec<f64>,

    target_eps: f64,

    max_iterations: u32,

    pub nits: u32,
    pub nreductions: u32,
    pub optimal_soln_found: bool,

    best_bids: Vec<f64>,
    best_bidders: Vec<I>,

    // assignment storage
    unassigned_people: Vec<I>,
    person_to_assignment_idx: Vec<I>,
}

impl<I: solution::UnsignedInt> AuctionSolver<I> for ForwardAuctionSolver<I> {
    type AuctionSolution = solution::AuctionSolution<I>;
}

impl<I: solution::UnsignedInt> ForwardAuctionSolver<I> {
    const REDUCTION_FACTOR: f64 = 0.15;
    const MAX_ITERATIONS: u32 = 100000;

    pub fn new(
        row_capacity: usize,
        column_capacity: usize,
        arcs_capacity: usize,
    ) -> (Self, <Self as AuctionSolver<I>>::AuctionSolution) {
        (
            Self {
                num_rows: I::zero(),
                num_cols: I::zero(),
                i_starts_stops: Vec::with_capacity(row_capacity + 1),
                j_counts: Vec::with_capacity(row_capacity),
                prices: Vec::with_capacity(column_capacity),
                column_indices: Vec::with_capacity(arcs_capacity),
                values: Vec::with_capacity(arcs_capacity),
                target_eps: f64::NAN,

                max_iterations: Self::MAX_ITERATIONS,

                nits: 0,
                nreductions: 0,
                optimal_soln_found: false,

                best_bids: Vec::with_capacity(column_capacity),
                best_bidders: Vec::with_capacity(column_capacity),

                unassigned_people: Vec::with_capacity(row_capacity),
                person_to_assignment_idx: Vec::with_capacity(row_capacity),
            },
            <Self as AuctionSolver<I>>::AuctionSolution::new(row_capacity, column_capacity),
        )
    }
    #[inline]
    pub fn init(
        &mut self,
        num_rows: I,
        num_cols: I,
        max_iterations: Option<u32>,
        target_eps: Option<f64>,
    ) -> Result<(), anyhow::Error> {
        self.init_csr_storage(num_rows, num_cols);

        let float_num_rows: f64 = self.num_rows.as_();

        self.target_eps = if let Some(eps) = target_eps {
            eps
        } else {
            1.0 / float_num_rows
        };

        self.max_iterations = if let Some(iterations) = max_iterations {
            iterations
        } else {
            Self::MAX_ITERATIONS
        };
        Ok(())
    }

    fn init_solve(
        &mut self,
        solution: &mut <Self as AuctionSolver<I>>::AuctionSolution,
        maximize: bool,
    ) {
        self.init_solve(solution, maximize);

        let num_cols_usize: usize = self.num_cols.as_();

        self.nits = 0;
        self.nreductions = 0;
        self.optimal_soln_found = false;

        self.best_bids.clear();
        self.best_bids.resize(num_cols_usize, f64::NEG_INFINITY);
        self.best_bidders.clear();
        self.best_bidders.resize(num_cols_usize, I::max_value());

        self.unassigned_people.clear();
        let num_rows_usize = self.num_rows.as_();
        let mut range = num_iter::range(I::zero(), self.num_rows);
        self.unassigned_people
            .resize_with(num_rows_usize, || range.next().unwrap());
        let mut range = num_iter::range(I::zero(), self.num_rows);
        self.person_to_assignment_idx.clear();
        self.person_to_assignment_idx
            .resize_with(num_rows_usize, || range.next().unwrap());
    }
    #[inline]
    fn solve(
        &mut self,
        solution: &mut <Self as AuctionSolver<I>>::AuctionSolution,
        maximize: bool,
        start_eps: Option<f64>,
    ) -> Result<(), anyhow::Error> {
        self.validate_input()?;
        self.init_solve(solution, maximize);

        // choose eps values
        // Calculate optimum initial eps and target eps
        // C = max |aij| for all i, j in A(i)
        let c = self.values.iter().fold(0_f64, |acc, x| acc.max(x.abs()));
        trace!("c: {}", c);
        let toleration = self.get_toleration(c);
        trace!("toleration: {:e}", toleration);

        let mut start_from_optimal_eps = if let Some(eps) = start_eps {
            eps < self.target_eps
        } else {
            false
        };
        if self.num_rows != self.num_cols {
            // It's possible to use reverse auction for to speed up asymmetric instances
            // https://www.researchgate.net/publication/239574228_Reverse_auction_and_the_solution_of_inequality_constrained_assignment_problems
            // But this implementations uses only forward auction algorithm that doesn't support
            // eps-scaling technique.
            // We start from the target_eps.
            start_from_optimal_eps = true;
            if let Some(_) = start_eps {
                info!("Disabling epsilon scaling for asymmetric assignment problem");
            }
            solution.eps = self.target_eps - f64::EPSILON;
        } else {
            solution.eps = if let Some(eps) = start_eps {
                eps
            } else {
                c / 2.0
            };
        }

        loop {
            self.bid_and_assign(solution);
            self.nits += 1;

            if solution.num_unassigned == I::zero() {
                let is_optimal = start_from_optimal_eps
                    || self.ecs_satisfied(solution.person_to_object.as_slice(), toleration);
                if is_optimal {
                    solution.optimal_soln_found = true;
                    break;
                } else {
                    // full assignment made, but not all people happy, so restart with same prices, but lower eps
                    if solution.eps < self.target_eps {
                        // terminate, shown to be optimal for eps < 1/n
                        break;
                    }

                    solution.eps *= AuctionSolver::<I>::REDUCTION_FACTOR;
                    trace!("REDUCTION: eps {}", solution.eps);

                    // reset all trackers of people and objects
                    solution
                        .person_to_object
                        .iter_mut()
                        .for_each(|i_ref| *i_ref = I::max_value());
                    solution
                        .object_to_person
                        .iter_mut()
                        .for_each(|i_ref| *i_ref = I::max_value());
                    solution.num_unassigned = self.num_rows;
                    num_iter::range(
                        I::zero(),
                        I::from_usize(self.unassigned_people.len()).unwrap(),
                    )
                    .zip(self.unassigned_people.iter_mut())
                    .for_each(|(i, item_ref)| *item_ref = i);

                    num_iter::range(
                        I::zero(),
                        I::from_usize(self.person_to_assignment_idx.len()).unwrap(),
                    )
                    .zip(self.person_to_assignment_idx.iter_mut())
                    .for_each(|(i, item_ref)| *item_ref = i);

                    solution.nreductions += 1
                }
            }
            if solution.nits >= self.max_iterations {
                break;
            }
        }

        Ok(())
    }

    fn bid_and_assign(&mut self, solution: &mut <Self as AuctionSolver<I>>::AuctionSolution) {
        // number of bids to be made
        let num_bidders = solution.num_unassigned.as_();
        let mut bidders = vec![I::max_value(); num_bidders];
        let mut objects_bidded = vec![I::max_value(); num_bidders];
        let mut bids = vec![f64::NEG_INFINITY; num_bidders];

        // BIDDING PHASE
        // each person now makes a bid:
        bidders
            .iter_mut()
            .enumerate()
            .for_each(|(nbidder, bidder_refmut)| {
                let i: I = self.unassigned_people[nbidder];
                let i_usize: usize = i.as_();
                let num_objects_i: I = self.j_counts[i_usize];
                let num_objects = num_objects_i.as_(); // the number of objects this person is able to bid on
                let start_i: I = self.i_starts_stops[i_usize];
                let start: usize = start_i.as_(); // in flattened index format, the starting index of this person's objects/values
                                                  // initially 0 object is considered the best
                let mut jbest = I::zero();
                let mut max_edge_value = f64::NEG_INFINITY;
                // best net reword
                let mut max_profit = f64::NEG_INFINITY;
                // second best net reword
                let mut second_max_profit = f64::NEG_INFINITY;
                // Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
                for idx in 0..num_objects {
                    let glob_idx = start + idx;
                    let j: I = self.column_indices[glob_idx];
                    let j_usize: usize = j.as_();
                    let edge_value = self.values[glob_idx];
                    let profit = edge_value - self.prices[j_usize];
                    if profit > max_profit {
                        // if best so far (or first entry)
                        jbest = j;
                        second_max_profit = max_profit; // store current vbest as second best, wi
                        max_profit = profit;
                        max_edge_value = edge_value;
                    } else if profit > second_max_profit {
                        second_max_profit = profit;
                    }
                }

                let bbest = max_edge_value - second_max_profit + solution.eps; // value of new bid

                // store bid & its value
                *bidder_refmut = i;
                bids[nbidder] = bbest;
                objects_bidded[nbidder] = jbest
            });

        let mut num_successful_bids = 0; // counter of how many succesful bids

        objects_bidded.iter().enumerate().for_each(|(n, jbid_ref)| {
            // for each bid made,
            let i = bidders[n]; // bidder
            let bid_val = bids[n]; // value
            let jbid_i: I = *jbid_ref;
            let jbid: usize = jbid_i.as_(); // object
            if bid_val > self.best_bids[jbid] {
                // if beats current best bid for this object
                if self.best_bidders[jbid] == I::max_value() {
                    // if not overwriting existing bid, increment bid counter
                    num_successful_bids += 1
                }

                // store bid
                self.best_bids[jbid] = bid_val;
                self.best_bidders[jbid] = i;
            }
        });
        trace!("best_bidders {:?}", self.best_bidders);
        trace!("best_bids {:?}", self.best_bids);

        // ASSIGNMENT PHASE
        let mut people_to_unassign_ctr = I::zero(); // counter of how many people have been unassigned
        let mut people_to_assign_ctr = I::zero(); // counter of how many people have been assigned
        let mut bid_ctr = 0;

        for j in num_iter::range(I::zero(), self.num_cols) {
            let j_usize: usize = j.as_();
            let i = self.best_bidders[j_usize];
            if i != I::max_value() {
                self.prices[j_usize] = self.best_bids[j_usize];
                let i_usize: usize = i.as_();
                let assignment_idx: I = self.person_to_assignment_idx[i_usize];
                let assignment_idx_usize: usize = assignment_idx.as_();

                // unassign previous i (if any)
                let prev_i = solution.object_to_person[j_usize];
                if prev_i != I::max_value() {
                    people_to_unassign_ctr += I::one();
                    let prev_i_usize: usize = prev_i.as_();
                    solution.person_to_object[prev_i_usize] = I::max_value();

                    // let old i take new i's place in unassigned people list for faster reading
                    self.person_to_assignment_idx[i_usize] = I::max_value();
                    self.person_to_assignment_idx[prev_i_usize] = assignment_idx;
                    self.unassigned_people[assignment_idx_usize] = prev_i;
                } else {
                    self.unassigned_people[assignment_idx_usize] = I::max_value(); // store empty space in assignment list
                    self.person_to_assignment_idx[i_usize] = I::max_value();
                }

                // make new assignment
                people_to_assign_ctr += I::one();
                solution.person_to_object[i_usize] = j;
                solution.object_to_person[j_usize] = i;

                // bid has been processed, reset best bids store to NONE
                self.best_bidders[j_usize] = I::max_value();
                self.best_bids[j_usize] = f64::NEG_INFINITY;

                // keep track of number of bids. Stop early if reached all bids
                bid_ctr += 1;

                if bid_ctr >= num_successful_bids {
                    break;
                }
            }
        }
        solution.num_unassigned += people_to_unassign_ctr;
        solution.num_unassigned -= people_to_assign_ctr;
        push_all_left(
            &mut self.unassigned_people,
            &mut self.person_to_assignment_idx,
            solution.num_unassigned,
            self.num_cols,
        );

        trace!("person_to_object: {:?}", solution.person_to_object);
        trace!("unassigned_people: {:?}", self.unassigned_people);
        trace!("prices: {:?}", self.prices);
    }
}

fn push_all_left<I>(data: &mut [I], mapper: &mut [I], num_ints: I, size: I)
where
    I: PrimInt + Unsigned + AsPrimitive<usize> + FromPrimitive + NumAssign,
{
    // Given an array of valid positive integers (size <size>) and NONEs (I::MAX), arrange so that all valid positive integers are at the start of the array.
    // Provided with N (number of valid positive integers) for speed increase.
    // eg [4294967295, 1, 2, 3, 4294967295, 4294967295] -> [3, 1, 2, 4294967295, 4294967295, 4294967295] (order not important).
    // Also updates mapper in tandem, a 1d array in which the ith idx gives the position of integer i in the array data.
    // All modifications are inplace.

    if num_ints.is_zero() {
        return;
    }

    let mut left_track = I::zero(); // cursor on left hand side of partition
    let mut right_track = num_ints; // cursor on right hand side of partition

    while left_track < num_ints {
        // keep going until found all N components
        let left_track_usize = left_track.as_();
        if data[left_track_usize] == I::max_value() {
            // if empty space
            // move through right track until hit a valid positive integer (or the end of the array)
            while data[right_track.as_()] == I::max_value() && right_track < size {
                right_track += I::one();
            }

            let right_track_usize = right_track.as_();
            // swap two elements
            let i = data[right_track_usize]; // integer taken through
            data[left_track_usize] = i;
            data[right_track_usize] = I::max_value();
            mapper[i.as_()] = left_track;
        }

        left_track += I::one();
    }
}

#[cfg(test)]
mod tests {
    use super::push_all_left;
    use super::{f64, AuctionSolution, AuctionSolver};
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;
    use test_env_log::test;
    use tracing::{debug, trace};

    #[test]
    fn test_push_all_left() {
        const NONE: u16 = u16::MAX;
        let mut arr = [NONE, 1, 2, 3, NONE, NONE];
        let mut mapper = [NONE, 1, 2, 3];
        push_all_left::<u16>(&mut arr, &mut mapper, 3, 3);
        assert_eq!(arr, [3, 1, 2, NONE, NONE, NONE]);
    }

    #[test]
    fn test_cumulative_idx_diff() {
        let arr = [0, 0, 0, 1, 1, 1, 1];
        let (mut solver, _) = AuctionSolver::new(arr.len(), arr.len(), arr.len());
        solver
            .init(arr.len() as u16, arr.len() as u16, None, None)
            .unwrap();
        arr.iter()
            .for_each(|i| solver.add_value(*i, 0, 0.).unwrap());
        assert_eq!(solver.i_starts_stops, [0, 3, 7]);
        assert_eq!(solver.j_counts, [3, 4]);
    }

    fn solver_with_ksparse_input(
        num_rows: u32,
        num_cols: u32,
        arcs_per_person: usize,
    ) -> (AuctionSolver<u32>, AuctionSolution<u32>) {
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: f64 = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);

        let (mut solver, solution) = AuctionSolver::new(
            num_cols as usize,
            num_rows as usize,
            arcs_per_person * num_rows as usize,
        );

        solver.init(num_rows, num_cols, None, None).unwrap();

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
        (solver, solution)
    }

    #[test]
    fn test_random_solve_small() -> Result<(), Box<dyn std::error::Error>> {
        let cases = [(false,), (true,)];
        for (maximize,) in cases.iter() {
            debug!("maximize {}", *maximize);
            let (mut solver, mut solution) = solver_with_ksparse_input(5, 5, 2);
            let mut approx_solution = solution.clone();
            solver
                .solve_approx(&mut approx_solution, *maximize, None)
                .unwrap();
            let approx_objective = solver.get_objective(&approx_solution);
            solver.solve(&mut solution, *maximize, None).unwrap();
            assert_eq!(approx_objective, solver.get_objective(&solution));
            debug!(
                "approx objective {}, {:?}",
                solver.get_objective(&approx_solution),
                approx_solution
            );
            debug!(
                "auction objective {}, {:?}",
                solver.get_objective(&solution),
                solution
            );
            assert_eq!(solution.num_unassigned, 0);
            assert_eq!(approx_solution.num_unassigned, 0);
        }
        Ok(())
    }

    #[test]
    fn test_random_no_perfect_matching() -> Result<(), Box<dyn std::error::Error>> {
        const NUM_ROWS: u32 = 9;
        const NUM_COLS: u32 = 9;
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: f64 = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);
        const ARCS_PER_PERSON: usize = 3;

        let (mut solver, mut solution) = AuctionSolver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        let mut approx_solution = solution.clone();
        solver.init(NUM_ROWS, NUM_COLS, Some(200), None).unwrap();

        (0..NUM_ROWS)
            .map(|i| {
                let mut j_samples = [0; ARCS_PER_PERSON];
                reservoir_sample(0..NUM_COLS, &mut j_samples, &mut filter_rng);
                j_samples.sort_unstable();
                (i, j_samples)
            })
            .for_each(|(i, j_samples)| {
                let j_values = j_samples
                    .iter()
                    .map(|_| between.sample(&mut val_rng))
                    .collect::<Vec<_>>();
                solver
                    .extend_from_values(i, &j_samples[..], j_values.as_slice())
                    .unwrap();

                debug!("({} -> {:?}: {:?})", i, j_samples, j_values);
            });
        assert!(solver.i_starts_stops.len() == NUM_ROWS as usize + 1);
        solver
            .solve_approx(&mut approx_solution, false, None)
            .unwrap();
        assert!(approx_solution.num_unassigned == 1);
        let approx_objective = solver.get_objective(&approx_solution);
        solver.solve(&mut solution, false, None).unwrap();
        assert!(solution.num_unassigned == 1);
        let auction_objective = solver.get_objective(&solution);
        debug!("approx {:?} {:?}", approx_objective, approx_solution);
        debug!("auction {:?} {:?}", auction_objective, solution);
        assert!(approx_objective <= auction_objective);
        Ok(())
    }
    #[test]
    fn test_random_large() -> Result<(), Box<dyn std::error::Error>> {
        const NUM_ROWS: u32 = 90;
        const NUM_COLS: u32 = 900;
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: f64 = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);
        const ARCS_PER_PERSON: usize = 32;

        let (mut solver, mut solution) = AuctionSolver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        let mut approx_solution = solution.clone();
        solver.init(NUM_ROWS, NUM_COLS, Some(1000), None).unwrap();

        (0..NUM_ROWS)
            .map(|i| {
                let mut j_samples = [0; ARCS_PER_PERSON];
                reservoir_sample(0..NUM_COLS, &mut j_samples, &mut filter_rng);
                j_samples.sort_unstable();
                (i, j_samples)
            })
            .for_each(|(i, j_samples)| {
                let j_values = j_samples
                    .iter()
                    .map(|_| between.sample(&mut val_rng))
                    .collect::<Vec<_>>();
                solver
                    .extend_from_values(i, &j_samples[..], j_values.as_slice())
                    .unwrap();
            });
        assert!(solver.i_starts_stops.len() == NUM_ROWS as usize + 1);
        solver
            .solve_approx(&mut approx_solution, false, None)
            .unwrap();
        let approx_objective = solver.get_objective(&approx_solution);
        solver
            .solve(&mut solution, false, Some(1.0 / NUM_ROWS as f64))
            .unwrap();
        let auction_objective = solver.get_objective(&solution);
        debug!(
            "approx nits {}, standard nits {}",
            approx_solution.nits, solution.nits
        );
        assert_eq!(approx_objective, auction_objective);
        assert_eq!(approx_solution.num_unassigned, 0);
        assert_eq!(solution.num_unassigned, 0);
        Ok(())
    }
    #[test]
    fn test_fixed_cases() -> Result<(), Box<dyn std::error::Error>> {
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

        let (mut solver, mut solution) = AuctionSolver::new(10, 10, 100);
        let mut approx_solution = solution.clone();

        for (maximize, costs, (optimal_cost, person_to_object, object_to_person)) in cases.iter() {
            let num_rows = costs.len();
            let num_cols = costs[0].len();

            solver
                .init(num_rows as u32, num_cols as u32, None, None)
                .unwrap();
            (0..costs.len() as u32)
                .zip(costs.iter())
                .for_each(|(i, row_ref)| {
                    let j_indices = (0..row_ref.len() as u32).collect::<Vec<_>>();
                    let values = row_ref.iter().map(|v| ((*v) as f64)).collect::<Vec<_>>();
                    solver
                        .extend_from_values(i, j_indices.as_slice(), values.as_slice())
                        .unwrap();
                });
            solver
                .solve_approx(&mut approx_solution, *maximize, None)
                .unwrap();
            trace!("approx: {:?}", approx_solution);
            assert_eq!(solver.get_objective(&approx_solution), *optimal_cost);

            solver.solve(&mut solution, *maximize, None).unwrap();
            trace!("exact {:?}", solution);
            assert_eq!(solution.num_unassigned, 0);
            assert_eq!(solver.get_objective(&solution), *optimal_cost);
            assert!(solution.optimal_soln_found);
            assert_eq!(
                solution.person_to_object, *person_to_object,
                "person_to_object"
            );
            assert_eq!(
                solution.object_to_person, *object_to_person,
                "object_to_person"
            );
        }

        Ok(())
    }
}
