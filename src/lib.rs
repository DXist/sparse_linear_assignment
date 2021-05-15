#![feature(array_methods, array_map, is_sorted, total_cmp)]
use anyhow;
use anyhow::{anyhow as anyhow_error, ensure, Result};
use log;
use log::trace;
use num_iter;
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, PrimInt, Unsigned};
use std::fmt::{Debug, Display};
pub type Float = f64;

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

/// Solution of the linear assignment problem
#[derive(Debug, Clone)]
pub struct AuctionSolution<I>
where
    I: PrimInt
        + Unsigned
        + Display
        + Debug
        + AsPrimitive<usize>
        + AsPrimitive<Float>
        + FromPrimitive
        + NumAssign,
{
    // index i gives the object, j, owned by person i
    pub person_to_object: Vec<I>,
    // index j gives the person, i, who owns object j
    pub object_to_person: Vec<I>,

    pub eps: Float,
    pub nits: u32,
    pub nreductions: u32,
    pub optimal_soln_found: bool,
    pub num_assigned: I,
    pub num_unassigned: I,
}

impl<I> AuctionSolution<I>
where
    I: PrimInt
        + Unsigned
        + Display
        + Debug
        + AsPrimitive<usize>
        + AsPrimitive<Float>
        + FromPrimitive
        + NumAssign,
{
    pub fn new(row_capacity: usize, column_capacity: usize) -> AuctionSolution<I> {
        AuctionSolution::<I> {
            person_to_object: Vec::with_capacity(row_capacity),
            object_to_person: Vec::with_capacity(column_capacity),
            eps: Float::NAN,
            nits: 0,
            nreductions: 0,
            optimal_soln_found: false,
            num_assigned: I::zero(),
            num_unassigned: I::max_value(),
        }
    }
}

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
#[derive(Clone)]
pub struct AuctionSolver<I>
where
    I: PrimInt
        + Unsigned
        + Display
        + Debug
        + AsPrimitive<usize>
        + AsPrimitive<Float>
        + FromPrimitive
        + NumAssign,
{
    num_rows: I,
    num_cols: I,
    prices: Vec<Float>,
    i_starts_stops: Vec<I>,
    j_counts: Vec<I>,
    column_indices: Vec<I>,
    // memory view of all values
    values: Vec<Float>,

    target_eps: Float,

    max_iterations: u32,

    best_bids: Vec<Float>,
    best_bidders: Vec<I>,

    // assignment storage
    unassigned_people: Vec<I>,
    person_to_assignment_idx: Vec<I>,
}

impl<I> AuctionSolver<I>
where
    I: PrimInt
        + Unsigned
        + Display
        + Debug
        + AsPrimitive<usize>
        + AsPrimitive<Float>
        + FromPrimitive
        + NumAssign,
{
    const REDUCTION_FACTOR: Float = 0.15;
    const MAX_ITERATIONS: u32 =
        if log::STATIC_MAX_LEVEL as usize == log::LevelFilter::Trace as usize {
            100
        } else {
            10u32.pow(6)
        };

    pub fn new(
        row_capacity: usize,
        column_capacity: usize,
        arcs_capacity: usize,
    ) -> (AuctionSolver<I>, AuctionSolution<I>) {
        (
            AuctionSolver::<I> {
                num_rows: I::zero(),
                num_cols: I::zero(),
                i_starts_stops: Vec::with_capacity(row_capacity + 1),
                j_counts: Vec::with_capacity(row_capacity),
                prices: Vec::with_capacity(column_capacity),
                column_indices: Vec::with_capacity(arcs_capacity),
                values: Vec::with_capacity(arcs_capacity),
                target_eps: Float::NAN,

                max_iterations: AuctionSolver::<I>::MAX_ITERATIONS,

                best_bids: Vec::with_capacity(column_capacity),
                best_bidders: Vec::with_capacity(column_capacity),

                unassigned_people: Vec::with_capacity(row_capacity),
                person_to_assignment_idx: Vec::with_capacity(row_capacity),
            },
            AuctionSolution::<I>::new(row_capacity, column_capacity),
        )
    }
    #[inline]
    pub fn init(
        &mut self,
        num_rows: I,
        num_cols: I,
        max_iterations: Option<u32>,
    ) -> Result<(), anyhow::Error> {
        ensure!(num_rows <= num_cols);
        ensure!(num_rows < I::max_value());
        self.num_rows = num_rows;
        self.num_cols = num_cols;

        self.i_starts_stops.clear();
        self.i_starts_stops.resize(2, I::zero());
        self.j_counts.clear();
        self.j_counts.push(I::zero());

        let num_cols_usize: usize = num_cols.as_();
        self.prices.clear();
        self.prices.resize(num_cols_usize, 0.);
        self.column_indices.clear();
        self.values.clear();
        self.target_eps = Float::NAN;

        self.max_iterations = if let Some(iterations) = max_iterations {
            iterations
        } else {
            Self::MAX_ITERATIONS
        };

        self.best_bids.clear();
        self.best_bids.resize(num_cols_usize, Float::NEG_INFINITY);
        self.best_bidders.clear();
        self.best_bidders.resize(num_cols_usize, I::max_value());

        self.unassigned_people.clear();
        let num_rows_usize = num_rows.as_();
        let mut range = num_iter::range(I::zero(), num_rows);
        self.unassigned_people
            .resize_with(num_rows_usize, || range.next().unwrap());
        let mut range = num_iter::range(I::zero(), num_rows);
        self.person_to_assignment_idx.clear();
        self.person_to_assignment_idx
            .resize_with(num_rows_usize, || range.next().unwrap());
        Ok(())
    }

    #[inline]
    pub fn add_value(&mut self, row: I, column: I, value: Float) -> Result<(), anyhow::Error> {
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
    pub fn extend_from_values(
        &mut self,
        row: I,
        columns: &[I],
        values: &[Float],
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
    pub fn num_of_arcs(&self) -> usize {
        self.column_indices.len()
    }

    #[inline]
    pub fn solve(&mut self, solution: &mut AuctionSolution<I>) -> Result<(), anyhow::Error> {
        let arcs_count = self.num_of_arcs();
        ensure!(arcs_count > 0);
        ensure!(self.num_rows > I::zero() && self.num_cols > I::zero());
        ensure!(arcs_count < I::max_value().as_());
        ensure!(
            arcs_count == self.column_indices.len()
                && self.column_indices.len() == self.values.len()
        );
        debug_assert!(*self.column_indices.iter().max().unwrap() < self.num_cols);

        let float_num_rows: Float = self.num_rows.as_();
        self.target_eps = 1.0 / float_num_rows;

        solution.person_to_object.clear();
        solution
            .person_to_object
            .resize(self.num_rows.as_(), I::max_value());
        solution.object_to_person.clear();
        solution
            .object_to_person
            .resize(self.num_cols.as_(), I::max_value());
        // choose eps values
        // Calculate optimum initial eps and target eps
        // C = max |aij| for all i, j in A(i)
        let c = self
            .values
            .iter()
            .max_by(|x, y| x.abs().total_cmp(&y.abs()))
            .expect("values should not be empty")
            .abs();
        trace!("c: {}", c);
        let toleration: Float =
            1.0 / 2_u64.pow(Float::MANTISSA_DIGITS - (c + 1e-7).log2() as u32) as Float;
        trace!("toleration: {:e}", toleration);
        let start_eps = c / 2.0;
        solution.eps = start_eps;
        solution.nits = 0;
        solution.nreductions = 0;
        solution.optimal_soln_found = false;
        solution.num_assigned = I::zero();
        solution.num_unassigned = self.num_rows;

        loop {
            self.bid_and_assign(solution);
            trace!("OBJECTIVE: {:?}", self.get_objective(solution));
            solution.nits += 1;

            let is_optimal = (solution.num_unassigned == I::zero())
                && self.ecs_satisfied(solution.person_to_object.as_slice(), toleration);
            if is_optimal {
                solution.optimal_soln_found = true;
                break;
            }
            if solution.nits >= self.max_iterations {
                break;
            }
            // full assignment made, but not all people happy, so restart with same prices, but lower eps
            else if solution.num_unassigned == I::zero() {
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

        solution.num_assigned = self.num_rows - solution.num_unassigned;
        Ok(())
    }

    fn bid_and_assign(&mut self, solution: &mut AuctionSolution<I>) {
        // number of bids to be made
        let num_bidders = solution.num_unassigned.as_();
        let mut bidders = vec![I::max_value(); num_bidders];
        let mut objects_bidded = vec![I::max_value(); num_bidders];
        let mut bids = vec![Float::NEG_INFINITY; num_bidders];

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
                let mut jbest: I = self.column_indices[start];
                let mut costbest = self.values[start];
                // best net reword
                let jbest_usize: usize = jbest.as_();
                let mut vbest = costbest - self.prices[jbest_usize];
                // second best net reword
                let mut wi = Float::NEG_INFINITY; //0.;
                                                  // Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
                for idx in 1..num_objects {
                    let glob_idx = start + idx;
                    let j: I = self.column_indices[glob_idx];
                    let j_usize: usize = j.as_();
                    let cost = self.values[glob_idx];
                    let vi = cost - self.prices[j_usize];
                    if vi > vbest {
                        // if best so far (or first entry)
                        jbest = j;
                        wi = vbest; // store current vbest as second best, wi
                        vbest = vi;
                        costbest = cost;
                    } else if vi > wi {
                        wi = vi;
                    }
                }

                let bbest = costbest - wi + solution.eps; // value of new bid

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
                self.best_bids[j_usize] = Float::NEG_INFINITY;

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
    /// Returns current objective value of assignments
    pub fn get_objective(&self, solution: &AuctionSolution<I>) -> Float {
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
                    obj += self.values[glob_idx];
                }
            }
        }

        return obj;
    }

    /// Checks if current solution is a complete solution that satisfies eps-complementary slackness.
    ///
    /// As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
    /// it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure
    /// Returns True if eps-complementary slackness condition is satisfied
    /// e-CS: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
    fn ecs_satisfied(&self, person_to_object: &[I], toleration: Float) -> bool {
        for i in num_iter::range(I::zero(), self.num_rows) {
            let i_usize: usize = i.as_();
            let num_objects = self.j_counts[i_usize]; // the number of objects this person is able to bid on

            let start = self.i_starts_stops[i_usize]; // in flattened index format, the starting index of this person's objects/values
            let j = person_to_object[i_usize]; // chosen object

            let mut choice_cost = Float::NEG_INFINITY;
            // first, get cost of choice j
            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let l: I = self.column_indices[glob_idx];
                if l == j {
                    choice_cost = self.values[glob_idx];
                }
            }

            //  k are all possible biddable objects.
            // Go through each, asserting that (a_ij - p_j) + tol >= max(a_ik - p_k) - eps for all k
            // tolerance to deal with floating point precision for eCS, due to eps being stored as float
            let j_usize: usize = j.as_();
            let lhs: Float = choice_cost - self.prices[j_usize] + toleration; // left hand side of inequality

            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let k: usize = self.column_indices[glob_idx].as_();
                let cost: Float = self.values[glob_idx];
                if lhs < cost - self.prices[k] - self.target_eps {
                    trace!("ECS CONDITION is not met");
                    return false;
                }
            }
        }
        trace!("ECS CONDITION met");
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{push_all_left, AuctionSolver, Float};
    use env_logger;
    use log::trace;
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;

    #[test]
    fn test_cumulative_idx_diff() {
        let arr = [0, 0, 0, 1, 1, 1, 1];
        let (mut solver, _) = AuctionSolver::new(arr.len(), arr.len(), arr.len());
        solver
            .init(arr.len() as u16, arr.len() as u16, None)
            .unwrap();
        arr.iter()
            .for_each(|i| solver.add_value(*i, 0, 0.).unwrap());
        assert_eq!(solver.i_starts_stops, [0, 3, 7]);
        assert_eq!(solver.j_counts, [3, 4]);
    }

    #[test]
    fn test_push_all_left() {
        const NONE: u16 = u16::MAX;
        let mut arr = [NONE, 1, 2, 3, NONE, NONE];
        let mut mapper = [NONE, 1, 2, 3];
        push_all_left::<u16>(&mut arr, &mut mapper, 3, 3);
        assert_eq!(arr, [3, 1, 2, NONE, NONE, NONE]);
    }
    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
    /// Ensure every row and column has at least one element labelled False, for valid connections

    #[test]
    fn test_sparse_solve() -> Result<(), Box<dyn std::error::Error>> {
        init();
        const NUM_ROWS: u16 = 5;
        const NUM_COLS: u16 = 5;
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: Float = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);
        const ARCS_PER_PERSON: usize = 2;

        let (mut solver, mut solution) = AuctionSolver::new(
            NUM_ROWS.into(),
            NUM_COLS.into(),
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        solver.init(NUM_ROWS, NUM_COLS, None).unwrap();

        (0..NUM_ROWS)
            .map(|i| {
                let mut j_samples = [0; ARCS_PER_PERSON];
                reservoir_sample(0..NUM_COLS, &mut j_samples, &mut filter_rng);
                j_samples.sort_unstable();
                (i, j_samples)
            })
            .for_each(|(i, j_samples)| {
                let j_values = j_samples.map(|_| between.sample(&mut val_rng));
                solver
                    .extend_from_values(i, j_samples.as_slice(), j_values.as_slice())
                    .unwrap();

                trace!("({} -> {:?}: {:?})", i, j_samples, j_values);
            });
        solver.solve(&mut solution).unwrap();
        assert!(solution.optimal_soln_found);
        assert!(solution.num_unassigned == 0);
        trace!("{:?}", solution);
        Ok(())
    }
    #[test]
    fn test_fixed_cases() -> Result<(), Box<dyn std::error::Error>> {
        init();
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
        ];

        let (mut solver, mut solution) = AuctionSolver::new(10, 10, 100);

        for (maximize, costs, (optimal_cost, person_to_object, object_to_person)) in cases.iter() {
            let num_rows = costs.len();
            let num_cols = costs[0].len();
            let cost_multiplier = if *maximize { 1. } else { -1. };

            solver.init(num_rows as u32, num_cols as u32, None).unwrap();
            (0..costs.len() as u32)
                .zip(costs.iter())
                .for_each(|(i, row_ref)| {
                    let j_indices = (0..row_ref.len() as u32).collect::<Vec<_>>();
                    let values = row_ref
                        .iter()
                        .map(|v| ((*v) as Float) * cost_multiplier)
                        .collect::<Vec<_>>();
                    solver
                        .extend_from_values(i, j_indices.as_slice(), values.as_slice())
                        .unwrap();
                });
            solver.solve(&mut solution).unwrap();
            trace!("{:?}", solution);
            assert!(solution.optimal_soln_found);
            assert!(solution.num_unassigned == 0);
            assert_eq!(
                solver.get_objective(&solution) * cost_multiplier,
                *optimal_cost
            );
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
