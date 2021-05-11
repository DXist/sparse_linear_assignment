#![feature(array_methods, array_map, is_sorted, total_cmp)]
use log;
use log::trace;
use num_iter;
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, PrimInt, Unsigned};
use std::fmt::{Debug, Display};
pub type Float = f64;

#[inline]
pub fn cumulative_idxs<I>(arr: &[I]) -> Vec<I>
where
    I: PrimInt + Unsigned + NumAssign + FromPrimitive,
{
    // Given an ordered set of integers 0-N, returns an array of size N+1, where each element gives the index of
    //  stop of the number / start of the next
    // [0, 0, 0, 1, 1, 1, 1] -> [0, 3, 7]
    let mut out: Vec<I> = Vec::with_capacity(arr.len() + 1);
    out.push(I::zero());
    if arr.len() == 0 {
        return out;
    }
    let mut value = I::zero();
    let arr_len = I::from_usize(arr.len()).unwrap();
    for (i, arr_i_ref) in num_iter::range(I::zero(), arr_len).zip(arr.iter()) {
        if *arr_i_ref > value {
            out.push(i); // set start of new value to i
            value += I::one();
        }
    }

    out.push(arr_len); // add on last value's stop (one after to match convention of loop)
    out
}

#[inline]
pub fn diff<I>(arr: &[I]) -> Vec<I>
where
    I: PrimInt + Unsigned,
{
    // Returns the 1D difference of a provided memory space of size N
    let mut iter = arr.iter().peekable();
    let mut out = Vec::with_capacity(arr.len() + 1);

    while let Some(item_ref) = iter.next() {
        if let Some(&next_item_ref) = iter.peek() {
            out.push(*next_item_ref - *item_ref);
        }
    }
    out
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

#[derive(Debug)]
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

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
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

    start_eps: Float,
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
            30
        } else {
            10u32.pow(6)
        };

    pub fn new(
        num_rows: I,
        num_cols: I,
        row_indices: &[I],
        column_indices: Vec<I>,
        values: Vec<Float>,
    ) -> AuctionSolver<I> {
        assert!(num_rows <= num_cols);
        assert!(row_indices.len() == column_indices.len() && column_indices.len() == values.len());
        assert!(row_indices.len() < I::max_value().as_());
        debug_assert!(row_indices.is_sorted(), "expecting sorted row indices");
        // Calculate optimum initial eps and target eps
        // C = max |aij| for all i, j in A(i)
        let c = values
            .iter()
            .max_by(|x, y| x.abs().total_cmp(&y.abs()))
            .expect("values should not be empty");

        let prices = vec![0.; num_cols.as_()];
        let i_starts_stops = cumulative_idxs(row_indices);
        let j_counts = diff(&i_starts_stops);

        // choose eps values
        let start_eps = c / 2.0;
        let float_num_rows: Float = num_rows.as_();
        let target_eps = 1.0 / float_num_rows;

        AuctionSolver::<I> {
            num_rows,
            num_cols,
            i_starts_stops,
            j_counts,
            prices,
            column_indices,
            values,
            start_eps,
            target_eps,

            max_iterations: AuctionSolver::<I>::MAX_ITERATIONS,

            best_bids: vec![Float::NEG_INFINITY; num_cols.as_()],
            best_bidders: vec![I::max_value(); num_cols.as_()],

            unassigned_people: num_iter::range(I::zero(), num_rows).collect(),
            person_to_assignment_idx: num_iter::range(I::zero(), num_rows).collect(),
        }
    }

    #[inline]
    pub fn solve(&mut self) -> AuctionSolution<I> {
        let mut solution = AuctionSolution::<I> {
            person_to_object: vec![I::max_value(); self.num_rows.as_()],
            object_to_person: vec![I::max_value(); self.num_cols.as_()],
            eps: self.start_eps,
            nits: 0,
            nreductions: 0,
            optimal_soln_found: false,
            num_assigned: I::zero(),
            num_unassigned: self.num_rows,
        };
        loop {
            self.bid_and_assign(&mut solution);
            trace!("OBJECTIVE: {:?}", self.get_objective(&solution));
            solution.nits += 1;

            let is_optimal = (solution.num_unassigned == I::zero())
                && self.ece_satisfied(solution.person_to_object.as_slice());
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
                self.unassigned_people
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item_ref)| *item_ref = I::from_usize(i).unwrap());
                self.person_to_assignment_idx
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item_ref)| *item_ref = I::from_usize(i).unwrap());

                solution.nreductions += 1
            }
        }

        solution.num_assigned = self.num_rows - solution.num_unassigned;
        solution
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
                let assignment_idx: usize = self.person_to_assignment_idx[i_usize].as_();

                // unassign previous i (if any)
                let prev_i = solution.object_to_person[j_usize];
                if prev_i != I::max_value() {
                    people_to_unassign_ctr += I::one();
                    let prev_i_usize: usize = prev_i.as_();
                    solution.person_to_object[prev_i_usize] = I::max_value();

                    // let old i take new i's place in unassigned people list for faster reading
                    self.person_to_assignment_idx[i_usize] = I::max_value();
                    self.person_to_assignment_idx[prev_i_usize] =
                        I::from_usize(assignment_idx).unwrap();
                    self.unassigned_people[assignment_idx] = prev_i;
                } else {
                    self.unassigned_people[assignment_idx] = I::max_value(); // store empty space in assignment list
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
    fn get_objective(&self, solution: &AuctionSolution<I>) -> Float {
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

    const TOLERATION: Float = if Float::DIGITS == 6 { 1.0e-7 } else { 1.0e-15 };

    /// Checks if current solution is a complete solution that satisfies eps-complementary slackness.
    ///
    /// As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
    /// it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure
    /// Returns True if eps-complementary slackness condition is satisfied
    /// e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
    fn ece_satisfied(&self, person_to_object: &[I]) -> bool {
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
            // tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
            let j_usize: usize = j.as_();
            let lhs: Float = choice_cost - self.prices[j_usize] + Self::TOLERATION; // left hand side of inequality

            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let k: usize = self.column_indices[glob_idx].as_();
                let cost: Float = self.values[glob_idx];
                if lhs < cost - self.prices[k] - self.target_eps {
                    trace!("ECE CONDITION is not met");
                    return false; // The eCE condition is not met.
                }
            }
        }
        trace!("ECE CONDITION met");
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{cumulative_idxs, diff, push_all_left, AuctionSolver, Float};
    use env_logger;
    use log::trace;
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;

    #[test]
    fn test_cumulative_idx() {
        let arr = [0, 0, 0, 1, 1, 1, 1];
        let res = cumulative_idxs::<u16>(&arr);
        assert_eq!(res, [0, 3, 7]);
    }

    #[test]
    fn test_diff() {
        let arr = [0, 3, 7];
        let res = diff::<u16>(&arr);
        assert_eq!(res, [3, 4]);
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
        let mut row_indices = Vec::with_capacity(NUM_ROWS as usize);
        let mut column_indices = Vec::with_capacity(NUM_COLS as usize);
        let mut values = Vec::with_capacity((NUM_ROWS * NUM_COLS) as usize);
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: Float = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);
        const ARCS_PER_PERSON: usize = 2;

        (0..NUM_ROWS)
            .map(|i| {
                let mut j_samples = [0; ARCS_PER_PERSON];
                reservoir_sample(0..NUM_COLS, &mut j_samples, &mut filter_rng);
                j_samples.sort_unstable();
                (i, j_samples)
            })
            .for_each(|(i, j_samples)| {
                row_indices.extend(std::iter::repeat(i).take(j_samples.len()));
                column_indices.extend_from_slice(j_samples.as_slice());
                let j_values = j_samples.map(|_| between.sample(&mut val_rng));
                values.extend_from_slice(j_values.as_slice());

                trace!("({} -> {:?}: {:?})", i, j_samples, j_values);
            });

        let mut solver = AuctionSolver::new(
            NUM_ROWS,
            NUM_COLS,
            row_indices.as_slice(),
            column_indices,
            values,
        );
        let solution = solver.solve();
        assert!(solution.optimal_soln_found);
        assert!(solution.num_unassigned == 0);
        trace!("{:?}", solution,);
        Ok(())
    }
}
