#![feature(array_methods, array_map, is_sorted, total_cmp)]
use log::trace;
use log::LevelFilter;
use log::STATIC_MAX_LEVEL;

const NONE: u32 = u32::MAX;
const NONE_F32: f32 = f32::NEG_INFINITY;

#[inline]
fn cumulative_idxs(arr: &[u32]) -> Vec<u32> {
    // Given an ordered set of integers 0-N, returns an array of size N+1, where each element gives the index of
    //  stop of the number / start of the next
    // [0, 0, 0, 1, 1, 1, 1] -> [0, 3, 7]
    let mut out = Vec::with_capacity(arr.len() + 1);
    out.push(0);
    if arr.len() == 0 {
        return out;
    }
    let mut value = 0;
    for (i, arr_i_ref) in arr.iter().enumerate() {
        if *arr_i_ref > value {
            out.push(i as u32); // set start of new value to i
            value += 1;
        }
    }

    out.push(arr.len() as u32); // add on last value's stop (one after to match convention of loop)
    out
}

fn diff(arr: &[u32]) -> Vec<u32> {
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

fn push_all_left(data: &mut [u32], mapper: &mut [u32], num_ints: usize, size: usize) {
    // Given an array of valid positive integers (size <size>) and NONEs (u32::MAX), arrange so that all valid positive integers are at the start of the array.
    // Provided with N (number of valid positive integers) for speed increase.
    // eg [4294967295, 1, 2, 3, 4294967295, 4294967295] -> [3, 1, 2, 4294967295, 4294967295, 4294967295] (order not important).
    // Also updates mapper in tandem, a 1d array in which the ith idx gives the position of integer i in the array data.
    // All modifications are inplace.

    if num_ints == 0 {
        return;
    }

    let mut left_track = 0; // cursor on left hand side of partition
    let mut right_track = num_ints; // cursor on right hand side of partition
    while left_track < num_ints {
        // keep going until found all N components
        if data[left_track] == NONE {
            // if empty space
            // move through right track until hit a valid positive integer (or the end of the array)
            while data[right_track] == NONE && right_track < size {
                right_track += 1;
            }

            // swap two elements
            let i = data[right_track]; // integer taken through
            data[left_track] = i;
            data[right_track] = NONE;
            mapper[i as usize] = left_track as u32;
        }

        left_track += 1;
    }
}

#[derive(Debug)]
struct AuctionSolution {
    // index i gives the object, j, owned by person i
    person_to_object: Vec<u32>,
    // index j gives the person, i, who owns object j
    object_to_person: Vec<u32>,

    eps: f32,
    pub nits: u32,
    pub nreductions: u32,
    pub optimal_soln_found: bool,
    pub num_assigned: u32,
    pub num_unassigned: u32,
}

impl AuctionSolution {
    /// Checks if current solution is a complete solution that satisfies eps-complementary slackness.
    ///
    /// As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
    /// it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure
    fn is_optimal(&self, solver: &AuctionSolver) -> bool {
        if self.num_unassigned > 0 {
            false
        } else {
            self.ece_satisfied(solver)
        }
    }

    /// tolerance to deal with floating point precision for eCE, due to eps being stored as float 32
    const TOLERANCE: f32 = 1e-7;

    /// Returns True if eps-complementary slackness condition is satisfied
    /// e-CE: for k (all valid j for a given i), max (a_ik - p_k) - eps <= a_ij - p_j
    fn ece_satisfied(&self, solver: &AuctionSolver) -> bool {
        for i in 0..solver.num_rows {
            let num_objects = solver.j_counts[i as usize]; // the number of objects this person is able to bid on

            let start = solver.i_starts_stops[i as usize]; // in flattened index format, the starting index of this person's objects/values
            let j = self.person_to_object[i as usize]; // chosen object

            let mut choice_cost = f32::NAN;
            // first, get cost of choice j
            for idx in 0..num_objects {
                let glob_idx = start + idx;
                let l = solver.column_indices[glob_idx as usize];
                if l == j {
                    choice_cost = solver.values[glob_idx as usize];
                }
            }

            //  k are all possible biddable objects.
            // Go through each, asserting that (a_ij - p_j) + tol >= max(a_ik - p_k) - eps for all k
            let lhs = choice_cost - solver.prices[j as usize] + AuctionSolution::TOLERANCE; // left hand side of inequality

            for idx in 0..num_objects {
                let glob_idx = start + idx;
                let k = solver.column_indices[glob_idx as usize];
                let cost = solver.values[glob_idx as usize];
                if lhs < cost - solver.prices[k as usize] - solver.target_eps {
                    trace!("ECE CONDITION is not met");
                    return false; // The eCE condition is not met.
                }
            }
        }
        trace!("ECE CONDITION met");
        true
    }
}

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
struct AuctionSolver {
    num_rows: u32,
    num_cols: u32,
    prices: Vec<f32>,
    i_starts_stops: Vec<u32>,
    j_counts: Vec<u32>,
    column_indices: Vec<u32>,
    // memory view of all values
    values: Vec<f32>,

    start_eps: f32,
    target_eps: f32,

    max_iterations: u32,

    best_bids: Vec<f32>,
    best_bidders: Vec<u32>,

    // assignment storage
    unassigned_people: Vec<u32>,
    person_to_assignment_idx: Vec<u32>,
}

impl AuctionSolver {
    pub const REDUCTION_FACTOR: f32 = 0.15;
    pub const MAX_ITERATIONS: u32 = if STATIC_MAX_LEVEL as usize == LevelFilter::Trace as usize {
        30
    } else {
        10u32.pow(6)
    };

    pub fn new(
        num_rows: u32,
        num_cols: u32,
        row_indices: &[u32],
        column_indices: Vec<u32>,
        values: Vec<f32>,
    ) -> AuctionSolver {
        debug_assert!(num_rows <= num_cols);
        debug_assert!(row_indices.is_sorted(), "expecting sorted row indices");
        let prices = vec![0.; num_cols as usize];
        let i_starts_stops = cumulative_idxs(row_indices);
        let j_counts = diff(&i_starts_stops);
        // Calculate optimum initial eps and target eps
        // C = max |aij| for all i, j in A(i)
        let c = values
            .iter()
            .max_by(|x, y| x.abs().total_cmp(&y.abs()))
            .expect("values should not be empty");

        // choose eps values
        let start_eps = c / 2.0;
        let target_eps = 1.0 / num_rows as f32;

        AuctionSolver {
            num_rows,
            num_cols,
            i_starts_stops,
            j_counts,
            prices,
            column_indices,
            values,
            start_eps,
            target_eps,

            max_iterations: AuctionSolver::MAX_ITERATIONS,

            best_bids: vec![NONE_F32; num_cols as usize],
            best_bidders: vec![NONE; num_cols as usize],

            unassigned_people: (0..num_rows).collect(),
            person_to_assignment_idx: (0..num_rows).collect(),
        }
    }

    pub fn solve(&mut self) -> AuctionSolution {
        let mut solution = AuctionSolution {
            person_to_object: vec![NONE; self.num_rows as usize],
            object_to_person: vec![NONE; self.num_cols as usize],
            eps: self.start_eps,
            nits: 0,
            nreductions: 0,
            optimal_soln_found: false,
            num_assigned: 0,
            num_unassigned: self.num_rows,
        };
        loop {
            self.bid_and_assign(&mut solution);
            trace!("OBJECTIVE: {}", self.get_objective(&solution));
            solution.nits += 1;

            let is_optimal = (solution.num_unassigned == 0) && solution.is_optimal(&self);
            if is_optimal {
                solution.optimal_soln_found = true;
                break;
            }
            if solution.nits >= self.max_iterations {
                break;
            }
            // full assignment made, but not all people happy, so restart with same prices, but lower eps
            else if solution.num_unassigned == 0 {
                if solution.eps < self.target_eps {
                    // terminate, shown to be optimal for eps < 1/n
                    break;
                }

                solution.eps *= AuctionSolver::REDUCTION_FACTOR;
                trace!("REDUCTION: eps {}", solution.eps);

                // reset all trackers of people and objects
                solution
                    .person_to_object
                    .iter_mut()
                    .for_each(|i_ref| *i_ref = NONE);
                solution
                    .object_to_person
                    .iter_mut()
                    .for_each(|i_ref| *i_ref = NONE);
                solution.num_unassigned = self.num_rows;
                self.unassigned_people
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item_ref)| *item_ref = i as u32);
                self.person_to_assignment_idx
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item_ref)| *item_ref = i as u32);

                solution.nreductions += 1
            }
        }

        solution.num_assigned = self.num_rows - solution.num_unassigned;
        solution
    }

    fn bid_and_assign(&mut self, solution: &mut AuctionSolution) {
        // number of bids to be made
        let num_bidders = solution.num_unassigned as usize;
        let mut bidders = vec![NONE; num_bidders];
        let mut objects_bidded = vec![NONE; num_bidders];
        let mut bids = vec![f32::NEG_INFINITY; num_bidders];

        // BIDDING PHASE
        // each person now makes a bid:
        for nbidder in 0..num_bidders {
            let i = self.unassigned_people[nbidder];
            let num_objects = self.j_counts[i as usize] as usize; // the number of objects this person is able to bid on
            let start = self.i_starts_stops[i as usize] as usize; // in flattened index format, the starting index of this person's objects/values
                                                                  // initially 0 object is considered the best
            let mut jbest = self.column_indices[start];
            let mut costbest = self.values[start];
            // best net reword
            let mut vbest = costbest - self.prices[jbest as usize];
            // second best net reword
            let mut wi = f32::NEG_INFINITY; //0.;
                                            // Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
            for idx in 1..num_objects {
                let glob_idx = start + idx;
                let j = self.column_indices[glob_idx];
                let cost = self.values[glob_idx];
                let vi = cost - self.prices[j as usize];
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
            bidders[nbidder] = i;
            bids[nbidder] = bbest;
            objects_bidded[nbidder] = jbest
        }

        let mut num_successful_bids = 0; // counter of how many succesful bids

        for n in 0..num_bidders {
            // for each bid made,
            let i = bidders[n]; // bidder
            let bid_val = bids[n]; // value
            let jbid = objects_bidded[n] as usize; // object
            if bid_val > self.best_bids[jbid] {
                // if beats current best bid for this object
                if self.best_bidders[jbid] == NONE {
                    // if not overwriting existing bid, increment bid counter
                    num_successful_bids += 1
                }

                // store bid
                self.best_bids[jbid] = bid_val;
                self.best_bidders[jbid] = i;
            }
        }
        trace!("best_bidders {:?}", self.best_bidders);
        trace!("best_bids {:?}", self.best_bids);

        // ASSIGNMENT PHASE
        let mut people_to_unassign_ctr = 0; // counter of how many people have been unassigned
        let mut people_to_assign_ctr = 0; // counter of how many people have been assigned
        let mut bid_ctr = 0;

        for j in 0..self.num_cols as usize {
            let i = self.best_bidders[j];
            if i != NONE {
                self.prices[j] = self.best_bids[j];
                let assignment_idx = self.person_to_assignment_idx[i as usize];

                // unassign previous i (if any)
                let prev_i = solution.object_to_person[j];
                if prev_i != NONE {
                    people_to_unassign_ctr += 1;
                    solution.person_to_object[prev_i as usize] = NONE;

                    // let old i take new i's place in unassigned people list for faster reading
                    self.person_to_assignment_idx[i as usize] = NONE;
                    self.person_to_assignment_idx[prev_i as usize] = assignment_idx;
                    self.unassigned_people[assignment_idx as usize] = prev_i;
                } else {
                    self.unassigned_people[assignment_idx as usize] = NONE; // store empty space in assignment list
                    self.person_to_assignment_idx[i as usize] = NONE;
                }

                // make new assignment
                people_to_assign_ctr += 1;
                solution.person_to_object[i as usize] = j as u32;
                solution.object_to_person[j] = i;

                // bid has been processed, reset best bids store to NONE
                self.best_bidders[j] = NONE;
                self.best_bids[j] = NONE_F32;

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
            solution.num_unassigned as usize,
            self.num_cols as usize,
        );

        trace!("person_to_object: {:?}", solution.person_to_object);
        trace!("unassigned_people: {:?}", self.unassigned_people);
        trace!("prices: {:?}", self.prices);
    }
    /// Returns current objective value of assignments
    fn get_objective(&self, solution: &AuctionSolution) -> f32 {
        let mut obj = 0.;
        for i in 0..self.num_rows {
            // due to the way data is stored, need to go do some searching to find the corresponding value
            // to assignment i -> j
            let j = solution.person_to_object[i as usize]; // chosen j
            if j == NONE {
                // skip any unassigned
                continue;
            }

            let num_objects = self.j_counts[i as usize];
            let start = self.i_starts_stops[i as usize];

            for idx in 0..num_objects {
                let glob_idx = (start + idx) as usize;
                let l = self.column_indices[glob_idx];
                if l == j {
                    obj += self.values[glob_idx];
                }
            }
        }

        return obj;
    }
}

#[cfg(test)]
mod tests {
    use super::{cumulative_idxs, diff, push_all_left, AuctionSolver, NONE};
    use env_logger;
    use log::trace;
    use log::STATIC_MAX_LEVEL;
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;

    #[test]
    fn test_cumulative_idx() {
        let arr = [0, 0, 0, 1, 1, 1, 1];
        let res = cumulative_idxs(&arr);
        assert_eq!(res, [0, 3, 7]);
    }

    #[test]
    fn test_diff() {
        let arr = [0, 3, 7];
        let res = diff(&arr);
        assert_eq!(res, [3, 4]);
    }

    #[test]
    fn test_push_all_left() {
        let mut arr = [NONE, 1, 2, 3, NONE, NONE];
        let mut mapper = [NONE, 1, 2, 3];
        push_all_left(&mut arr, &mut mapper, 3, 3);
        assert_eq!(arr, [3, 1, 2, NONE, NONE, NONE]);
    }
    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
    /// Ensure every row and column has at least one element labelled False, for valid connections

    #[test]
    fn test_sparse_solve() -> Result<(), Box<dyn std::error::Error>> {
        init();
        const NUM_ROWS: u32 = 5;
        const NUM_COLS: u32 = 5;
        let mut row_indices = Vec::with_capacity(NUM_ROWS as usize);
        let mut column_indices = Vec::with_capacity(NUM_COLS as usize);
        let mut values = Vec::with_capacity((NUM_ROWS * NUM_COLS) as usize);
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: f32 = 10.0;
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
        trace!("{:?}", solution,);
        Ok(())
    }
}
