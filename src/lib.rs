#![feature(array_methods, array_map, is_sorted, total_cmp)]
use anyhow;
use anyhow::{anyhow as anyhow_error, ensure, Result};
use log;
use log::{info, trace};
use num_integer::Integer;
use num_iter;
use num_traits::{AsPrimitive, FromPrimitive, NumAssign, PrimInt, Unsigned};
use std::fmt::{Debug, Display};
pub type Float = f64;

/// Solution of the linear assignment problem found by AuctionSolver
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
        + NumAssign
        + Integer,
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
        + NumAssign
        + Integer,
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
        target_eps: Option<Float>,
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

        let float_num_rows: Float = self.num_rows.as_();

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
    pub fn num_of_arcs(&self) -> usize {
        self.column_indices.len()
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

    #[inline]
    pub fn solve(
        &mut self,
        solution: &mut AuctionSolution<I>,
        maximize: bool,
        start_eps: Option<Float>,
    ) -> Result<(), anyhow::Error> {
        self.validate_input()?;

        if maximize {
            self.values.iter_mut().for_each(|v_ref| *v_ref *= -1.);
        }

        self.prices.clear();
        self.prices.resize(self.num_cols.as_(), 0.);
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
            solution.eps = self.target_eps - Float::EPSILON;
        } else {
            solution.eps = if let Some(eps) = start_eps {
                eps
            } else {
                c / 2.0
            };
        }
        solution.nits = 0;
        solution.nreductions = 0;
        solution.optimal_soln_found = false;
        solution.num_unassigned = self.num_rows;

        loop {
            self.bid_and_assign(solution);
            solution.nits += 1;

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

    pub fn solve_approx(
        &mut self,
        solution: &mut AuctionSolution<I>,
        maximize: bool,
        eps: Option<Float>,
    ) -> Result<(), anyhow::Error> {
        self.validate_input()?;

        // if maximize {
        //     self.values.iter_mut().for_each(|v_ref| *v_ref *= -1.);
        // }

        let num_cols_f: Float = self.num_cols.as_();
        let eps = if let Some(eps) = eps {
            eps
        } else {
            1.0 / num_cols_f
        };

        solution.person_to_object.clear();
        solution
            .person_to_object
            .resize(self.num_rows.as_(), I::max_value());
        solution.object_to_person.clear();
        solution
            .object_to_person
            .resize(self.num_cols.as_(), I::max_value());
        solution.eps = eps;
        solution.nits = 0;
        solution.nreductions = 0;
        solution.optimal_soln_found = false;
        solution.num_unassigned = self.num_rows;

        let (w_min, w_max) =
            self.values
                .iter()
                .fold((Float::INFINITY, Float::NEG_INFINITY), |(min, max), el| {
                    (
                        if min < *el { min } else { *el },
                        if max > *el { max } else { *el },
                    )
                });

        let price_threshold = (num_cols_f / 2.) * (w_max - w_min + eps);
        trace!("APPROX: price threshold: {}", price_threshold);

        let mut ustack = Vec::with_capacity(self.num_rows.as_());
        ustack.extend(num_iter::range(I::zero(), self.num_rows).rev());

        while let Some(u_i) = ustack.pop() {
            solution.nits += 1;
            let u: usize = u_i.as_();
            trace!("APPROX u: {}", u);
            trace!("APPROX prices {:?}", self.prices);
            let start: usize = self.i_starts_stops[u].as_();
            let num_of_u_objects: usize = self.j_counts[u].as_();
            let mut min_new_price = Float::INFINITY;
            let mut min_edge_cost = Float::INFINITY;
            let mut matched_v_i: I = I::zero();

            let mut second_min_new_price = Float::INFINITY;

            // choice rule
            for idx in 0..num_of_u_objects {
                let glob_idx = start + idx;
                let j: I = self.column_indices[glob_idx];
                let j_usize: usize = j.as_();
                let edge_cost = self.values[glob_idx];
                let new_price = edge_cost + self.prices[j_usize];
                if new_price < min_new_price {
                    matched_v_i = j;
                    second_min_new_price = min_new_price;
                    min_new_price = new_price;
                    min_edge_cost = edge_cost;
                } else if new_price < second_min_new_price {
                    second_min_new_price = new_price;
                }
            }
            let matched_v: usize = matched_v_i.as_();
            trace!(
                "APPROX: matched_v: {}, min_new_price: {}",
                matched_v,
                min_new_price
            );

            if self.prices[matched_v] > price_threshold {
                continue;
            }

            // update rule
            if second_min_new_price.is_finite() {
                self.prices[matched_v] = second_min_new_price - min_edge_cost + eps;
            } else {
                self.prices[matched_v] += eps;
            }

            let moved_out_u_i = solution.object_to_person[matched_v];

            if moved_out_u_i != I::max_value() {
                trace!("APPROX - move out {}", moved_out_u_i);
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
        trace!("APPROX OBJECTIVE: {:?}", self.get_objective(solution));
        trace!("APPROX person_to_object: {:?}", solution.person_to_object);
        trace!("APPROX prices: {:?}", self.prices);
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
                let mut jbest = I::zero();
                let mut costbest = Float::INFINITY;
                // best net reword
                let mut vbest = Float::INFINITY;
                // second best net reword
                let mut vsecond_best = Float::INFINITY; //0.;
                                                        // Go through each object, storing its index & cost if vi is largest, and value if vi is second largest
                for idx in 0..num_objects {
                    let glob_idx = start + idx;
                    let j: I = self.column_indices[glob_idx];
                    let j_usize: usize = j.as_();
                    let cost = self.values[glob_idx];
                    let vi = cost + self.prices[j_usize];
                    if vi < vbest {
                        // if best so far (or first entry)
                        jbest = j;
                        vsecond_best = vbest; // store current vbest as second best, wi
                        vbest = vi;
                        costbest = cost;
                    } else if vi < vsecond_best {
                        vsecond_best = vi;
                    }
                }

                let bbest = vsecond_best - costbest + solution.eps; // value of new bid

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
    /// Returns current objective value of assignments.
    /// Checks for the sign of the first element to return positive objective.
    pub fn get_objective(&self, solution: &AuctionSolution<I>) -> Float {
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

    fn get_toleration(&self, max_abs_cost: Float) -> Float {
        1.0 / 2_u64.pow(Float::MANTISSA_DIGITS - (max_abs_cost + 1e-7).log2() as u32) as Float
    }

    /// Checks if current solution is a complete solution that satisfies eps-complementary slackness.
    ///
    /// As eps-complementary slackness is preserved through each iteration, and we start with an empty set,
    /// it is true that any solution satisfies eps-complementary slackness. Will add a check to be sure
    /// Returns True if eps-complementary slackness condition is satisfied
    /// e-CS: for k (all valid j for a given i), min (a_ik + p_k) + eps >= a_ij + p_j
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
            // Go through each, asserting that min(a_ik + p_k) + eps >= (a_ij + p_j) - tol for all k.
            // Tolerance is added to deal with floating point precision for eCS, due to eps being stored as float
            let j_usize: usize = j.as_();
            let lhs: Float = choice_cost + self.prices[j_usize] - toleration; // left hand side of inequality

            for idx in num_iter::range(I::zero(), num_objects) {
                let glob_idx: usize = (start + idx).as_();
                let k: usize = self.column_indices[glob_idx].as_();
                let cost: Float = self.values[glob_idx];
                if lhs > cost + self.prices[k] + self.target_eps {
                    trace!("ECS CONDITION is not met");
                    return false;
                }
            }
        }
        trace!("ECS CONDITION met");
        true
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
    use super::{push_all_left, AuctionSolution, AuctionSolver, Float};
    use env_logger;
    use log::{debug, trace};
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use reservoir_sampling::unweighted::core::r as reservoir_sample;

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

    fn solver_with_ksparse_input(
        num_rows: u32,
        num_cols: u32,
        arcs_per_person: usize,
    ) -> (AuctionSolver<u32>, AuctionSolution<u32>) {
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: Float = 10.0;
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
        init();
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
            debug!("approx {:?}", approx_solution);
            debug!("auction {:?}", solution);
            assert_eq!(solution.num_unassigned, 0);
            assert_eq!(approx_solution.num_unassigned, 0);
        }
        Ok(())
    }

    #[test]
    fn test_random_no_perfect_matching() -> Result<(), Box<dyn std::error::Error>> {
        init();
        const NUM_ROWS: u32 = 9;
        const NUM_COLS: u32 = 9;
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: Float = 10.0;
        let between = Uniform::from(0.0..MAX_VALUE);
        const ARCS_PER_PERSON: usize = 3;

        let (mut solver, mut solution) = AuctionSolver::new(
            NUM_ROWS as usize,
            NUM_COLS as usize,
            ARCS_PER_PERSON * NUM_ROWS as usize,
        );

        let mut approx_solution = solution.clone();
        solver.init(NUM_ROWS, NUM_COLS, None, None).unwrap();

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
        init();
        const NUM_ROWS: u32 = 90;
        const NUM_COLS: u32 = 900;
        let mut val_rng = ChaCha8Rng::seed_from_u64(1);
        let mut filter_rng = ChaCha8Rng::seed_from_u64(2);

        const MAX_VALUE: Float = 10.0;
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
                let j_values = j_samples.map(|_| between.sample(&mut val_rng));
                solver
                    .extend_from_values(i, j_samples.as_slice(), j_values.as_slice())
                    .unwrap();
            });
        assert!(solver.i_starts_stops.len() == NUM_ROWS as usize + 1);
        solver
            .solve_approx(&mut approx_solution, false, None)
            .unwrap();
        let approx_objective = solver.get_objective(&approx_solution);
        solver
            .solve(&mut solution, false, Some(1.0 / NUM_ROWS as Float))
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
                    let values = row_ref.iter().map(|v| ((*v) as Float)).collect::<Vec<_>>();
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
            assert!(solution.optimal_soln_found);
            assert!(solution.num_unassigned == 0);
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

        Ok(())
    }
}
