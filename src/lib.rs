#![feature(total_cmp)]
const NONE: u32 = u32::MAX;

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

fn push_all_left(data: &mut [u32], mapper: &mut [u32], num_ints: usize) {
    // Given an array of valid positive integers (size <size>) and NONEs (u32::MAX), arrange so that all valid positive integers are at the start of the array.
    // Provided with N (number of valid positive integers) for speed increase.
    // eg [4294967295, 1, 2, 3, 4294967295, 4294967295] -> [3, 1, 2, 4294967295, 4294967295, 4294967295] (order not important).
    // Also updates mapper in tandem, a 1d array in which the ith idx gives the position of integer i in the array data.
    // All modifications are inplace.

    if num_ints == 0 {
        return;
    }
    let size = data.len();

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

/// Solver for auction problem
/// Which finds an assignment of N people -> M objects, by having people 'bid' for objects
struct AuctionSolver {
    num_rows: usize,
    num_cols: usize,
    prices: Vec<f64>,
    i_starts_stops: Vec<u32>,
    j_counts: Vec<u32>,
    column_indices: Vec<u32>,
    // memory view of all values
    values: Vec<f32>,
    // index i gives the object, j, owned by person i
    person_to_object: Vec<u32>,
    // index j gives the person, i, who owns object j
    // object_to_person: Vec<u32>,
    eps: f32,
    target_eps: f32,
    theta: f32,
    //meta
    nits: u32,
    nreductions: u32,
    ece_satisfied: bool,
    optimal_soln_found: bool,
    final_eps: f32,
    // assignment storage
    num_unassigned: u32,
    unassigned_people: Vec<u32>,
    person_to_assignment_idx: Vec<u32>,
}

impl AuctionSolver {
    fn new(
        num_rows: usize,
        num_cols: usize,
        row_indices: &[u32],
        column_indices: Vec<u32>,
        values: Vec<f32>,
    ) -> AuctionSolver {
        let prices = vec![0.; num_cols];
        let i_starts_stops = cumulative_idxs(row_indices);
        let j_counts = diff(&i_starts_stops);
        // Calculate optimum initial eps and target eps
        // = max |aij| for all i, j in A(i)
        let c = values
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .expect("values should be empty");

        // choose eps values
        let eps = c / 2.0;
        let target_eps = 1.0 / num_rows as f32;
        let theta = 0.15; // reduction factor

        AuctionSolver {
            num_rows,
            num_cols,
            i_starts_stops,
            j_counts,
            prices,
            column_indices,
            values,
            person_to_object: vec![NONE; num_rows],
            // object_to_person: vec![NONE, num_cols],
            eps,
            target_eps,
            theta,
            nits: 0,
            nreductions: 0,
            ece_satisfied: false,
            optimal_soln_found: false,
            final_eps: 0.,
            num_unassigned: num_rows as u32,
            unassigned_people: (0..num_rows as u32).collect(),
            person_to_assignment_idx: (0..num_rows as u32).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cumulative_idxs, diff, push_all_left, NONE};

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
        push_all_left(&mut arr, &mut mapper, 3);
        assert_eq!(arr, [3, 1, 2, NONE, NONE, NONE]);
    }
}
