use num_traits::{AsPrimitive, FromPrimitive, NumAssign, PrimInt, Unsigned};
use std::fmt::{Debug, Display};

pub trait UnsignedInt:
    PrimInt
    + Unsigned
    + Display
    + Debug
    + AsPrimitive<usize>
    + AsPrimitive<f64>
    + FromPrimitive
    + NumAssign
{
}

///
/// Solution of the linear assignment problem
///
#[derive(Debug, Clone)]
pub struct AuctionSolution<I>
where
    I: UnsignedInt,
{
    /// index i gives the object, j, owned by person i
    ///
    /// Unassigned people are marked by MAX value of the integer type (u32::MAX for u32)
    pub person_to_object: Vec<I>,
    /// index j gives the person, i, who owns object j
    ///
    /// Unassigned objects are marked by MAX value of the integer type (u32::MAX for u32)
    pub object_to_person: Vec<I>,
    /// number of unnassigned people in case perfect matching doesn't exist
    pub num_unassigned: I,
    /// found solution is ε-optimal if perfect matching exists. For integer weights small enough ε
    /// gives optimum.
    pub eps: f64,
}

impl<I> AuctionSolution<I>
where
    I: UnsignedInt,
{
    pub fn new(row_capacity: usize, column_capacity: usize) -> AuctionSolution<I> {
        AuctionSolution::<I> {
            person_to_object: Vec::with_capacity(row_capacity),
            object_to_person: Vec::with_capacity(column_capacity),
            eps: f64::NAN,
            num_unassigned: I::max_value(),
        }
    }
}
