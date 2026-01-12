use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::{bail, DType, Error, Result, Tensor};

/// Specialization of `std::ops::RangeBounds` for `usize` to allow trait objects.
pub trait RangeBound {
    fn start_bound(&self) -> std::ops::Bound<usize>;
    fn end_bound(&self) -> std::ops::Bound<usize>;
}

macro_rules! range_bound {
    ($name:ident) => {
        impl RangeBound for $name<usize> {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
    // Use the marker to designate no generics
    ($name:ident, $marker:expr) => {
        impl RangeBound for $name {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
    // Use the marker to designate no generics
    ($name:ty) => {
        impl RangeBound for $name {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
}

range_bound!(Range);
range_bound!(RangeFrom);
range_bound!(RangeFull, ());
range_bound!(RangeInclusive);
range_bound!(RangeTo);
range_bound!(RangeToInclusive);
range_bound!((std::ops::Bound<usize>, std::ops::Bound<usize>));

impl RangeBound for usize {
    fn end_bound(&self) -> std::ops::Bound<usize> {
        std::ops::Bound::Excluded(self + 1)
    }
    fn start_bound(&self) -> std::ops::Bound<usize> {
        std::ops::Bound::Included(*self)
    }
}

impl Tensor {
    /// Returns a copy of `self` where the values within `ranges` have been replaced with the
    /// content of `src`. This is analogous to slice asignment in `torch`.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Device, Tensor};
    ///
    /// let dev = Device::Cpu;
    /// let tensor = Tensor::arange(0u32, 4 * 5, &dev)?.reshape((4, 5))?;
    /// let src = Tensor::arange(100u32, (2 * 3) + 100, &dev)?.reshape((3, 2))?;
    /// let out = tensor.slice_assign(&[&(..3), &(3..5)], &src)?;
    /// assert_eq!(
    ///     out.to_vec2::<u32>()?,
    ///     &[
    ///         [0, 1, 2, 100, 101],
    ///         [5, 6, 7, 102, 103],
    ///         [10, 11, 12, 104, 105],
    ///         [15, 16, 17, 18, 19]
    ///     ]
    /// );
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn slice_assign(&self, ranges: &[&dyn RangeBound], src: &Tensor) -> Result<Self> {
        let src_dims = src.dims();
        let self_dims = self.dims();
        if self_dims.len() != src_dims.len() {
            bail!(
                "slice-assign requires input with the same rank {} <> {}",
                self_dims.len(),
                src_dims.len()
            )
        }
        if self_dims.len() != ranges.len() {
            bail!(
                "slice-assign requires input with the same rank as there are ranges {} <> {}",
                self_dims.len(),
                ranges.len()
            )
        }
        let mut src = src.clone();
        let mut mask = Self::ones(src.shape(), DType::U8, src.device())?;
        for (i, range) in ranges.iter().enumerate() {
            let start_included = match range.start_bound() {
                std::ops::Bound::Unbounded => 0,
                std::ops::Bound::Included(v) => v,
                std::ops::Bound::Excluded(v) => v + 1,
            };
            let end_excluded = match range.end_bound() {
                std::ops::Bound::Unbounded => self_dims[i],
                std::ops::Bound::Included(v) => v + 1,
                std::ops::Bound::Excluded(v) => v,
            };
            if end_excluded <= start_included {
                bail!("slice-assign: empty range for dim {i}, {start_included} {end_excluded}")
            }
            if self_dims[i] < end_excluded {
                bail!(
                    "slice-assign: upper bound is out of range for dim {i}, {end_excluded} {}",
                    self_dims[i]
                )
            }
            if end_excluded - start_included != src_dims[i] {
                bail!(
                    "slice-assign: the range for dim {i} ({start_included}..{end_excluded}) does not match the size of src {}", src_dims[i]
                )
            }
            src = src.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?;
            mask = mask.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?
        }
        mask.where_cond(/* on_true= */ &src, /* on_false= */ self)
    }
}
