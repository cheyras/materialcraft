use super::{COPIES_PER_BATCH, Piece, Range, RemainingCopies, SiblingOffsets, Submission, UnderfilledQuadrants};
use crate::pattern::underfilled_quadrants::REPEATED_GROUPS;

impl SiblingOffsets {
    fn depth_five_starts(&self, range: Range<u32>) -> &[u32] {
        let starts = &self.starts[2];
        let remaining = &starts[starts.partition_point(|&start| start < range.start)..];
        &remaining[..remaining.partition_point(|&start| start < range.end)]
    }
}

impl UnderfilledQuadrants {
    pub(super) fn carry_into_third_tree(&self, lower: &[Piece<'_>], upper: &[Piece<'_>]) -> Option<Vec<Submission>> {
        let mut submissions = Vec::new();
        for (slot, carried) in [lower, upper].into_iter().enumerate() {
            let span = &self.repeated.get(slot)?.third_tree;
            let count = Self::piece_count(carried)?;
            if count >= COPIES_PER_BATCH || count.checked_add(span.total)? / COPIES_PER_BATCH != 2 {
                // Another full submission changes when the other pair can receive work.
                return None;
            }
            let mut pieces = carried.to_vec();
            pieces.push((self.span_start(2, REPEATED_GROUPS[slot], 0)?, span, 0..span.total));
            Self::append_copy_window(&pieces, 0, 2 * COPIES_PER_BATCH, &mut submissions)?;
        }
        Some(submissions)
    }

    pub(super) fn carry_single_transfers(
        &self,
        first: &RemainingCopies<'_>,
        second: &RemainingCopies<'_>,
        lower: &RemainingCopies<'_>,
        upper: &RemainingCopies<'_>,
        last: &RemainingCopies<'_>,
    ) -> Option<Vec<Submission>> {
        if self.next_collecting[3].branches.unvisited_subtrees(last.before.start) != 2
            || [upper, last].iter().any(|pair| pair.deferred[0].is_empty() || !pair.deferred[1].is_empty())
        {
            return None;
        }
        let [(start, span, left), (_, _, right)] = second.deferred();
        let offsets = &self.next_collecting[1].branches;
        // The receiving group filled in the first of two adjacent depth-three children.
        let siblings = &offsets.starts[0];
        let index = siblings.binary_search(&left.start).ok()?;
        if left.end != right.start || siblings.get(index + 1) != Some(&right.start) || siblings.get(index + 2).copied().unwrap_or(span.total) != right.end {
            return None;
        }
        let cut = left.start.checked_add(COPIES_PER_BATCH.checked_sub(Self::piece_count(&first.kept())?)?)?;
        if cut > left.end {
            return None;
        }
        let [lower_head, lower_tail] = lower.kept();
        let [upper_head, upper_tail] = upper.kept();
        let lower_pieces = [lower_head, lower_tail, upper.piece(upper.deferred[0].clone())];
        let mut upper_pieces = vec![upper_head, upper_tail, last.piece(last.deferred[0].clone())];
        match offsets.depth_five_starts(cut..left.end).len() {
            3 => {
                // The upper pair also receives the third child of the second depth-four quadrant.
                let children = offsets.depth_five_starts(right);
                if children.len() != 16 {
                    return None;
                }
                upper_pieces.push((start, span, children[6]..children[7]));
            }
            4 => {}
            _ => return None,
        }
        self.carry_into_third_tree(&lower_pieces, &upper_pieces)
    }
}
