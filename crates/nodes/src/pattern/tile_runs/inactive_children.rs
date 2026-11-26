use super::{COPIES_PER_BATCH, LEADING_SUBTREES};
use std::ops::Range;

// A depth-three quadrant contains sixteen depth-five subtrees. A boundary can
// split a tiled stamp, so convert subtree boundaries to physical copies before filling a batch.
const SUBTREES: usize = 16;

fn boundary(ends: &[u32], subtree: usize) -> Option<u32> {
    if subtree == 0 { Some(0) } else { ends.get(subtree - 1).copied() }
}

fn copies(ends: &[u32], subtrees: Range<usize>) -> Option<Range<u32>> {
    Some(boundary(ends, subtrees.start)?..boundary(ends, subtrees.end)?)
}

// At a full batch, half the pending siblings can pass to a later run. When the
// final sibling of a larger subtree is pending, its first two children can pass together.
fn pending_siblings(subtree: usize) -> Option<(Range<usize>, usize)> {
    for width in [1, 4, SUBTREES] {
        let group = subtree / width;
        let pending = 3 - group % 4;
        let next = (group + 1) * width;
        if pending >= 2 {
            return Some((next..next + pending.div_ceil(2) * width, width));
        }
        if pending == 1 && width > 1 {
            return Some((next..next + width / 2, width / 4));
        }
    }
    None
}

fn receiving_pair(ends: &[u32], partner: usize, main: usize, first: bool) -> Option<Vec<Range<u32>>> {
    let partner = copies(ends, partner * SUBTREES..(partner + 2) * SUBTREES)?;
    let extent = main * SUBTREES..(main + 8) * SUBTREES;
    let start = boundary(ends, extent.start)?;
    let room = COPIES_PER_BATCH.checked_sub(partner.end - partner.start)?;
    let cut = start.checked_add(room)?;
    let subtree = ends.partition_point(|&end| end < cut);
    if room == 0 || !extent.contains(&subtree) {
        return None;
    }
    let (deferred, width) = pending_siblings(subtree)?;
    let mut skips = vec![deferred.clone()];
    if first {
        // The first receiving pair hands off a second block before continuing.
        let extra = match width {
            1 => {
                let parent = subtree / 4;
                let pending = 3 - parent % 4;
                if pending < 2 {
                    return None;
                }
                let sibling = (parent + 1) * 4;
                sibling..sibling + pending.div_ceil(2) * 4
            }
            4 => deferred.end..deferred.end + 2,
            _ => return None,
        };
        skips.push(extra);
    }
    let mut runs = vec![partner];
    let mut remaining = room + COPIES_PER_BATCH;
    let mut next = extent.start;
    for skip in skips.into_iter().chain(std::iter::once(extent.end..extent.end)) {
        if skip.start < next || skip.end > extent.end {
            return None;
        }
        let available = copies(ends, next..skip.start)?;
        let take = remaining.min(available.end - available.start);
        if take > 0 {
            runs.push(available.start..available.start + take);
            remaining -= take;
        }
        next = skip.end;
    }
    (remaining == 0).then_some(runs)
}

pub(super) fn ranges(ends: &[u32]) -> Option<Vec<Range<u32>>> {
    if ends.len() != 64 * SUBTREES {
        return None;
    }
    let mut runs = Vec::new();
    for group in LEADING_SUBTREES {
        let half = copies(ends, group * SUBTREES..(group + 8) * SUBTREES)?;
        if !(COPIES_PER_BATCH..2 * COPIES_PER_BATCH).contains(&(half.end - half.start)) {
            return None;
        }
        runs.push(half.start..half.start + COPIES_PER_BATCH);
    }
    // The final root quadrant sends one pair of depth-three subtrees to each
    // receiving half. Each receiver can submit two complete batches.
    for (partner, main) in [(50, 16), (58, 0), (54, 32)] {
        let Some(pair) = receiving_pair(ends, partner, main, main == 16) else { break };
        runs.extend(pair);
    }
    // Preserve every remaining physical copy in traversal order.
    let mut submitted = runs.clone();
    submitted.sort_unstable_by_key(|range| range.start);
    let mut next = 0;
    for range in submitted {
        if range.start < next {
            return None;
        }
        if next < range.start {
            runs.push(next..range.start);
        }
        next = range.end;
    }
    let end = *ends.last()?;
    if next < end {
        runs.push(next..end);
    }
    Some(runs)
}
