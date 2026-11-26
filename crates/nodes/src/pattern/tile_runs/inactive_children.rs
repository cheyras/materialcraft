use super::{COPIES_PER_BATCH, LEADING_SUBTREES};
use std::ops::Range;

mod completion;
#[cfg(test)]
mod tests;
mod wide_transfer;

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

struct ReceivingPair {
    runs: Vec<Range<u32>>,
    tail: Range<u32>,
    pending: Range<usize>,
    handoff: Option<Range<u32>>,
    split_width: usize,
}

fn receiving_pair(ends: &[u32], partner: usize, main: usize, first: bool) -> Option<ReceivingPair> {
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
    let mut handoff = None;
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
            // After a whole depth-three sibling, the following half and quarter
            // of a sibling also transfer before this receiver resumes.
            SUBTREES if deferred.len() == SUBTREES => deferred.end..deferred.end + SUBTREES / 2 + SUBTREES / 4,
            _ => return None,
        };
        handoff = Some(copies(ends, extra.clone())?);
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
    if remaining != 0 {
        return None;
    }
    let tail = runs.last()?.end..boundary(ends, extent.end)?;
    Some(ReceivingPair { runs, tail, pending: deferred, handoff, split_width: width })
}

fn completed_half(ends: &[u32], group: usize, partner: usize) -> Option<[Range<u32>; 2]> {
    let half = copies(ends, group * SUBTREES..(group + 8) * SUBTREES)?;
    let remainder = half.start.checked_add(COPIES_PER_BATCH)?..half.end;
    let room = COPIES_PER_BATCH.checked_sub(remainder.end.checked_sub(remainder.start)?)?;
    let incoming = copies(ends, partner * SUBTREES..(partner + 2) * SUBTREES)?;
    if room == 0 || room >= incoming.end - incoming.start {
        return None;
    }
    Some([remainder, incoming.start..incoming.start + room])
}

fn full_outer_pair(ends: &[u32]) -> Option<[Range<u32>; 2]> {
    let preceding = copies(ends, 60 * SUBTREES..62 * SUBTREES)?;
    let own = copies(ends, 62 * SUBTREES..64 * SUBTREES)?;
    let incoming = copies(ends, 48 * SUBTREES..50 * SUBTREES)?;
    let room = COPIES_PER_BATCH.checked_sub(own.end - own.start)?;
    // An independently full outer pair has a different submission boundary.
    if preceding.end - preceding.start >= COPIES_PER_BATCH || room == 0 || !(room..COPIES_PER_BATCH).contains(&(incoming.end - incoming.start)) {
        return None;
    }
    Some([own, incoming.start..incoming.start + room])
}

fn following_batches(ends: &[u32], receivers: &[ReceivingPair], leading: &[Range<u32>], outer_tail: Option<&Range<u32>>) -> Option<Vec<Range<u32>>> {
    let [first, middle, last] = receivers else { return None };
    if first.split_width == SUBTREES {
        return wide_transfer::ranges(ends, receivers, leading, outer_tail?);
    }
    let handoff = first.handoff.as_ref()?;
    let middle_siblings = copies(ends, middle.pending.clone())?;
    let last_siblings = copies(ends, last.pending.clone())?;
    if first.tail.start < handoff.end || middle.tail.start < middle_siblings.end || last.tail.start < last_siblings.end {
        return None;
    }
    let third = copies(ends, 8 * SUBTREES..16 * SUBTREES)?;
    let remaining = (third.end - third.start).checked_sub(COPIES_PER_BATCH)?;
    // Filling the third half with the transferred siblings changes the continuation.
    if remaining.checked_add(handoff.end - handoff.start)? >= COPIES_PER_BATCH {
        return None;
    }
    let halves = [completed_half(ends, 24, 52)?, completed_half(ends, 40, 56)?];
    // Passing a depth-four sibling from the first receiver puts the other half first.
    let [lower, upper] = &halves;
    let order = if first.split_width == 4 { [upper, lower] } else { [lower, upper] };
    let mut full = leading.to_vec();
    full.extend(order.into_iter().flatten().cloned());
    let tails = [first.tail.clone(), middle_siblings, last.tail.clone()];
    if outer_tail.is_none()
        && let Some(completed) = completion::ranges(ends, receivers, &halves, third.start + COPIES_PER_BATCH..third.end, &full, &tails)
    {
        return Some(completed);
    }
    full.extend(tails);
    Some(full)
}

pub(super) fn ranges(ends: &[u32]) -> Option<Vec<Range<u32>>> {
    if ends.len() != 64 * SUBTREES {
        return None;
    }
    let mut runs = Vec::new();
    for group in LEADING_SUBTREES {
        let half = copies(ends, group * SUBTREES..(group + 8) * SUBTREES)?;
        let batches = (half.end - half.start) / COPIES_PER_BATCH;
        if !(1..=2).contains(&batches) {
            return None;
        }
        runs.push(half.start..half.start + batches * COPIES_PER_BATCH);
    }
    // The last two depth-three subtrees continue into the first two of the same
    // root quadrant. A complete batch submits before work reaches the receivers.
    let outer_tail = if let Some([own, incoming]) = full_outer_pair(ends) {
        let tail = incoming.end..boundary(ends, 50 * SUBTREES)?;
        runs.extend([own, incoming]);
        Some(tail)
    } else {
        None
    };
    // The final root quadrant sends one pair of depth-three subtrees to each
    // receiving half. Each receiver can submit two complete batches.
    let mut receivers = Vec::with_capacity(3);
    for (partner, main) in [(50, 16), (58, 0), (54, 32)] {
        let Some(pair) = receiving_pair(ends, partner, main, main == 16) else { break };
        runs.extend_from_slice(&pair.runs);
        receivers.push(pair);
    }
    if let Some(following) = following_batches(ends, &receivers, &runs, outer_tail.as_ref()) {
        runs = following;
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
