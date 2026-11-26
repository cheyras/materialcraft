use super::{COPIES_PER_BATCH, ReceivingPair, SUBTREES, copies};
use std::ops::Range;

fn remainder(ends: &[u32], group: usize, submitted: &Range<u32>, batches: u32) -> Option<Range<u32>> {
    let half = copies(ends, group * SUBTREES..(group + 8) * SUBTREES)?;
    (submitted.start == half.start && submitted.end - submitted.start == batches * COPIES_PER_BATCH && submitted.end <= half.end)
        .then_some(submitted.end..half.end)
}

pub(super) fn ranges(ends: &[u32], receivers: &[ReceivingPair], leading: &[Range<u32>], outer_tail: &Range<u32>) -> Option<Vec<Range<u32>>> {
    let [first, middle, last] = receivers else { return None };
    if receivers.iter().any(|pair| pair.split_width != SUBTREES || pair.pending.len() != SUBTREES) {
        return None;
    }
    let [lower, upper, third, ..] = leading else { return None };
    let lower = remainder(ends, 24, lower, 2)?;
    let upper = remainder(ends, 40, upper, 2)?;
    let third = remainder(ends, 8, third, 1)?;

    // The first receiver transfers a whole sibling, then a half and a quarter
    // of the following sibling. Their descendants can finish different streams.
    let whole = first.pending.start;
    let half = first.pending.end;
    let quarter = half + SUBTREES / 2;
    let children = |start: usize, offsets: Range<usize>| copies(ends, start + offsets.start..start + offsets.end);
    let donor = children(quarter, 0..1)?;
    let room = COPIES_PER_BATCH.checked_sub(third.end - third.start)?;
    if room == 0 || room > donor.end - donor.start {
        return None;
    }
    let cut = donor.start + room;
    let mut runs = leading.to_vec();
    runs.extend([third, donor.start..cut, first.tail.clone(), copies(ends, middle.pending.clone())?, last.tail.clone()]);

    // After the third half fills, each unfinished stream receives its remaining
    // siblings. A stream that fills at this stage has a different submission time.
    let streams = [
        vec![children(quarter, 1..2)?],
        vec![middle.tail.clone(), children(whole, 0..8)?, children(half, 0..2)?],
        vec![copies(ends, 60 * SUBTREES..62 * SUBTREES)?, children(whole, 8..16)?, copies(ends, last.pending.clone())?],
        vec![outer_tail.clone(), children(half, 3..8)?],
        vec![upper, copies(ends, 56 * SUBTREES..58 * SUBTREES)?, children(quarter, 2..3)?],
        vec![lower, copies(ends, 52 * SUBTREES..54 * SUBTREES)?],
        vec![cut..donor.end, children(quarter, 3..4)?, children(half, 2..3)?],
    ];
    for stream in streams {
        let length = stream.iter().try_fold(0_u32, |sum, range| sum.checked_add(range.end.checked_sub(range.start)?))?;
        if length >= COPIES_PER_BATCH {
            return None;
        }
        runs.extend(stream.into_iter().filter(|range| !range.is_empty()));
    }
    Some(runs)
}
