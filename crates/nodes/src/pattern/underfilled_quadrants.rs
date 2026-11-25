use materialcraft_core::{PatternNode, PatternNodeKind};

use super::{
    footprint::SampleGrid,
    pattern_stamp::{CopyPositions, DrawPosition, DrawRun, PatternStamp},
    raster::MIN_STAMP_SIZE,
    tile_runs::{COPIES_PER_BATCH, LEADING_SUBTREES},
};

// Each of the sixteen depth-two quadrants contributes five consecutive ranges:
// its own draw, then the four child subtrees. Overlapping squares expose these
// eight submission groups before any group fills a physical copy batch.
const GROUPS: [&[(usize, usize)]; 8] = [
    &[(60, 61), (63, 65), (20, 30)],
    &[(65, 66), (68, 70), (40, 50)],
    &[(70, 71), (73, 75), (0, 10)],
    &[(76, 78)],
    &[(75, 76), (78, 80), (61, 63)],
    &[(50, 60), (71, 73)],
    &[(30, 40), (66, 68)],
    &[(10, 20)],
];
// After the leading pairs fill their batches, the last root child collects a
// batch from these three ranges in submission order. They are separate in traversal.
const LAST_CHILD_GROUP: usize = 4;
const FOLLOWING_GROUP: usize = 0;
const RANGES_PER_CHILD: usize = 1 + 4 * 5;
const RANGES_PER_TREE: usize = 4 * RANGES_PER_CHILD;

#[derive(Default)]
struct QuadrantPair {
    start: usize,
    copies: u32,
    cut: Option<DrawPosition>,
}

impl QuadrantPair {
    fn record(&mut self, copies: u32, index: usize, retained: bool) {
        let needed = (2 * COPIES_PER_BATCH).saturating_sub(self.copies);
        if needed > 0 && needed <= copies {
            self.cut = Some(if retained && needed < copies {
                DrawPosition { stamp: index, copy: needed }
            } else {
                DrawPosition { stamp: index + usize::from(retained), copy: 0 }
            });
        }
        self.copies = self.copies.saturating_add(copies);
    }
}

#[derive(Default)]
pub(super) struct UnderfilledQuadrants {
    // Indices into the retained stamps at each shallow quadrant, including empty ranges.
    starts: Vec<usize>,
    visits: [usize; 9],
    occupied: Vec<bool>,
    copies: [u32; GROUPS.len()],
    group: usize,
    deeper_shape: Option<i32>,
    first_shape: Option<i32>,
    first_copies: u64,
    pairs: Vec<QuadrantPair>,
    last_child_spans: [CopyPositions; GROUPS[LAST_CHILD_GROUP].len()],
    following_spans: [CopyPositions; GROUPS[FOLLOWING_GROUP].len()],
    tracked_span: Option<usize>,
    // Copy offsets at the starts of the two most recent depth-five subtrees in the last span.
    last_subtree_starts: [u32; 2],
    active_deep_draws: usize,
}

impl UnderfilledQuadrants {
    pub(super) fn visit(&mut self, node: &PatternNode, depth: u32, retained: usize, root: bool) -> bool {
        match node.kind {
            PatternNodeKind::Iterate => return root && node.children.first() == Some(&None),
            PatternNodeKind::Switch | PatternNodeKind::Dot => return true,
            PatternNodeKind::Quadrant => {}
        }
        let Some(visits) = self.visits.get_mut(depth as usize) else { return false };
        *visits += 1;
        if depth == 0 {
            return self.starts.len() == (*visits - 1) * RANGES_PER_TREE;
        }
        if depth > 3 {
            if depth == 5 && self.group == LAST_CHILD_GROUP && self.tracked_span == Some(2) {
                self.last_subtree_starts = [self.last_subtree_starts[1], self.last_child_spans[2].total];
            }
            return true;
        }
        let child = self.starts.len() % RANGES_PER_TREE / RANGES_PER_CHILD;
        let offset = self.starts.len() % RANGES_PER_CHILD;
        self.starts.push(retained);
        if offset == 0 {
            return depth == 1;
        }
        let segment = child * 20 + offset - 1;
        let parent = (offset - 1).is_multiple_of(5);
        if (parent && depth != 2) || (!parent && depth != 3) {
            return false;
        }
        if parent && segment.is_multiple_of(10) {
            self.pairs.push(QuadrantPair { start: retained, ..Default::default() });
        }
        if !parent {
            self.occupied.push(false);
        }
        let Some((group, span)) =
            GROUPS.iter().enumerate().find_map(|(group, spans)| spans.iter().position(|&(lo, hi)| (lo..hi).contains(&segment)).map(|span| (group, span)))
        else {
            return false;
        };
        self.group = group;
        self.tracked_span = (self.visits[0] == 1 && matches!(group, LAST_CHILD_GROUP | FOLLOWING_GROUP)).then_some(span);
        true
    }

    pub(super) fn draw(&mut self, stamp: &PatternStamp, weights: &[f32], grid: Option<&SampleGrid>, index: usize, retained: bool) -> bool {
        let Some(weight) = weights.get(stamp.depth as usize) else { return false };
        let gain = weight * stamp.color[3];
        if stamp.blend != 0 || !(2..=16).contains(&stamp.pattern) || stamp.depth == 0 {
            return false;
        }
        if gain == 0.0 || stamp.size.iter().any(|side| side.abs() < MIN_STAMP_SIZE) {
            return true;
        }
        if !gain.is_normal() {
            return false;
        }
        let Some(grid) = grid else { return false };
        if stamp.depth == 1 {
            // The first-level shape owns a separate copy batch, including all its wrapped copies.
            let Some(count) = grid.copies_per_axis(stamp).and_then(|[x, y]| x.checked_mul(y)) else { return false };
            self.first_copies = self.first_copies.saturating_add(count);
            return self.first_copies <= u64::from(COPIES_PER_BATCH)
                && *self.first_shape.get_or_insert(stamp.pattern) == stamp.pattern
                && self.deeper_shape != Some(stamp.pattern);
        }
        let Some(count) = grid.copy_count(stamp) else { return false };
        if *self.deeper_shape.get_or_insert(stamp.pattern) != stamp.pattern || self.first_shape == Some(stamp.pattern) {
            return false;
        }
        if stamp.depth >= 3 {
            let Some(occupied) = self.occupied.last_mut() else { return false };
            *occupied = true;
        }
        let Some(pair) = self.pairs.last_mut() else { return false };
        pair.record(count, index, retained);
        if let Some(index) = self.tracked_span {
            let spans = if self.group == LAST_CHILD_GROUP { &mut self.last_child_spans } else { &mut self.following_spans };
            if spans[index].record(count, retained).is_none() {
                return false;
            }
        }
        self.active_deep_draws += 1;
        self.copies[self.group] = self.copies[self.group].saturating_add(count);
        true
    }

    fn append_batch(&self, group: usize, spans: &[CopyPositions], leading: &mut Vec<(DrawPosition, DrawPosition)>) -> Option<()> {
        let mut remaining = COPIES_PER_BATCH;
        for (&(first, _), span) in GROUPS[group].iter().zip(spans) {
            let start = *self.starts.get(first + 1 + first / 20)?;
            let count = remaining.min(span.total);
            let end = span.at(count);
            leading.push((DrawPosition { stamp: start, copy: 0 }, DrawPosition { stamp: start.checked_add(end.stamp)?, copy: end.copy }));
            remaining -= count;
            if remaining == 0 {
                return Some(());
            }
        }
        None
    }

    fn full_batches<'a>(&self, stamps: &'a [PatternStamp], trees: usize) -> Option<Vec<DrawRun<'a>>> {
        // Every quadrant from depth two through eight emits a draw. When a pair
        // fills a third batch the competing streams can interleave before this cut.
        if self.active_deep_draws != self.visits[2..].iter().sum::<usize>()
            || self.pairs.len() != trees * 8
            || self.pairs.iter().any(|pair| pair.copies / COPIES_PER_BATCH != 2)
        {
            return None;
        }
        let position = |stamp| DrawPosition { stamp, copy: 0 };
        let mut leading = Vec::new();
        for group in LEADING_SUBTREES {
            let pair = self.pairs.get(group / 8)?;
            leading.push((position(pair.start), pair.cut?));
        }
        let [head, middle, tail] = &self.last_child_spans;
        let before_tail = u64::from(head.total) + u64::from(middle.total);
        let copies = before_tail + u64::from(tail.total);
        if copies / u64::from(COPIES_PER_BATCH) == 1 {
            self.append_batch(LAST_CHILD_GROUP, &self.last_child_spans, &mut leading)?;
            // The next group submits first when the previous batch reaches either of its final two depth-five subtrees.
            if before_tail + u64::from(self.last_subtree_starts[0]) < u64::from(COPIES_PER_BATCH) {
                self.append_batch(FOLLOWING_GROUP, &self.following_spans, &mut leading)?;
            }
        }
        leading.retain(|(start, end)| start < end);
        let mut runs: Vec<_> = leading.iter().map(|&(start, end)| DrawRun::between(stamps, start, end)).collect::<Option<_>>()?;
        // Remove the submitted pieces in traversal order. A wrapped stamp may contribute to both sides of a cut.
        leading.sort_unstable();
        let mut submitted = leading.into_iter().peekable();
        for tree in 0..trees {
            for child in 0..4 {
                let base = tree * RANGES_PER_TREE + child * RANGES_PER_CHILD;
                let mut start = position(*self.starts.get(base + 1)?);
                let end = position(*self.starts.get(base + RANGES_PER_CHILD)?);
                while submitted.peek().is_some_and(|(first, _)| *first < end) {
                    let (first, next) = submitted.next()?;
                    runs.push(DrawRun::between(stamps, start, first)?);
                    start = next;
                }
                runs.push(DrawRun::between(stamps, start, end)?);
            }
        }
        Some(runs)
    }

    pub(super) fn finish(mut self, stamps: &[PatternStamp]) -> Option<Vec<DrawRun<'_>>> {
        let trees = self.visits[0];
        if trees == 0
            || self.visits.iter().enumerate().any(|(depth, &count)| count != trees << (2 * depth))
            || self.starts.len() != trees * RANGES_PER_TREE
            || self.occupied.len() != trees * 64
            || self.occupied.contains(&false)
        {
            return None;
        }
        self.starts.push(stamps.len());
        let filled = self.copies.iter().any(|&copies| copies >= COPIES_PER_BATCH);
        let mut runs = if filled { self.full_batches(stamps, trees)? } else { Vec::new() };
        let mut range = |tree: usize, first: usize, end: usize| {
            let start = *self.starts.get(tree * RANGES_PER_TREE + first)?;
            let stop = *self.starts.get(tree * RANGES_PER_TREE + end)?;
            let slice = stamps.get(start..stop)?;
            if !slice.is_empty() {
                runs.push(DrawRun::whole(slice));
            }
            Some(())
        };
        if !filled {
            for spans in GROUPS {
                for tree in 0..trees {
                    for &(first, end) in spans {
                        // Insert the first-level marker at the start of each twenty-range child.
                        let child_marker = 1 + first / 20;
                        range(tree, child_marker + first, child_marker + end)?;
                    }
                }
            }
        }
        // The first-level shape has its own submissions, in reverse child order across trees.
        for child in (0..4).rev() {
            for tree in 0..trees {
                range(tree, child * RANGES_PER_CHILD, child * RANGES_PER_CHILD + 1)?;
            }
        }
        Some(runs)
    }
}
