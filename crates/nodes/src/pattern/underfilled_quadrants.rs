use std::ops::Range;

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
// Indexed by child slot within the last root quadrant. Each group can fill a
// batch with its depth-two draw and its last two child subtrees alone.
const COLLECTING_GROUPS: [usize; 4] = [0, 1, 2, LAST_CHILD_GROUP];
// The first two leading pairs receive pending subtrees before continuing in the next tree.
const REPEATED_GROUPS: [usize; 2] = [6, 5];
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
struct FollowingPair {
    children: usize,
    second_child_tail: [u32; 2],
    third_child: u32,
    fourth_child: u32,
    fourth_child_branches: [u32; 4],
    deferred_subtrees: [u32; 3],
}

impl FollowingPair {
    fn visit(&mut self, depth: u32, visits_at_depth: usize, copies: u32) {
        match depth {
            3 => {
                self.children += 1;
                match self.children {
                    3 => self.third_child = copies,
                    4 => self.fourth_child = copies,
                    _ => {}
                }
            }
            4 if self.children == 4 => self.fourth_child_branches[(visits_at_depth - 1) % 4] = copies,
            5 if self.children == 2 => self.second_child_tail = [self.second_child_tail[1], copies],
            5 if self.children == 4 => {
                // The first two subtrees of the second branch join separate later batches.
                let child = (visits_at_depth - 1) % 16;
                if let Some(offset) = child.checked_sub(4).and_then(|i| self.deferred_subtrees.get_mut(i)) {
                    *offset = copies;
                }
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct RepeatedPair {
    middle: CopyPositions,
    next: CopyPositions,
}

#[derive(Clone, Copy)]
enum TrackedSpan {
    Collecting(usize, usize),
    Middle(usize),
    Next(usize),
}

#[derive(Default)]
struct CollectingGroup {
    spans: [CopyPositions; 3],
    pair: FollowingPair,
}

impl CollectingGroup {
    fn initial_copies(&self) -> u64 {
        u64::from(self.spans[0].total) + u64::from(self.spans[1].total)
    }

    fn fills_early(&self) -> bool {
        self.initial_copies() >= u64::from(COPIES_PER_BATCH)
    }

    fn reaches_final_subtrees(&self) -> bool {
        self.initial_copies() + u64::from(self.pair.second_child_tail[0]) < u64::from(COPIES_PER_BATCH)
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
    collecting: [CollectingGroup; COLLECTING_GROUPS.len()],
    repeated: [RepeatedPair; REPEATED_GROUPS.len()],
    tracked_span: Option<TrackedSpan>,
    has_deeper_level: bool,
    active_deep_draws: usize,
}

impl UnderfilledQuadrants {
    pub(super) fn new(graph_depth: u32) -> Self {
        Self { has_deeper_level: graph_depth > 8, ..Self::default() }
    }

    fn tracked_span(&self, group: usize, span: usize) -> Option<TrackedSpan> {
        if self.visits[0] == 1
            && let Some(slot) = COLLECTING_GROUPS.iter().position(|&g| g == group)
        {
            return Some(TrackedSpan::Collecting(slot, span));
        }
        let pair = REPEATED_GROUPS.iter().position(|&g| g == group)?;
        match (self.visits[0], span) {
            (1, 1) => Some(TrackedSpan::Middle(pair)),
            (2, 0) => Some(TrackedSpan::Next(pair)),
            _ => None,
        }
    }

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
            if depth <= 5
                && let Some(TrackedSpan::Collecting(slot, 2)) = self.tracked_span
            {
                let group = &mut self.collecting[slot];
                group.pair.visit(depth, *visits, group.spans[2].total);
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
        self.tracked_span = self.tracked_span(group, span);
        if depth == 3
            && let Some(TrackedSpan::Collecting(slot, 2)) = self.tracked_span
        {
            let group = &mut self.collecting[slot];
            group.pair.visit(depth, self.visits[3], group.spans[2].total);
        }
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
        if let Some(tracked) = self.tracked_span {
            let positions = match tracked {
                TrackedSpan::Collecting(slot, index) => &mut self.collecting[slot].spans[index],
                TrackedSpan::Middle(slot) => &mut self.repeated[slot].middle,
                TrackedSpan::Next(slot) => &mut self.repeated[slot].next,
            };
            if positions.record(count, retained).is_none() {
                return false;
            }
        }
        self.active_deep_draws += 1;
        self.copies[self.group] = self.copies[self.group].saturating_add(count);
        true
    }

    fn span_start(&self, tree: usize, group: usize, span: usize) -> Option<usize> {
        let &(first, _) = GROUPS[group].get(span)?;
        self.starts.get(tree * RANGES_PER_TREE + first + 1 + first / 20).copied()
    }

    fn copy_range(start: usize, span: &CopyPositions, range: Range<u32>) -> Option<(DrawPosition, DrawPosition)> {
        let position = |offset| {
            let cut = span.at(offset);
            Some(DrawPosition { stamp: start.checked_add(cut.stamp)?, copy: cut.copy })
        };
        Some((position(range.start)?, position(range.end)?))
    }

    fn append_copies(
        &self,
        slot: usize,
        pieces: impl IntoIterator<Item = (usize, Range<u32>)>,
        mut remaining: u32,
        leading: &mut Vec<(DrawPosition, DrawPosition)>,
    ) -> Option<()> {
        for (index, range) in pieces {
            let span = self.collecting[slot].spans.get(index)?;
            let start = self.span_start(0, COLLECTING_GROUPS[slot], index)?;
            let count = remaining.min(range.end.checked_sub(range.start)?);
            if count != 0 {
                leading.push(Self::copy_range(start, span, range.start..range.start + count)?);
            }
            remaining -= count;
            if remaining == 0 {
                return Some(());
            }
        }
        None
    }

    fn append_batch(&self, slot: usize, leading: &mut Vec<(DrawPosition, DrawPosition)>) -> Option<()> {
        let spans = &self.collecting[slot].spans;
        self.append_copies(slot, spans.iter().enumerate().map(|(i, span)| (i, 0..span.total)), COPIES_PER_BATCH, leading)
    }

    fn append_following(&self, leading: &mut Vec<(DrawPosition, DrawPosition)>) -> Option<()> {
        // A cut before the last two depth-five subtrees can interleave the next stream.
        for slot in [0, 2, 1] {
            let group = &self.collecting[slot];
            if group.fills_early() {
                return Some(());
            }
            let late = group.reaches_final_subtrees();
            // At connected depth eight only the third slot submits two consecutive batches in this prefix.
            if late && (self.has_deeper_level || slot == 2) {
                self.append_paired_batches(slot, leading)?;
            } else {
                self.append_batch(slot, leading)?;
            }
            if !late {
                return Some(());
            }
        }
        self.append_repeated_pairs(leading)
    }

    fn append_paired_batches(&self, slot: usize, leading: &mut Vec<(DrawPosition, DrawPosition)>) -> Option<()> {
        let group = &self.collecting[slot];
        let [first, middle, tail] = &group.spans;
        let pair = &group.pair;
        // All three streams skip the third child. The first also skips the first
        // three branches of the fourth child after submitting that child's own draw.
        let gap = if slot == 0 { pair.fourth_child_branches[0]..pair.fourth_child_branches[3] } else { pair.fourth_child..pair.fourth_child };
        let pieces = [(0, 0..first.total), (1, 0..middle.total), (2, 0..pair.third_child), (2, pair.fourth_child..gap.start), (2, gap.end..tail.total)];
        self.append_copies(slot, pieces, 2 * COPIES_PER_BATCH, leading)
    }

    fn append_repeated_pairs(&self, leading: &mut Vec<(DrawPosition, DrawPosition)>) -> Option<()> {
        if !self.has_deeper_level || self.visits[0] < 2 {
            return Some(());
        }
        let first_group = &self.collecting[0];
        let deferred = &first_group.pair.deferred_subtrees;
        let branch_start = self.span_start(0, COLLECTING_GROUPS[0], 2)?;
        let mut runs = Vec::with_capacity(8);
        for (slot, &group) in REPEATED_GROUPS.iter().enumerate() {
            let subtree = LEADING_SUBTREES[slot];
            let pair = self.pairs.get(subtree / 8)?;
            let repeated = &self.repeated[slot];
            // The pair's remainder and two neighboring subtrees precede a pending
            // depth-five subtree. Those pending subtrees arrive in reverse child order.
            let branch = deferred[1 - slot]..deferred[2 - slot];
            let tail = pair.copies.checked_sub(2 * COPIES_PER_BATCH)?;
            let carried = tail.checked_add(repeated.middle.total)?.checked_add(branch.end.checked_sub(branch.start)?)?;
            if carried >= COPIES_PER_BATCH {
                // Either stream filling before the next tree can change their relative order.
                return Some(());
            }
            // The new tree can fill a third batch with copies carried from the first.
            let available = carried.checked_add(repeated.next.total)?;
            let next_copies = repeated.next.total.checked_sub(available % COPIES_PER_BATCH)?;
            let end = *self.starts.get((subtree / 16 + 1) * RANGES_PER_CHILD)?;
            runs.push((pair.cut?, DrawPosition { stamp: end, copy: 0 }));
            runs.push(Self::copy_range(self.span_start(0, group, 1)?, &repeated.middle, 0..repeated.middle.total)?);
            runs.push(Self::copy_range(branch_start, &first_group.spans[2], branch)?);
            runs.push(Self::copy_range(self.span_start(1, group, 0)?, &repeated.next, 0..next_copies)?);
        }
        leading.extend(runs);
        Some(())
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
        let last = &self.collecting[3];
        // Full batches from the last quadrant can precede the three leading pairs.
        // Its last child submits first, then each preceding child precedes its own pair.
        if last.fills_early() {
            self.append_batch(3, &mut leading)?;
        }
        for (slot, subtree) in LEADING_SUBTREES.into_iter().enumerate() {
            if self.collecting[slot].fills_early() {
                self.append_batch(slot, &mut leading)?;
            }
            let pair = self.pairs.get(subtree / 8)?;
            leading.push((position(pair.start), pair.cut?));
        }
        let copies = last.initial_copies() + u64::from(last.spans[2].total);
        if copies / u64::from(COPIES_PER_BATCH) == 1 && !last.fills_early() {
            self.append_batch(3, &mut leading)?;
            if last.reaches_final_subtrees() {
                self.append_following(&mut leading)?;
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
