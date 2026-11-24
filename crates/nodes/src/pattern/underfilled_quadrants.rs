use materialcraft_core::{PatternNode, PatternNodeKind};

use super::{
    footprint::SampleGrid,
    pattern_stamp::{DrawRun, PatternStamp},
    raster::MIN_STAMP_SIZE,
    tile_runs::COPIES_PER_BATCH,
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
const RANGES_PER_CHILD: usize = 1 + 4 * 5;
const RANGES_PER_TREE: usize = 4 * RANGES_PER_CHILD;

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
        if !parent {
            self.occupied.push(false);
        }
        let Some(group) = GROUPS.iter().position(|spans| spans.iter().any(|&(lo, hi)| (lo..hi).contains(&segment))) else { return false };
        self.group = group;
        true
    }

    pub(super) fn draw(&mut self, stamp: &PatternStamp, weights: &[f32], grid: Option<&SampleGrid>) -> bool {
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
        let Some(count) = grid.and_then(|grid| grid.copy_count(stamp)) else { return false };
        if stamp.depth == 1 {
            return *self.first_shape.get_or_insert(stamp.pattern) == stamp.pattern && self.deeper_shape != Some(stamp.pattern);
        }
        if *self.deeper_shape.get_or_insert(stamp.pattern) != stamp.pattern || self.first_shape == Some(stamp.pattern) {
            return false;
        }
        if stamp.depth >= 3 {
            let Some(occupied) = self.occupied.last_mut() else { return false };
            *occupied = true;
        }
        self.copies[self.group] += count;
        // A full batch can split the subtree and move later submissions into another group.
        self.copies[self.group] < COPIES_PER_BATCH
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
        let mut runs = Vec::new();
        let mut range = |tree: usize, first: usize, end: usize| {
            let start = *self.starts.get(tree * RANGES_PER_TREE + first)?;
            let stop = *self.starts.get(tree * RANGES_PER_TREE + end)?;
            let slice = stamps.get(start..stop)?;
            if !slice.is_empty() {
                runs.push(DrawRun::whole(slice));
            }
            Some(())
        };
        for spans in GROUPS {
            for tree in 0..trees {
                for &(first, end) in spans {
                    // Insert the first-level marker at the start of each twenty-range child.
                    let child_marker = 1 + first / 20;
                    range(tree, child_marker + first, child_marker + end)?;
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
