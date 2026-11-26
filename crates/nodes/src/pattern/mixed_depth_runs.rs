use std::borrow::Cow;

use materialcraft_core::{PatternNode, PatternNodeKind};

use super::{
    pattern_stamp::{DrawPosition, DrawRun, PatternStamp},
    raster::MIN_STAMP_SIZE,
    tile_runs::{FOUR_CHILDREN, TileCopies, UNIT_WINDOW},
};
use crate::CookContext;

// Each pair is the exclusive end in depth-seven parent visits and its compositing position.
// The boundaries come from overlapping original squares. A prefix can end inside a grandchild.
const TWO_CHILDREN: &[(usize, usize)] = &[
    (1308, 8),
    (1536, 15),
    (1792, 14),
    (2048, 16),
    (3868, 2),
    (4096, 25),
    (5404, 4),
    (5632, 9),
    (5888, 18),
    (6144, 10),
    (7964, 0),
    (8192, 23),
    (9500, 6),
    (9728, 12),
    (9984, 11),
    (10240, 13),
    (12060, 1),
    (12288, 21),
    (12800, 20),
    (13312, 3),
    (13824, 24),
    (14336, 5),
    (14848, 22),
    (15360, 7),
    (15872, 17),
    (16384, 19),
];

// With four children and four grandchildren, the independent subtree boundaries
// are 64 parents apart. Runs ending between them fill one or more physical batches.
const FOUR_CHILDREN_AND_GRANDCHILDREN: &[(usize, usize)] = &[
    (64, 15),
    (192, 32),
    (1956, 16),
    (2048, 33),
    (3868, 6),
    (4096, 17),
    (4160, 9),
    (4288, 38),
    (6052, 10),
    (6144, 28),
    (7964, 2),
    (8192, 19),
    (8256, 12),
    (8384, 29),
    (10148, 13),
    (10240, 31),
    (12060, 4),
    (12288, 25),
    (12352, 23),
    (12416, 30),
    (12480, 43),
    (12684, 24),
    (12800, 39),
    (13060, 1),
    (13312, 8),
    (13376, 20),
    (13440, 34),
    (13504, 45),
    (13732, 21),
    (13824, 42),
    (14084, 3),
    (14336, 11),
    (14400, 26),
    (14464, 35),
    (14528, 40),
    (14756, 27),
    (14848, 41),
    (15108, 5),
    (15360, 14),
    (15620, 7),
    (15680, 36),
    (15712, 18),
    (15808, 44),
    (15872, 37),
    (16132, 0),
    (16384, 22),
];

#[derive(Default)]
pub(super) struct MixedDepthRuns {
    layout: &'static [(usize, usize)],
    starts: Vec<DrawPosition>,
    next_boundary: usize,
    parents: usize,
    saw_root: bool,
    children_per_parent: usize,
    draws_per_parent: usize,
    deepest: u32,
    draws: usize,
    shape: Option<i32>,
    copies: Option<TileCopies>,
}

impl MixedDepthRuns {
    pub(super) fn new(context: &CookContext, deepest: u32) -> Option<Self> {
        if context.b("cull_outside") {
            return None;
        }
        let axes = context.tiling_axes();
        // A common render and tiling window determines the repeated copies along its edges.
        let window = if matches!(deepest, 8 | 10) && axes == [true; 2] { context.v4("render_region") } else { UNIT_WINDOW };
        let copies = if axes != [false; 2] || deepest == 10 { Some(TileCopies::new(context, deepest, window)?) } else { None };
        Some(Self { copies, deepest, ..Self::default() })
    }

    pub(super) fn visit(&mut self, node: &PatternNode, depth: u32, retained: usize) -> bool {
        if node.kind != PatternNodeKind::Quadrant {
            return true;
        }
        // Empty outer repetitions emit no quadrants. A second actual tree has a different draw order.
        if depth == 0 {
            if self.saw_root {
                return false;
            }
            self.saw_root = true;
        }
        if depth != 7 {
            return match depth {
                0..=6 => node.children.iter().flatten().count() == 4,
                8 if self.deepest == 9 => node.children.iter().all(Option::is_some),
                // Connected child routes can terminate without visiting another quadrant.
                8 if self.deepest == 10 => true,
                9.. if self.deepest == 10 => false,
                depth if depth == self.deepest => node.children.iter().all(Option::is_none),
                _ => false,
            };
        }
        let children = node.children.iter().flatten().count();
        if self.children_per_parent == 0 {
            (self.layout, self.draws_per_parent) = match (self.deepest, children) {
                (8, 2) => (TWO_CHILDREN, 3),
                (8, 4) => (FOUR_CHILDREN, 5),
                (9, 4) => (FOUR_CHILDREN_AND_GRANDCHILDREN, 21),
                (10, 4) => (&[][..], 5),
                _ => return false,
            };
            self.children_per_parent = children;
            self.starts.reserve_exact(self.layout.len() + 1);
        }
        if children != self.children_per_parent {
            return false;
        }
        if !self.layout.is_empty() && self.parents == self.next_boundary {
            let Some(&(end, _)) = self.layout.get(self.starts.len()) else { return false };
            self.starts.push(DrawPosition { stamp: retained, copy: 0 });
            self.next_boundary = end;
        }
        self.parents += 1;
        true
    }

    pub(super) fn draw(&mut self, stamp: &PatternStamp, opacities: &[f32], retained: bool) -> bool {
        let Some(weight) = opacities.get(stamp.depth as usize) else { return false };
        // Turning off unrelated draws changes the order even where those draws never overlap the readout.
        if !(7..=self.deepest).contains(&stamp.depth)
            || stamp.blend != 0
            || !(2..=16).contains(&stamp.pattern)
            || *self.shape.get_or_insert(stamp.pattern) != stamp.pattern
            || !(stamp.color[3] * weight).is_normal()
            || stamp.size.iter().any(|axis| axis.abs() < MIN_STAMP_SIZE)
        {
            return false;
        }
        self.draws += 1;
        self.copies.as_mut().is_none_or(|copies| copies.record(stamp, retained, self.draws, self.draws_per_parent).is_some())
    }

    pub(super) fn finish(mut self, stamps: &[PatternStamp]) -> Option<Vec<DrawRun<'_>>> {
        if self.parents != 16384 || self.draws != self.parents * self.draws_per_parent || self.starts.len() != self.layout.len() {
            return None;
        }
        if self.deepest == 10 {
            return self.copies?.inactive_child_runs(stamps);
        }
        self.starts.push(DrawPosition { stamp: stamps.len(), copy: 0 });
        let mut layout = Cow::Borrowed(self.layout);
        if let Some(copies) = self.copies {
            if self.children_per_parent == 2 {
                return copies.two_child_runs(stamps);
            }
            layout = copies.layout(self.layout)?;
            self.starts = copies.boundaries(&layout, self.draws_per_parent)?;
        }
        let mut groups = vec![DrawRun::whole(&[]); layout.len()];
        for ((_, rank), bounds) in layout.iter().zip(self.starts.windows(2)) {
            *groups.get_mut(*rank)? = DrawRun::between(stamps, bounds[0], bounds[1])?;
        }
        Some(groups)
    }
}
