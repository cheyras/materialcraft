use super::pattern_stamp::PatternStamp;
use crate::function::Program;
use materialcraft_core::Value;
use std::collections::{BTreeMap, BTreeSet};

/// A terminal draw can reuse its properties across positions when its programs
/// only depend on the traversal seed, depth and inputs fixed for this cook.
pub(super) fn reusable(program: &Program, assigned: &BTreeSet<&str>) -> bool {
    program.instructions.iter().all(|op| {
        let id = op.definition.id;
        if matches!(id, "set" | "while" | "samplecol" | "samplelum") {
            return false;
        }
        if (id.starts_with("get_") || id == "isdefined")
            && let Some(Value::Text(name)) = &op.value
            && (matches!(name.as_str(), "$pos" | "$number") || assigned.contains(name.as_str()))
        {
            return false;
        }
        op.call.as_ref().is_none_or(|call| reusable(&call.program, assigned))
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Key {
    pub node: u32,
    pub depth: u32,
    pub seed: u32,
}

#[derive(Clone)]
pub(super) struct Properties {
    pub branch_offset: [f32; 2],
    /// The center holds the frame offset until the two translations have rounded separately.
    pub stamp: PatternStamp,
}

// Only these five floats can vary for an eligible grayscale image leaf at a fixed depth.
// Keep the original properties as a template so size and position retain their F32 rounding.
struct ImageLeaf {
    node: u32,
    depth: u32,
    template: Properties,
    pages: BTreeMap<u16, Box<[Option<[f32; 5]>]>>,
}

impl ImageLeaf {
    const PAGE_SIZE: usize = 256;
    const MAX_PAGES: usize = 64 * 1024 * 1024 / (Self::PAGE_SIZE * size_of::<Option<[f32; 5]>>());

    fn address(&self, key: Key) -> Option<(u16, usize)> {
        // The child generator produces 24-bit words. Low bits shared by many children
        // select one page. A different leaf, depth or wider seed uses the small cache.
        (key.node == self.node && key.depth == self.depth && key.seed >> 24 == 0).then_some((key.seed as u16, (key.seed >> 16) as usize))
    }

    fn get(&self, key: Key) -> Option<Properties> {
        let (page, index) = self.address(key)?;
        let &[x, y, width, height, alpha] = self.pages.get(&page)?.get(index)?.as_ref()?;
        let mut properties = self.template.clone();
        properties.branch_offset = [x, y];
        properties.stamp.size = [width, height];
        properties.stamp.color[3] = alpha;
        Some(properties)
    }

    fn insert(&mut self, key: Key, properties: &Properties) -> bool {
        let Some((page, index)) = self.address(key) else { return false };
        // Limit the materialized pages even when every visit supplies a distinct random seed.
        if self.pages.len() >= Self::MAX_PAGES && !self.pages.contains_key(&page) {
            return false;
        }
        let entries = self.pages.entry(page).or_insert_with(|| vec![None; Self::PAGE_SIZE].into_boxed_slice());
        let Some(entry) = entries.get_mut(index) else { return false };
        let [x, y] = properties.branch_offset;
        let [width, height] = properties.stamp.size;
        *entry = Some([x, y, width, height, properties.stamp.color[3]]);
        true
    }
}

#[derive(Default)]
pub(super) struct StampCache {
    slots: Vec<Option<(Key, Properties)>>,
    lookups: usize,
    image_leaf: Option<ImageLeaf>,
}

impl StampCache {
    // Bound memory independently of the number of nodes, seeds and iterations in a package.
    const MAX_SLOTS: usize = 65_536;

    fn index(&self, key: Key) -> usize {
        let hash = (key.seed ^ key.node.rotate_left(7) ^ key.depth.rotate_left(19)).wrapping_mul(2_654_435_769);
        (hash >> 16) as usize & self.slots.len().saturating_sub(1)
    }

    pub fn get(&mut self, key: Key) -> Option<Properties> {
        self.lookups = self.lookups.saturating_add(1);
        if self.slots.len() < Self::MAX_SLOTS && self.lookups > self.slots.len() * 2 {
            // A shallow or sparsely connected graph only needs a small table.
            self.slots.resize_with((self.slots.len() * 2).max(16), || None);
        }
        if let Some(Some((stored, value))) = self.slots.get(self.index(key))
            && *stored == key
        {
            return Some(value.clone());
        }
        self.image_leaf.as_ref()?.get(key)
    }

    pub fn insert(&mut self, key: Key, properties: Properties, streamed_image: bool) {
        if streamed_image && self.lookups > Self::MAX_SLOTS * 2 && key.seed >> 24 == 0 {
            let leaf =
                self.image_leaf.get_or_insert_with(|| ImageLeaf { node: key.node, depth: key.depth, template: properties.clone(), pages: BTreeMap::new() });
            if leaf.insert(key, &properties) {
                return;
            }
        }
        let index = self.index(key);
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = Some((key, properties));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_image_cache_distinguishes_seed_bits_and_drawing_depth() {
        let mut cache = StampCache { lookups: StampCache::MAX_SLOTS * 2 + 1, ..Default::default() };
        let first = Key { node: 17, depth: 9, seed: 0x5611 };
        let second = Key { seed: 0x815611, ..first };
        let mut properties = Properties {
            branch_offset: [0.125, -0.25],
            stamp: PatternStamp {
                center: [0.25, 0.125],
                size: [0.75, 0.25],
                rotation: -0.125,
                literal_geometry: false,
                color: [1.0, 1.0, 1.0, 0.375],
                depth: 9,
                pattern: 257,
                variation: 0.0,
                blend: 0,
                filtering: 1,
                premultiplied: false,
            },
        };
        cache.insert(first, properties.clone(), true);
        properties.branch_offset = [-0.375, 0.25];
        properties.stamp.size = [0.125, 0.875];
        properties.stamp.color[3] = 0.625;
        cache.insert(second, properties, true);
        let a = cache.get(first).unwrap();
        let b = cache.get(second).unwrap();
        assert_eq!((a.branch_offset, a.stamp.size, a.stamp.color[3]), ([0.125, -0.25], [0.75, 0.25], 0.375));
        assert_eq!((b.branch_offset, b.stamp.size, b.stamp.color[3]), ([-0.375, 0.25], [0.125, 0.875], 0.625));
        assert_eq!((b.stamp.center, b.stamp.rotation, b.stamp.pattern), ([0.25, 0.125], -0.125, 257));
        assert!(cache.get(Key { depth: 8, ..second }).is_none());
        assert!(cache.get(Key { node: 18, ..second }).is_none());
        assert!(cache.get(Key { seed: second.seed | (1 << 24), ..second }).is_none());
    }
}
