use super::pattern_stamp::PatternStamp;
use crate::function::Program;
use materialcraft_core::Value;
use std::collections::{BTreeSet, HashMap};

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

#[derive(Clone, Copy)]
pub(super) enum Storage {
    Full,
    FixedImage,
    Placement,
}

// Each node and depth has immutable properties. A sparse seed page points
// to its changing fields, which occupy contiguous blocks across all depths.
#[derive(Default)]
struct PackedStamps<const FIELDS: usize> {
    templates: HashMap<(u32, u32), Properties>,
    pages: HashMap<(u32, u32, u16), Box<[u32]>>,
    values: Vec<Vec<[f32; FIELDS]>>,
    bytes: usize,
}

impl<const FIELDS: usize> PackedStamps<FIELDS> {
    const PAGE_SIZE: usize = 256;
    const BLOCK_SIZE: usize = 4_096;

    fn address(key: Key) -> ((u32, u32, u16), usize) {
        ((key.node, key.depth, key.seed as u16), (key.seed >> 16) as usize)
    }

    fn get(&self, key: Key) -> Option<(Properties, [f32; FIELDS])> {
        let (page, index) = Self::address(key);
        let at = self.pages.get(&page)?.get(index)?.checked_sub(1)? as usize;
        let fields = *self.values.get(at / Self::BLOCK_SIZE)?.get(at % Self::BLOCK_SIZE)?;
        Some((self.templates.get(&(key.node, key.depth))?.clone(), fields))
    }

    fn insert(&mut self, key: Key, properties: &Properties, fields: [f32; FIELDS], limit: usize) -> bool {
        // Child streams have 24-bit seeds. Wider explicit seeds use the small cache.
        if key.seed >> 24 != 0 {
            return false;
        }
        let (page, index) = Self::address(key);
        let new_block = self.values.last().is_none_or(|block| block.len() == Self::BLOCK_SIZE);
        let growth = usize::from(new_block) * Self::BLOCK_SIZE * size_of::<[f32; FIELDS]>()
            + usize::from(!self.pages.contains_key(&page)) * Self::PAGE_SIZE * size_of::<u32>();
        // Bound both indices and floating-point storage independently of the traversal length.
        if self.bytes.saturating_add(growth) > limit {
            return false;
        }
        let entries = self.pages.entry(page).or_insert_with(|| vec![0; Self::PAGE_SIZE].into_boxed_slice());
        let Some(entry) = entries.get_mut(index) else { return false };
        if *entry != 0 {
            return true;
        }
        self.bytes += growth;
        if new_block {
            self.values.push(Vec::with_capacity(Self::BLOCK_SIZE));
        }
        let base = self.values.len().saturating_sub(1) * Self::BLOCK_SIZE;
        let Some(block) = self.values.last_mut() else { return false };
        let Ok(at) = u32::try_from(base + block.len() + 1) else { return false };
        block.push(fields);
        *entry = at;
        self.templates.entry((key.node, key.depth)).or_insert_with(|| properties.clone());
        true
    }
}

#[derive(Default)]
pub(super) struct StampCache {
    slots: Vec<Option<(Key, Properties)>>,
    lookups: usize,
    images: PackedStamps<5>,
    placements: PackedStamps<4>,
}

impl StampCache {
    // Bound memory independently of the number of nodes, seeds and iterations in a package.
    const MAX_SLOTS: usize = 65_536;

    fn index(&self, key: Key) -> usize {
        let hash = (key.seed ^ key.node.rotate_left(7) ^ key.depth.rotate_left(19)).wrapping_mul(2_654_435_769);
        (hash >> 16) as usize & self.slots.len().saturating_sub(1)
    }

    pub fn get(&mut self, key: Key, storage: Storage) -> Option<Properties> {
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
        match storage {
            Storage::Full => None,
            Storage::FixedImage => {
                let (mut properties, [x, y, width, height, alpha]) = self.images.get(key)?;
                properties.branch_offset = [x, y];
                properties.stamp.size = [width, height];
                properties.stamp.color[3] = alpha;
                Some(properties)
            }
            Storage::Placement => {
                let (mut properties, [x, y, rotation, alpha]) = self.placements.get(key)?;
                properties.stamp.center = [x, y];
                properties.stamp.rotation = rotation;
                properties.stamp.color[3] = alpha;
                Some(properties)
            }
        }
    }

    pub fn insert(&mut self, key: Key, properties: Properties, storage: Storage, streaming: bool) {
        if streaming {
            let stamp = &properties.stamp;
            let stored = match storage {
                Storage::FixedImage if self.lookups > Self::MAX_SLOTS * 2 => {
                    let [x, y] = properties.branch_offset;
                    let [width, height] = stamp.size;
                    self.images.insert(key, &properties, [x, y, width, height, stamp.color[3]], 64 << 20)
                }
                Storage::Placement => {
                    let [x, y] = stamp.center;
                    self.placements.insert(key, &properties, [x, y, stamp.rotation, stamp.color[3]], 96 << 20)
                }
                _ => false,
            };
            if stored {
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
        cache.insert(first, properties.clone(), Storage::FixedImage, true);
        properties.branch_offset = [-0.375, 0.25];
        properties.stamp.size = [0.125, 0.875];
        properties.stamp.color[3] = 0.625;
        cache.insert(second, properties, Storage::FixedImage, true);
        let a = cache.get(first, Storage::FixedImage).unwrap();
        let b = cache.get(second, Storage::FixedImage).unwrap();
        assert_eq!((a.branch_offset, a.stamp.size, a.stamp.color[3]), ([0.125, -0.25], [0.75, 0.25], 0.375));
        assert_eq!((b.branch_offset, b.stamp.size, b.stamp.color[3]), ([-0.375, 0.25], [0.125, 0.875], 0.625));
        assert_eq!((b.stamp.center, b.stamp.rotation, b.stamp.pattern), ([0.25, 0.125], -0.125, 257));
        assert!(cache.get(Key { depth: 8, ..second }, Storage::FixedImage).is_none());
        assert!(cache.get(Key { node: 18, ..second }, Storage::FixedImage).is_none());
        assert!(cache.get(Key { seed: second.seed | (1 << 24), ..second }, Storage::FixedImage).is_none());
    }
}
