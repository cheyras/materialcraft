use super::pattern_stamp::PatternStamp;
use crate::function::Program;
use materialcraft_core::Value;
use std::collections::BTreeSet;

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

#[derive(Default)]
pub(super) struct StampCache {
    slots: Vec<Option<(Key, Properties)>>,
    lookups: usize,
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
        let (stored, value) = self.slots.get(self.index(key))?.as_ref()?;
        (*stored == key).then(|| value.clone())
    }

    pub fn insert(&mut self, key: Key, properties: Properties) {
        let index = self.index(key);
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = Some((key, properties));
        }
    }
}
