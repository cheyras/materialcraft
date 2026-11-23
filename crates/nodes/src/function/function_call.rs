use super::{FunctionDefinition, Program};
use materialcraft_core::Value;
use std::collections::BTreeMap;

pub(crate) const DEFINITION: FunctionDefinition = FunctionDefinition { id: "instance", label: "Function instance", inputs: &[], category: "Graph" };

pub(crate) struct FunctionCall {
    pub program: Program,
    pub inputs: Vec<String>,
    pub defaults: BTreeMap<String, Value>,
    /// Parent registers for an argument used only in scalar sums within the call.
    pub(super) argument_factors: Option<[super::reciprocal::Factor; 2]>,
    pub(super) assigned_products: Vec<super::assigned_product::Binding>,
    pub(super) reciprocals: Option<Box<super::reciprocal::CallFactors>>,
    pub(super) pattern_arguments: Option<PatternArguments>,
    /// The called graph sets variables, directly or in a graph it calls; the caller reads them after the call
    pub exports: bool,
}

/// First uses of scalar arguments in a straight-line pattern function, including directly returned channels.
pub(super) struct PatternArguments {
    pub registers: Vec<usize>,
    pub target: PatternTarget,
    pub channels: [Option<usize>; 4],
}

pub(super) enum PatternTarget {
    Arguments(Vec<usize>),
    ReturnedChannels,
}

impl PatternArguments {
    pub(super) fn identify(program: &Program, names: &[String], defaults: &BTreeMap<String, Value>, registers: &[usize]) -> Option<Self> {
        if names.is_empty() || !defaults.values().all(|v| matches!(v, Value::Float(_))) {
            return None;
        }
        let slots_by_name: BTreeMap<_, _> = names.iter().enumerate().map(|(slot, name)| (name.as_str(), slot)).collect();
        let mut seen = vec![false; names.len()];
        let mut slots = Vec::new();
        let mut origins = Vec::with_capacity(program.instructions.len());
        for op in &program.instructions {
            let mut lanes = [None; 4];
            if let ("get_float1", Some(Value::Text(name))) = (op.definition.id, &op.value) {
                if name.starts_with('$') {
                    return None;
                }
                defaults.get(name)?;
                if let Some(&slot) = slots_by_name.get(name.as_str()) {
                    lanes[0] = Some(slot);
                    if !std::mem::replace(seen.get_mut(slot)?, true) {
                        slots.push(slot);
                    }
                }
            } else if let Some(forwarded) = super::returned_product::forwarded(op, &program.instructions, &origins, None) {
                lanes = forwarded;
            } else if !op.definition.id.starts_with("const_") && (!op.depends_only_on_operands() || matches!(op.definition.id, "ifelse" | "sequence")) {
                // Random draws, assignments and control flow can interleave with the caller's inputs.
                return None;
            }
            origins.push(lanes);
        }
        let selected = origins.get(program.output)?;
        let width = super::returned_product::float_width(program.output_type());
        if width > 0 && selected[..width].iter().all(Option::is_some) {
            slots.clear();
            seen.fill(false);
            for &slot in selected[..width].iter().flatten() {
                if !std::mem::replace(seen.get_mut(slot)?, true) {
                    slots.push(slot);
                }
            }
        }
        Some(Self {
            registers: slots.iter().map(|slot| registers.get(*slot).copied()).collect::<Option<_>>()?,
            target: PatternTarget::Arguments(slots),
            channels: selected.map(|slot| slot.and_then(|i| registers.get(i)).copied()),
        })
    }
}

impl FunctionCall {
    /// A function made entirely of argument reads, vector packs and swizzles can return
    /// caller components directly. Sampling one of its return channels then stays lazy.
    pub(super) fn packed_projection(&self, registers: &[usize], sources: &[[Option<super::vector::PatternComponent>; 4]]) -> Option<PatternArguments> {
        if !self.defaults.values().any(|value| super::returned_product::float_width(super::value_type::type_code(value)) > 1) {
            return None;
        }
        let mut origins = Vec::with_capacity(self.program.instructions.len());
        for op in &self.program.instructions {
            let lanes = match (op.definition.id, &op.value) {
                ("get_float1" | "get_float2" | "get_float3" | "get_float4", Some(Value::Text(name))) => {
                    if name.starts_with('$') || super::value_type::type_code(self.defaults.get(name)?) != op.type_code {
                        return None;
                    }
                    let slot = self.inputs.iter().position(|input| input == name)?;
                    sources.get(*registers.get(slot)?)?.map(|component| match component {
                        Some(super::vector::PatternComponent::Scalar(register)) => Some(register),
                        _ => None,
                    })
                }
                _ => super::returned_product::forwarded(op, &self.program.instructions, &origins, None)?,
            };
            origins.push(lanes);
        }
        let channels = *origins.get(self.program.output)?;
        let width = super::returned_product::float_width(self.program.output_type());
        if width == 0 {
            return None;
        }
        Some(PatternArguments { registers: channels.get(..width)?.iter().copied().collect::<Option<_>>()?, target: PatternTarget::ReturnedChannels, channels })
    }
}

pub(super) fn pack_scalars(values: &[Value]) -> Result<Value, String> {
    let mut channels = [0.0; 4];
    let channels = channels.get_mut(..values.len()).ok_or("too many color components")?;
    for (channel, value) in channels.iter_mut().zip(values) {
        let Value::Float(component) = value else { return Err("expected a scalar color component".into()) };
        *channel = *component;
    }
    super::value_type::intermediate_float_value(channels)
}

/// Wire presence is local to a call. Free variables can be defined by an executed Set in a caller.
pub(super) struct CallBindings<'a> {
    pub call: &'a FunctionCall,
    pub caller_assignments: &'a BTreeMap<String, Value>,
    pub parent: Option<&'a CallBindings<'a>>,
}

impl CallBindings<'_> {
    pub(super) fn is_defined(&self, name: &str, assignments: &BTreeMap<String, Value>) -> bool {
        if self.call.defaults.contains_key(name) {
            return self.call.inputs.iter().any(|input| input == name);
        }
        if assignments.contains_key(name) {
            return true;
        }
        let mut frame = Some(self);
        while let Some(current) = frame {
            if current.caller_assignments.contains_key(name) {
                return true;
            }
            frame = current.parent;
        }
        false
    }
}
