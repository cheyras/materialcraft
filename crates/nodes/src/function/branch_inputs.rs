use super::{Program, function_call::FunctionCall, vector::PatternComponent};
use materialcraft_core::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct BranchInputs {
    shared: Vec<usize>,
    true_count: usize,
}

impl BranchInputs {
    // The true arm occurs first. A skipped arm still evaluates effects shared with another consumer.
    pub(super) fn skipped(&self, take_true: bool) -> &[usize] {
        let (through_true, through_false_only) = self.shared.split_at(self.true_count);
        if take_true { through_false_only } else { through_true }
    }
}

/// Schedule shared writes and draws from a skipped arm. Private expressions stay lazy.
pub(super) fn identify(program: &mut Program) {
    let instructions = &program.instructions;
    if !instructions.iter().any(|op| op.definition.id == "ifelse")
        || !instructions.iter().any(|op| matches!(op.definition.id, "rand" | "set"))
        || instructions.iter().any(|op| op.definition.id == "while" || op.call.as_ref().is_some_and(|call| call.exports))
    {
        return;
    }

    let written: BTreeSet<_> = instructions
        .iter()
        .filter_map(|op| match (&op.value, op.definition.id) {
            (Some(Value::Text(name)), "set") => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // A range depending on an assignment or image sample must stay at its original read.
    let mut independent_range = Vec::with_capacity(instructions.len());
    for op in instructions {
        let fixed_input = op.definition.id == "get_float1" && matches!(&op.value, Some(Value::Text(name)) if !written.contains(name.as_str()));
        independent_range
            .push(op.type_code == 256 && (op.literal || fixed_input || (op.definition.id == "rand" && op.inputs.iter().all(|&i| independent_range[i]))));
    }
    // A called graph can observe the caller's assignments without listing them among its inputs.
    let has_calls_and_writes = !written.is_empty() && instructions.iter().any(|op| op.call.is_some());
    if !has_calls_and_writes {
        program.processor_branches = shared_inputs(program, &independent_range, None);
    }
    if !has_calls_and_writes || instructions.iter().filter_map(|op| op.call.as_ref()).all(FunctionCall::depends_only_on_arguments) {
        program.pattern_branches = shared_inputs(program, &independent_range, Some(&program.pattern_components));
    }
}

fn shared_inputs(program: &Program, ranges: &[bool], components: Option<&BTreeMap<usize, PatternComponent>>) -> BTreeMap<usize, BranchInputs> {
    let instructions = &program.instructions;
    let mut paths = vec![0_u8; instructions.len()];
    let mut stack = Vec::new();
    let mut effects = Vec::new();
    let mut plans = BTreeMap::new();
    for (branch, op) in instructions.iter().enumerate().filter(|(_, op)| op.definition.id == "ifelse") {
        let &[_, yes, no] = op.inputs.as_slice() else { continue };
        paths.fill(0);
        effects.clear();
        // Record the order of first use through each arm.
        for (root, path) in [(yes, 1), (no, 2), (program.output, 4)] {
            if path == 4 && effects.is_empty() {
                break;
            }
            stack.push(root);
            while let Some(index) = stack.pop() {
                if index == branch || paths[index] & path != 0 {
                    continue;
                }
                let node = &instructions[index];
                let shared_write = op.control_registers.binary_search(&index).is_ok();
                let draw = node.definition.id == "rand" && ranges[index] && !node.inputs.is_empty();
                if paths[index] == 0 && path != 4 && (shared_write || draw) {
                    effects.push(index);
                }
                paths[index] |= path;
                // Evaluating a shared expression performs any conditional writes within it in their own scope.
                if shared_write {
                    continue;
                }
                match components.and_then(|components| components.get(&index)) {
                    Some(PatternComponent::Scalar(source)) => stack.push(*source),
                    Some(PatternComponent::Zero) => {}
                    None => {
                        let inputs = node
                            .call
                            .as_ref()
                            .and_then(|call| call.pattern_arguments.as_ref())
                            .filter(|plan| components.is_some() && matches!(plan.target, super::function_call::PatternTarget::ReturnedChannels))
                            .map_or(node.inputs.as_slice(), |plan| plan.registers.as_slice());
                        stack.extend(inputs.iter().rev().copied());
                    }
                }
            }
        }
        let shared: Vec<_> = effects.iter().copied().filter(|&i| paths[i].count_ones() > 1).collect();
        if !shared.is_empty() {
            let true_count = shared.iter().take_while(|&&i| paths[i] & 1 != 0).count();
            plans.insert(branch, BranchInputs { shared, true_count });
        }
    }
    plans
}

#[cfg(test)]
#[path = "branch_inputs.test.rs"]
mod tests;
