use super::{EvaluationState, Program, instruction::Instruction, product_sum::ProductSum, returned_product::float_width};
use materialcraft_core::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Factors from the current frame, or a snapshot supplied when its caller entered this frame.
#[derive(Clone, Copy)]
pub(super) enum Binding {
    Local { left: usize, right: usize, assignment: Option<usize> },
    Argument(usize),
}

impl Binding {
    pub(super) fn read(self, state: &EvaluationState) -> Option<[Value; 2]> {
        match self {
            Self::Argument(slot) => state.assigned_arguments.get(slot)?.clone(),
            Self::Local { left, right, assignment } => {
                if let Some(index) = assignment {
                    state.registers.get(index)?.as_ref()?;
                }
                Some([state.registers.get(left)?.as_ref()?.clone(), state.registers.get(right)?.as_ref()?.clone()])
            }
        }
    }
}

fn variable(op: &Instruction) -> Option<&str> {
    match &op.value {
        Some(Value::Text(name)) if !name.starts_with('$') => Some(name),
        _ => None,
    }
}

fn product(op: &Instruction) -> Option<(usize, usize)> {
    let &[a, b] = op.inputs.as_slice() else { return None };
    (!op.literal && float_width(op.type_code) > 0 && matches!(op.definition.id, "mul" | "mulscalar")).then_some((a, b))
}

fn unit(op: &Instruction, instructions: &[Instruction]) -> Option<usize> {
    let &[input, scalar] = op.inputs.as_slice() else { return None };
    (op.definition.id == "mulscalar" && instructions.get(scalar).is_some_and(|i| i.definition.id == "const_float1" && i.value == Some(Value::Float(1.0))))
        .then_some(input)
}

#[derive(Default)]
struct Uses {
    reads: BTreeSet<String>,
    writes: BTreeMap<String, usize>,
    observed: BTreeSet<String>,
    rounded: Vec<bool>,
    calls: BTreeMap<usize, Uses>,
}

impl Uses {
    fn analyze(program: &Program) -> Self {
        let mut uses = Self { rounded: vec![false; program.instructions.len()], ..Self::default() };
        if let Some(output) = uses.rounded.get_mut(program.output) {
            *output = true;
        }
        for (index, op) in program.instructions.iter().enumerate().rev() {
            let observed = uses.rounded.get(index).copied().unwrap_or(true);
            let mut mark = |input: usize, rounded: bool| {
                if let Some(flag) = uses.rounded.get_mut(input) {
                    *flag |= rounded;
                }
            };
            if let Some(call) = &op.call {
                let child = Self::analyze(&call.program);
                for (name, register) in call.inputs.iter().zip(&op.inputs) {
                    mark(*register, child.observed.contains(name));
                }
                let free = |name: &String| !call.defaults.contains_key(name);
                uses.reads.extend(child.reads.iter().filter(|name| free(name)).cloned());
                uses.observed.extend(child.observed.iter().filter(|name| free(name)).cloned());
                for (name, count) in &child.writes {
                    if free(name) {
                        *uses.writes.entry(name.clone()).or_default() += count;
                    }
                }
                uses.calls.insert(index, child);
            } else if let Some(input) = unit(op, &program.instructions) {
                mark(input, observed);
            } else {
                match (op.definition.id, op.inputs.as_slice()) {
                    ("add" | "sub", _) => {}
                    ("sequence", &[_, last]) | ("passthrough", &[last]) => mark(last, observed),
                    ("set", &[input]) => {
                        mark(input, observed || program.instructions.get(input).and_then(product).is_none());
                        if let Some(name) = variable(op) {
                            *uses.writes.entry(name.to_owned()).or_default() += 1;
                        }
                    }
                    ("get_float1" | "get_float2" | "get_float3" | "get_float4", []) => {
                        if let Some(name) = variable(op) {
                            uses.reads.insert(name.to_owned());
                            if observed {
                                uses.observed.insert(name.to_owned());
                            }
                        }
                    }
                    _ => {
                        for &input in &op.inputs {
                            mark(input, true);
                        }
                    }
                }
            }
        }
        uses
    }
}

fn has_assignment(program: &Program) -> bool {
    program.instructions.iter().any(|op| {
        (op.definition.id == "set" && op.inputs.first().and_then(|&i| program.instructions.get(i)).and_then(product).is_some())
            || op.call.as_ref().is_some_and(|call| has_assignment(&call.program))
    })
}

pub(super) fn identify(program: &mut Program) {
    if program.instructions.iter().any(|op| op.call.is_some()) && has_assignment(program) {
        let uses = Uses::analyze(program);
        install(program, &uses, &uses.observed, BTreeMap::new());
    }
}

fn install(program: &mut Program, uses: &Uses, observed: &BTreeSet<String>, mut names: BTreeMap<String, Binding>) {
    let instructions = &program.instructions;
    // An outer binding must remain unchanged throughout the call. Repeated assignments in a loop can replace cached factors.
    names.retain(|name, _| !uses.writes.contains_key(name));
    let mut sources = vec![None; instructions.len()];
    let mut products_written = BTreeMap::<usize, usize>::new();
    let straight = instructions.iter().all(|op| !matches!(op.definition.id, "ifelse" | "while"));
    if !instructions.iter().any(|op| op.definition.id == "while") {
        for op in instructions {
            if let ("set", &[input]) = (op.definition.id, op.inputs.as_slice()) {
                *products_written.entry(input).or_default() += 1;
            }
        }
        for (index, setter) in instructions.iter().enumerate() {
            if let ("set", &[input], Some(name)) = (setter.definition.id, setter.inputs.as_slice(), variable(setter))
                && uses.writes.get(name) == Some(&1)
                && products_written.get(&input) == Some(&1)
                && !observed.contains(name)
                && !uses.observed.contains(name)
                && uses.rounded.get(input) == Some(&false)
                && let Some((left, right)) = instructions.get(input).and_then(product)
            {
                names.insert(name.to_owned(), Binding::Local { left, right, assignment: Some(index) });
                if let Some(slot) = sources.get_mut(input) {
                    *slot = Some(Binding::Local { left, right, assignment: None });
                }
            }
        }
    }
    for index in 0..program.instructions.len() {
        let (previous, current) = program.instructions.split_at_mut(index);
        let Some(op) = current.first_mut() else { continue };
        let source = |input: usize| sources.get(input).copied().flatten();
        if let Some(call) = &mut op.call {
            let Some(child) = uses.calls.get(&index) else { continue };
            let mut imported = BTreeMap::new();
            for (name, binding) in &names {
                if !call.defaults.contains_key(name) && child.reads.contains(name) && !child.observed.contains(name) && !child.writes.contains_key(name) {
                    imported.insert(name.clone(), Binding::Argument(call.assigned_products.len()));
                    call.assigned_products.push(*binding);
                }
            }
            for (name, register) in call.inputs.iter().zip(&op.inputs) {
                if child.reads.contains(name)
                    && !child.observed.contains(name)
                    && !child.writes.contains_key(name)
                    && let Some(binding) = source(*register)
                {
                    imported.insert(name.clone(), Binding::Argument(call.assigned_products.len()));
                    call.assigned_products.push(binding);
                }
            }
            install(&mut call.program, child, observed, imported);
            continue;
        }
        let forwarded = match (op.definition.id, op.inputs.as_slice()) {
            ("get_float1" | "get_float2" | "get_float3" | "get_float4", []) => variable(op).and_then(|name| names.get(name)).copied().filter(|binding| {
                // In straight-line code an earlier getter can retain the value from before the Set.
                !matches!(binding, Binding::Local { assignment: Some(set), .. } if !straight || *set >= index)
            }),
            ("passthrough", &[i]) => source(i),
            _ => unit(op, previous).and_then(source),
        };
        if let Some(binding) = forwarded
            && let Some(slot) = sources.get_mut(index)
        {
            *slot = Some(binding);
        }
        if let ("add" | "sub", &[left, right]) = (op.definition.id, op.inputs.as_slice()) {
            let a = sources.get(left).copied().flatten();
            let b = sources.get(right).copied().flatten();
            let Some(binding) = a.xor(b) else { continue };
            let operand = if a.is_some() { left } else { right };
            let other = if a.is_some() { right } else { left };
            if op.pixel_product_sum.is_none()
                || previous.get(operand).and_then(|op| unit(op, previous)).is_some() && previous.get(other).and_then(product).is_none()
            {
                op.pixel_product_sum = Some(ProductSum::assigned(&op.inputs, b.is_some(), binding));
            }
        }
    }
}
