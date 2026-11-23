use super::value_type::{dimensions, floats, integer_value, integers, intermediate_float_value as float_value};
use materialcraft_core::Value;

#[derive(Clone, Copy)]
pub(super) enum PatternComponent {
    Scalar(usize),
    Zero,
}

/// A pattern that selects one packed channel visits only the expressions that supply it.
pub(super) fn identify_pattern_components(program: &mut super::Program) {
    let instructions = &program.instructions;
    if !instructions.iter().any(|op| op.definition.id == "swizzle1" || op.call.is_some())
        || !instructions.iter().any(|op| matches!(op.definition.id, "rand" | "set") || op.call.is_some())
    {
        return;
    }
    // A call can read state written elsewhere in this graph, without an argument wire.
    // Keep the original evaluation order when those dependencies are unknown.
    if instructions.iter().any(|op| op.definition.id == "set")
        && instructions.iter().any(|op| op.call.as_ref().is_some_and(|call| call.pattern_arguments.is_none()))
    {
        return;
    }
    let mut sources = Vec::with_capacity(instructions.len());
    let mut projections = Vec::new();
    let mut generated = Vec::new();
    let mut ranges = Vec::with_capacity(instructions.len());
    for (index, op) in instructions.iter().enumerate() {
        let start = instructions.len() + generated.len();
        let mut lanes = [None; 4];
        if let Some(forwarded) = super::returned_product::forwarded(op, instructions, &sources, Some(PatternComponent::Zero)) {
            lanes = forwarded;
            if op.definition.id == "swizzle1"
                && let Some(source) = lanes[0]
            {
                program.pattern_components.insert(index, source);
            }
        } else if let Some(projected) = super::pattern_channel_operations::project(op, instructions, &sources, &mut generated) {
            lanes = projected;
        } else if let Some(call) = &op.call {
            if let Some(arguments) = &call.pattern_arguments {
                lanes = arguments.channels.map(|register| register.map(PatternComponent::Scalar));
            } else if let Some(projection) = call.packed_projection(&op.inputs, &sources) {
                lanes = projection.channels.map(|register| register.map(PatternComponent::Scalar));
                projections.push((index, projection));
            }
        }
        if op.type_code == 256 && lanes[0].is_none() {
            lanes[0] = Some(PatternComponent::Scalar(index));
        }
        sources.push(lanes);
        ranges.push(start..instructions.len() + generated.len());
    }
    for (index, projection) in projections {
        if let Some(call) = program.instructions.get_mut(index).and_then(|op| op.call.as_mut()) {
            call.pattern_arguments = Some(projection);
        }
    }
    if !generated.is_empty() {
        for op in program.instructions.iter_mut().filter(|op| op.definition.id == "while") {
            // Each repetition also invalidates the scalar channels derived from its condition or body.
            let repeated: Vec<_> = op.control_registers.iter().filter_map(|&i| ranges.get(i)).flat_map(|r| r.clone()).collect();
            op.control_registers.extend(repeated);
        }
        program.instructions.extend(generated);
    }
}

pub(crate) fn evaluate(op: &str, args: &[Value], setting: Option<&Value>) -> Result<Value, String> {
    let first = args.first().ok_or("missing vector input")?;
    let count = op.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(1) as usize;
    if !(1..=4).contains(&count) {
        return Err("invalid vector size".into());
    }

    let integer = op.starts_with('i') || op.starts_with("toint");
    let mut result_float = [0.0; 4];
    let mut result_int = [0; 4];
    if op.contains("swizzle") {
        let indices = setting.map(integers).transpose()?.unwrap_or([0; 4]);
        let floats = if integer { [0.0; 4] } else { floats(first)? };
        let integers = if integer { integers(first)? } else { [0; 4] };
        for (lane, index) in indices.iter().take(count).enumerate() {
            let index = usize::try_from(*index).ok().filter(|index| *index < 4).ok_or("swizzle component is outside its input")?;
            if integer {
                if let Some(result) = result_int.get_mut(lane) {
                    // Swizzles read zero in undeclared lanes without scalar broadcast
                    *result = if index < dimensions(first) { *integers.get(index).ok_or("invalid swizzle")? } else { 0 };
                }
            } else if let Some(result) = result_float.get_mut(lane) {
                *result = if index < dimensions(first) { *floats.get(index).ok_or("invalid swizzle")? } else { 0.0 };
            }
        }
    } else if op.contains("vector") {
        let last = args.get(1).ok_or("missing final vector component")?;
        if dimensions(first) + dimensions(last) != count {
            return Err(format!("vector{count} requires inputs totaling {count} components"));
        }

        if integer {
            let left = integers(first)?;
            let right = integers(last)?;
            for (result, value) in result_int.iter_mut().zip(left.into_iter().take(dimensions(first)).chain(right.into_iter().take(dimensions(last)))) {
                *result = value;
            }
        } else {
            let left = floats(first)?;
            let right = floats(last)?;
            for (result, value) in result_float.iter_mut().zip(left.into_iter().take(dimensions(first)).chain(right.into_iter().take(dimensions(last)))) {
                *result = value;
            }
        }
    } else {
        if dimensions(first) != count {
            return Err("casts preserve the number of components".into());
        }

        if integer {
            let values = floats(first)?;
            for (result, value) in result_int.iter_mut().zip(values).take(count) {
                if !value.is_finite() || !(-2147483648.0..2147483648.0).contains(&value) {
                    return Err("float is outside the integer range".into());
                }

                // Round down because truncation shifts negative texture and lattice indices
                *result = value.floor() as i32;
            }
        } else {
            for (result, value) in result_float.iter_mut().zip(integers(first)?).take(count) {
                *result = value as f32;
            }
        }
    }

    if integer {
        integer_value(result_int.get(..count).ok_or("invalid integer dimensions")?)
    } else {
        float_value(result_float.get(..count).ok_or("invalid float dimensions")?)
    }
}
