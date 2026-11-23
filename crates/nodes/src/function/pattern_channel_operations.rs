use super::{catalog, instruction::Instruction, returned_product::float_width, vector::PatternComponent};
use materialcraft_core::Value;

/// A pattern swizzle evaluates a component-wise expression for just the requested channel.
/// Use ordinary scalar instructions so that shared operands, branches and random draws keep
/// the same evaluation order as a scalar expression authored directly.
pub(super) fn project(
    op: &Instruction,
    original: &[Instruction],
    sources: &[[Option<PatternComponent>; 4]],
    generated: &mut Vec<Instruction>,
) -> Option<[Option<PatternComponent>; 4]> {
    let width = float_width(op.type_code);
    if width < 2 {
        return None;
    }
    let kind = |i: usize| original.get(i).map(|input| input.type_code);
    let valid = match (op.definition.id, op.inputs.as_slice()) {
        ("ifelse", &[c, a, b]) => kind(c) == Some(4) && kind(a) == Some(op.type_code) && kind(b) == Some(op.type_code),
        ("mul", &[a, b]) => kind(a) == Some(op.type_code) && kind(b) == Some(op.type_code),
        ("lerp", &[a, b, x]) => kind(a) == Some(op.type_code) && kind(b) == Some(op.type_code) && kind(x) == Some(256),
        _ => false,
    };
    if !valid {
        return None;
    }

    let mut channels = [None; 4];
    for (lane, channel) in channels.iter_mut().take(width).enumerate() {
        let mut inputs = Vec::with_capacity(op.inputs.len());
        for &index in &op.inputs {
            let source = original.get(index)?;
            let input = if source.type_code != op.type_code {
                index
            } else {
                match sources.get(index)?[lane] {
                    Some(PatternComponent::Scalar(register)) => register,
                    Some(PatternComponent::Zero) => {
                        let register = original.len() + generated.len();
                        generated.push(Instruction::new(source.id, catalog::definition("const_float1")?, vec![], Some(Value::Float(0.0)), 256));
                        register
                    }
                    None => {
                        let register = original.len() + generated.len();
                        generated.push(Instruction::new(source.id, catalog::definition("swizzle1")?, vec![index], Some(Value::Int(lane as i32)), 256));
                        register
                    }
                }
            };
            inputs.push(input);
        }
        *channel = Some(PatternComponent::Scalar(original.len() + generated.len()));
        let mut instruction = Instruction::new(op.id, op.definition, inputs, None, 256);
        instruction.literal = op.literal;
        generated.push(instruction);
    }
    Some(channels)
}

#[cfg(test)]
#[path = "pattern_channels.test.rs"]
mod tests;
