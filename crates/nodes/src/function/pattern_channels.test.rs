use crate::function::{EvaluationContext, EvaluationState, FunctionScope, compile};
use materialcraft_core::{FunctionGraph, Value};
use std::collections::BTreeMap;

fn random_color(graph: &mut FunctionGraph, one: u32) -> u32 {
    let channels = [(); 4].map(|()| graph.add("rand", &[one], None));
    let lower = graph.add("vector2", &channels[..2], None);
    let upper = graph.add("vector2", &channels[2..], None);
    graph.add("vector4", &[lower, upper], None)
}

#[test]
fn selecting_a_computed_pattern_channel_preserves_the_next_random_draw() {
    for operation in ["ifelse", "mul", "lerp"] {
        for lane in [0, 3] {
            let mut graph = FunctionGraph::default();
            let one = graph.add_constant(Value::Float(1.0));
            let zero = graph.add_constant(Value::Float(0.0));
            let left = random_color(&mut graph, one);
            let right = random_color(&mut graph, one);
            let condition = graph.add("get_bool", &[], Some(Value::Text("choice".into())));
            let selected = match operation {
                "ifelse" => graph.add(operation, &[condition, left, right], None),
                "lerp" => {
                    let factor = graph.add_constant(Value::Float(0.375));
                    graph.add(operation, &[left, right, factor], None)
                }
                _ => graph.add(operation, &[left, right], None),
            };
            let channel = graph.add("swizzle1", &[selected], Some(Value::Int(lane)));
            let observed = graph.add("mul", &[channel, zero], None);
            let next = graph.add("rand", &[one], None);
            graph.output = Some(graph.add("add", &[observed, next], None));

            for choice in [true, false] {
                let variables = BTreeMap::from([("choice".into(), Value::Bool(choice))]);
                let context = EvaluationContext {
                    scope: FunctionScope::FxMap,
                    position: [0.5; 2],
                    size: [16.0; 2],
                    images: &[],
                    image_formats: &[],
                    tile: true,
                    variables: &variables,
                };
                for (seed, after_one, after_two) in [(0, 0x3f08_683c, 0x3f41_3194), (17, 0x3f4e_0964, 0x3ec2_8a10)] {
                    let graph = compile(&graph, seed).unwrap();
                    let expected = f32::from_bits(if operation == "ifelse" { after_one } else { after_two });
                    assert_eq!(graph.evaluate(&context, &mut EvaluationState::default()).unwrap(), Value::Float(expected), "{operation} channel {lane}");
                }
            }
        }
    }
}
