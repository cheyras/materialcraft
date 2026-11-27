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

#[test]
fn assigning_a_computed_pattern_vector_draws_the_operands_of_each_channel_together() {
    let variables = BTreeMap::new();
    let context = EvaluationContext {
        scope: FunctionScope::FxMap,
        position: [0.5; 2],
        size: [16.0; 2],
        images: &[],
        image_formats: &[],
        tile: true,
        variables: &variables,
    };
    // Read each channel back from the assigned vector at three measured stamp seeds.
    for (operation, expected) in [
        ("product", [[0x3c41a72f, 0x3e1491a6], [0x3e74d565, 0x3d31859f], [0x3ed198e7, 0x3e1fd4d9]]),
        ("sum", [[0x3e7ec4c0, 0x3f432dc0], [0x3f918972, 0x3f66fe54], [0x3fb19ce2, 0x3f4a5a34]]),
        ("lerp", [[0x3e0edcac, 0x3ec13990], [0x3efdf41e, 0x3f0d1cf2], [0x3f42c8d5, 0x3ecbb690]]),
        ("scalar", [[0x3c41a72f, 0x3cbfad3c], [0x3e74d565, 0x3f3b1f1e], [0x3ed198e7, 0x3e30d4b5]]),
        ("multiply_shared", [[0x3c41a72f, 0x3d8a8af7], [0x3e74d565, 0x3e72cfb6], [0x3ed198e7, 0x3ec7ebed]]),
    ] {
        let mut graph = FunctionGraph::default();
        let unit = graph.add_constant(Value::Float(1.0));
        let [a, b, c, d] = std::array::from_fn(|_| graph.add("rand", &[unit], None));
        let left = graph.add("vector2", &[a, b], None);
        let right = graph.add("vector2", &[c, if operation == "multiply_shared" { a } else { d }], None);
        let result = match operation {
            "lerp" => {
                let factor = graph.add_constant(Value::Float(0.375));
                graph.add("lerp", &[left, right, factor], None)
            }
            "scalar" => graph.add("mulscalar", &[left, c], None),
            "sum" => graph.add("add", &[left, right], None),
            _ => graph.add("mul", &[left, right], None),
        };
        let name = Some(Value::Text("recorded".into()));
        let assigned = graph.add("set", &[result], name.clone());
        let read = graph.add("get_float2", &[], name);
        for lane in [0, 1] {
            let selected = graph.add("swizzle1", &[read], Some(Value::Int(lane as i32)));
            graph.output = Some(graph.add("sequence", &[assigned, selected], None));
            for (seed, words) in [11_038_009, 9_766_418, 22_858_730].into_iter().zip(expected) {
                let result = compile(&graph, seed).unwrap().evaluate(&context, &mut EvaluationState::default()).unwrap();
                assert_eq!(result, Value::Float(f32::from_bits(words[lane])), "{operation}, {seed}, channel {lane}");
            }
        }
    }
}
