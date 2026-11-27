use crate::function::{EvaluationContext, EvaluationState, FunctionScope, compile};
use materialcraft_core::{FunctionGraph, Value};
use std::collections::BTreeMap;

fn constant_call(graph: &mut FunctionGraph, value: f32) -> u32 {
    let id = graph.add("instance", &[], None);
    let node = graph.nodes.last_mut().unwrap();
    node.instance = Some(materialcraft_core::GraphInstance {
        graph: "constant".into(),
        label: String::new(),
        package: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        defaults: BTreeMap::new(),
    });
    node.source = Some(Ok(std::sync::Arc::new(FunctionGraph::constant(Value::Float(value)))));
    id
}

#[test]
fn a_closed_function_call_preserves_shared_draw_order_in_a_pattern_assignment() {
    let mut graph = FunctionGraph::default();
    let first = graph.add("get_bool", &[], Some(Value::Text("first".into())));
    let unit = graph.add_constant(Value::Float(1.0));
    let shared = graph.add("rand", &[unit], None);
    let other = graph.add("rand", &[unit], None);
    let chosen = graph.add("ifelse", &[first, shared, other], None);
    let published = graph.add("set", &[chosen], Some(Value::Text("published".into())));
    let quarter = graph.add_constant(Value::Float(0.25));
    let weighted = graph.add("mul", &[shared, quarter], None);
    let total = graph.add("add", &[published, weighted], None);
    let angle = constant_call(&mut graph, std::f32::consts::PI);
    let eighth = graph.add_constant(Value::Float(0.125));
    let offset = graph.add("mul", &[angle, eighth], None);
    graph.output = Some(graph.add("add", &[total, offset], None));

    // Raw F32 tint words for the two choices on three independently measured stamp seeds.
    for (seed, expected) in [(11_038_009, [0x3f00baf2, 0x3f1faf8a]), (9_766_418, [0x3fa910f3, 0x3f3da6cd]), (22_858_730, [0x3f878418, 0x3fcc33e4])] {
        let program = compile(&graph, seed).unwrap();
        for (first, word) in [false, true].into_iter().zip(expected) {
            let variables = BTreeMap::from([("first".into(), Value::Bool(first))]);
            let context = EvaluationContext {
                scope: FunctionScope::FxMap,
                position: [0.5; 2],
                size: [16.0; 2],
                images: &[],
                image_formats: &[],
                tile: true,
                variables: &variables,
            };
            assert_eq!(program.evaluate(&context, &mut EvaluationState::default()).unwrap(), Value::Float(f32::from_bits(word)), "{seed}, {first}");
        }
    }
}

#[test]
fn reading_one_packed_channel_before_a_pure_call_consumes_only_its_draw() {
    for lane in [0, 1] {
        let mut graph = FunctionGraph::default();
        let unit = graph.add_constant(Value::Float(1.0));
        let [a, b, next] = std::array::from_fn(|_| graph.add("rand", &[unit], None));
        let pair = graph.add("vector2", &[a, b], None);
        let selected = graph.add("swizzle1", &[pair], Some(Value::Int(lane)));
        let saved = graph.add("set", &[selected], Some(Value::Text("published".into())));
        let quarter = graph.add_constant(Value::Float(0.25));
        let weighted = graph.add("mul", &[next, quarter], None);
        let sum = graph.add("add", &[saved, weighted], None);
        let offset = constant_call(&mut graph, 0.125);
        graph.output = Some(graph.add("add", &[sum, offset], None));
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
        for (seed, word) in [(11_038_009, 0x3ea6d4ee), (9_766_418, 0x3f1e3da6), (22_858_730, 0x3f98c3fa)] {
            let value = compile(&graph, seed).unwrap().evaluate(&context, &mut EvaluationState::default()).unwrap();
            assert_eq!(value, Value::Float(f32::from_bits(word)), "{seed}, lane {lane}");
        }
    }
}

#[test]
fn shared_conditional_draws_follow_their_arm_and_keep_their_original_position() {
    let mut graph = FunctionGraph::default();
    let choice = graph.add("get_bool", &[], Some(Value::Text("first".into())));
    let one = graph.add_constant(Value::Float(1.0));
    let quarter = graph.add_constant(Value::Float(0.25));
    let [a, b, c, next] = std::array::from_fn(|_| graph.add("rand", &[one], None));
    let left = graph.add("add", &[b, a], None);
    let right = graph.add("add", &[c, a], None);
    let either = graph.add("ifelse", &[choice, left, right], None);
    let false_only = graph.add("ifelse", &[choice, b, a], None);
    let sum = graph.add("add", &[next, a], None);
    let weighted = graph.add("mul", &[sum, quarter], None);
    let both_result = graph.add("add", &[either, weighted], None);
    let right_result = graph.add("add", &[false_only, weighted], None);
    let pair = graph.add("vector2", &[a, b], None);
    let control = graph.add("vector3", &[pair, c], None);

    for scope in [FunctionScope::Node, FunctionScope::PixelProcessor, FunctionScope::ValueProcessor, FunctionScope::FxMap] {
        for first in [true, false] {
            let values = BTreeMap::from([("first".into(), Value::Bool(first))]);
            let ctx = EvaluationContext { scope, size: [16.0; 2], position: [0.03125; 2], images: &[], image_formats: &[], tile: true, variables: &values };
            for seed in [3, 20] {
                graph.output = Some(control);
                let Value::Float3([r1, r2, r3]) = compile(&graph, seed).unwrap().evaluate(&ctx, &mut EvaluationState::default()).unwrap() else {
                    panic!("three draws")
                };
                let expected_both = r1 + r2 + (r3 + if first { r2 } else { r1 }) * 0.25;
                let expected_right = if first { r1 + (r3 + r2) * 0.25 } else { r1 + (r2 + r1) * 0.25 };
                for (output, expected) in [(both_result, expected_both), (right_result, expected_right)] {
                    graph.output = Some(output);
                    assert_eq!(
                        compile(&graph, seed).unwrap().evaluate(&ctx, &mut EvaluationState::default()).unwrap(),
                        Value::Float(expected),
                        "{scope:?}, {first}, {output}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_assignment_shared_only_with_the_later_arm_keeps_the_earlier_arms_read() {
    let mut graph = FunctionGraph::default();
    let name = Some(Value::Text("tone".into()));
    let choice = graph.add("get_bool", &[], Some(Value::Text("first".into())));
    let start = graph.add_constant(Value::Float(0.125));
    let changed = graph.add_constant(Value::Float(0.625));
    let quarter = graph.add_constant(Value::Float(0.25));
    let init = graph.add("set", &[start], name.clone());
    let set = graph.add("set", &[changed], name.clone());
    let read = graph.add("get_float1", &[], name);
    let left = graph.add("ifelse", &[choice, set, read], None);
    let right = graph.add("ifelse", &[choice, read, set], None);
    let weighted = graph.add("mul", &[set, quarter], None);
    let left_sum = graph.add("add", &[left, weighted], None);
    let right_sum = graph.add("add", &[right, weighted], None);
    let left_result = graph.add("sequence", &[init, left_sum], None);
    let right_result = graph.add("sequence", &[init, right_sum], None);
    for scope in [FunctionScope::Node, FunctionScope::PixelProcessor, FunctionScope::ValueProcessor, FunctionScope::FxMap] {
        for first in [false, true] {
            let values = BTreeMap::from([("first".into(), Value::Bool(first))]);
            let ctx = EvaluationContext { scope, size: [16.0; 2], position: [0.03125; 2], images: &[], image_formats: &[], tile: true, variables: &values };
            for (root, expected) in [(left_result, 0.78125), (right_result, if first { 0.28125 } else { 0.78125 })] {
                graph.output = Some(root);
                assert_eq!(
                    compile(&graph, 0).unwrap().evaluate(&ctx, &mut EvaluationState::default()).unwrap(),
                    Value::Float(expected),
                    "{scope:?}, {first}, {root}"
                );
            }
        }
    }
}

#[test]
fn a_later_random_call_preserves_the_draws_in_a_sequence_assignment() {
    // The second sequence input calls a separate graph that draws from the same stamp stream.
    for (vector, expected) in [
        (true, [[0x3e5c760a, 0x3ee8063f], [0x3f1b5b11, 0x3f297df4], [0x3f4c65e0, 0x3edef0a6]]),
        (false, [[0x3e1fa354, 0x3e74cd9a], [0x3f846c61, 0x3ee92f7e], [0x3f377cfc, 0x3fa0bebe]]),
    ] {
        let mut graph = FunctionGraph::default();
        let first = graph.add("get_bool", &[], Some(Value::Text("first".into())));
        let unit = graph.add_constant(Value::Float(1.0));
        let [a, b, c, d] = std::array::from_fn(|_| graph.add("rand", &[unit], None));
        let name = Some(Value::Text("firstValue".into()));
        let (saved, read) = if vector {
            let left = graph.add("vector2", &[a, b], None);
            let right = graph.add("vector2", &[c, d], None);
            let factor = graph.add_constant(Value::Float(0.375));
            let value = graph.add("lerp", &[left, right, factor], None);
            let saved = graph.add("set", &[value], name.clone());
            let read = graph.add("get_float2", &[], name.clone());
            let x = graph.add("swizzle1", &[read], Some(Value::Int(0)));
            let y = graph.add("swizzle1", &[read], Some(Value::Int(1)));
            (saved, graph.add("ifelse", &[first, y, x], None))
        } else {
            let value = graph.add("ifelse", &[first, a, b], None);
            let saved = graph.add("set", &[value], name.clone());
            let read = graph.add("get_float1", &[], name);
            let quarter = graph.add_constant(Value::Float(0.25));
            let shared = graph.add("mul", &[a, quarter], None);
            (saved, graph.add("add", &[read, shared], None))
        };
        let mut sampler = FunctionGraph::default();
        let range = sampler.add_constant(Value::Float(1.0));
        sampler.output = Some(sampler.add("rand", &[range], None));
        let sampled = graph.add("instance", &[], None);
        let node = graph.nodes.last_mut().unwrap();
        node.instance = Some(materialcraft_core::GraphInstance {
            graph: "next_sample".into(),
            label: String::new(),
            package: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            defaults: BTreeMap::new(),
        });
        node.source = Some(Ok(std::sync::Arc::new(sampler)));
        let eighth = graph.add_constant(Value::Float(0.125));
        let offset = graph.add("mul", &[sampled, eighth], None);
        let result = graph.add("add", &[read, offset], None);
        graph.output = Some(graph.add("sequence", &[saved, result], None));
        for (seed, words) in [11_038_009, 9_766_418, 22_858_730].into_iter().zip(expected) {
            let program = compile(&graph, seed).unwrap();
            for (first, word) in [false, true].into_iter().zip(words) {
                let variables = BTreeMap::from([("first".into(), Value::Bool(first))]);
                let context = EvaluationContext {
                    scope: FunctionScope::FxMap,
                    position: [0.5; 2],
                    size: [16.0; 2],
                    images: &[],
                    image_formats: &[],
                    tile: true,
                    variables: &variables,
                };
                assert_eq!(
                    program.evaluate(&context, &mut EvaluationState::default()).unwrap(),
                    Value::Float(f32::from_bits(word)),
                    "{seed}, {first}, {vector}"
                );
            }
        }
    }
}
