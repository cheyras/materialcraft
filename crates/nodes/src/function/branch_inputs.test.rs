use crate::function::{EvaluationContext, EvaluationState, FunctionScope, compile};
use materialcraft_core::{FunctionGraph, Value};
use std::collections::BTreeMap;

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
