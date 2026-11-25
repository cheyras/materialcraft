use super::*;

fn overlapping_subtree_pixel(repetitions: i32, negative_gain: f32, wide_first_level: bool) -> f32 {
    let mut graph = PatternGraph::with_root(PatternNodeKind::Quadrant);
    let root = graph.root.unwrap();
    if repetitions > 1 {
        let iterate = graph.add(PatternNodeKind::Iterate).unwrap();
        graph.root = Some(iterate);
        let node = &mut graph.nodes[(iterate - 1) as usize];
        node.children[1] = Some(root);
        node.parameters.insert("numberadded".into(), Value::Int(repetitions));
    }
    let mut tail = graph.add(PatternNodeKind::Quadrant).unwrap();
    for _ in 4..8 {
        let parent = graph.add(PatternNodeKind::Quadrant).unwrap();
        graph.nodes[(parent - 1) as usize].children.fill(Some(tail));
        tail = parent;
    }

    let mut parents = vec![(root, [0_usize; 2])];
    for depth in 1..=3 {
        let mut children = Vec::new();
        for (parent, [x, y]) in parents {
            for slot in 0..4 {
                let child = graph.add(PatternNodeKind::Quadrant).unwrap();
                graph.nodes[(parent - 1) as usize].children[slot] = Some(child);
                let cell = [2 * x + slot % 2, 2 * y + slot / 2];
                if depth == 1 && wide_first_level {
                    let node = &mut graph.nodes[(child - 1) as usize];
                    node.parameters = BTreeMap::from([
                        ("patterntype".into(), Value::Int(6)),
                        ("blendingmode".into(), Value::Int(0)),
                        ("patternsize".into(), Value::Float2([5.5 + slot as f32, 7.75 - 0.5 * slot as f32])),
                        ("patternrotation".into(), Value::Float([0.0, 0.125, 0.21875, 0.34375][slot])),
                        ("opacity".into(), Value::Float(2.0_f32.powi(-80))),
                    ]);
                }
                if depth == 3 {
                    let index = children.len();
                    let (target, gain) = match index {
                        50 => (0.25, negative_gain),
                        0 => (0.25, 2.0_f32.powi(-25)),
                        _ => (0.75, 0.125),
                    };
                    let shift = cell.map(|coordinate| target - (coordinate as f32 + 0.5) / 8.0);
                    let node = &mut graph.nodes[(child - 1) as usize];
                    node.parameters = additive_square(gain, [0.0; 2], shift);
                    node.children.fill(Some(tail));
                }
                children.push((child, cell));
            }
        }
        parents = children;
    }
    let image = cook_in_regions(graph, &BTreeMap::new(), 41381, [32; 2], [[0.0, 1.0, 0.0, 1.0]; 2], 1.0).unwrap().remove(0);
    image.get_gray(8, 8, false)
}

#[test]
fn overlapping_subtrees_in_an_underfilled_eight_level_tree() {
    let small = 2.0_f32.powi(-25);
    assert_eq!(overlapping_subtree_pixel(1, -1.0, false).to_bits(), small.to_bits());
    // The three negative draws reach 0.25 before the three small additions.
    assert_eq!(overlapping_subtree_pixel(3, -0.25, false).to_bits(), (0.25 + 3.0 * small).to_bits());
}

#[test]
fn wide_first_level_stamps_keep_the_order_of_deeper_draws() {
    assert_eq!(overlapping_subtree_pixel(1, -1.0, true).to_bits(), 2.0_f32.powi(-25).to_bits());
}

enum Continuation {
    None,
    EmptySwitch,
    InactiveBranch(u32),
}

fn quadrant_tail(graph: &mut PatternGraph, mut next: Option<u32>, wrapped: bool) -> Option<u32> {
    for depth in (4..=8).rev() {
        let id = graph.add(PatternNodeKind::Quadrant).unwrap();
        let node = &mut graph.nodes[(id - 1) as usize];
        node.parameters = additive_square(2.0_f32.powi(-76), [0.0; 2], [0.0; 2]);
        if wrapped && depth == 8 {
            let mut offset = FunctionGraph::default();
            let corner = offset.add_constant(Value::Float2([0.0; 2]));
            let center = offset.add("get_float2", &[], Some(Value::Text("$pos".into())));
            offset.output = Some(offset.add("sub", &[corner, center], None));
            node.parameters.insert("frameoffset".into(), Value::Function(Box::new(offset)));
        }
        node.children.fill(next);
        next = Some(id);
    }
    next
}

fn filled_tree_pixel(negative: (usize, usize), positive: (usize, usize), continuation: Continuation, wrapped_subtrees: &[usize]) -> f32 {
    let mut graph = PatternGraph::with_root(PatternNodeKind::Iterate);
    let iterate = graph.root.unwrap();
    let root = graph.add(PatternNodeKind::Quadrant).unwrap();
    graph.nodes[(iterate - 1) as usize].children[1] = Some(root);
    let end = matches!(continuation, Continuation::EmptySwitch).then(|| graph.add(PatternNodeKind::Switch).unwrap());
    let plain = quadrant_tail(&mut graph, end, false);
    let wrapped = if wrapped_subtrees.is_empty() { plain } else { quadrant_tail(&mut graph, end, true) };
    let mut parents = vec![(root, [0_usize; 2])];
    for depth in 1..=3 {
        let mut children = Vec::new();
        for (parent, [x, y]) in parents {
            for slot in 0..4 {
                let id = graph.add(PatternNodeKind::Quadrant).unwrap();
                graph.nodes[(parent - 1) as usize].children[slot] = Some(id);
                let cell = [2 * x + slot % 2, 2 * y + slot / 2];
                let ordinal = children.len();
                let node = &mut graph.nodes[(id - 1) as usize];
                if depth >= 2 {
                    let marker = (depth, ordinal);
                    let (gain, shift) = if marker == negative || marker == positive {
                        let shift = cell.map(|i| 0.265625 - (i as f32 + 0.5) / (1 << depth) as f32);
                        (if marker == negative { -1.0 } else { 2.0_f32.powi(-25) }, shift)
                    } else {
                        (2.0_f32.powi(-76), [0.0; 2])
                    };
                    node.parameters = additive_square(gain, [0.0; 2], shift);
                }
                if depth == 3 {
                    node.children.fill(if wrapped_subtrees.contains(&ordinal) { wrapped } else { plain });
                }
                children.push((id, cell));
            }
        }
        parents = children;
    }
    if let Continuation::InactiveBranch(depth) = continuation {
        let mut child = graph.add(PatternNodeKind::Quadrant).unwrap();
        for _ in 0..depth {
            let parent = graph.add(PatternNodeKind::Quadrant).unwrap();
            graph.nodes[(parent - 1) as usize].children[0] = Some(child);
            child = parent;
        }
        let switch = graph.add(PatternNodeKind::Switch).unwrap();
        graph.nodes[(switch - 1) as usize].children = vec![Some(root), Some(child)];
        graph.nodes[(iterate - 1) as usize].children[1] = Some(switch);
    }
    let image = cook_in_regions(graph, &BTreeMap::new(), 41381, [32; 2], [[0.0, 1.0, 0.0, 1.0]; 2], 1.0).unwrap().remove(0);
    image.get_gray(8, 8, false)
}

fn filled_quadrant_pixel(negative: usize, positive: usize, continuation: Continuation) -> f32 {
    filled_tree_pixel((2, negative), (2, positive), continuation, &[])
}

#[test]
fn successive_quadrant_batches_preserve_their_child_ranges() {
    for (negative, positive) in [((2, 14), (2, 8)), ((2, 1), (3, 2)), ((2, 13), (2, 8)), ((2, 9), (3, 34))] {
        assert_eq!(filled_tree_pixel(negative, positive, Continuation::EmptySwitch, &[]).to_bits(), 2.0_f32.powi(-25).to_bits());
    }
    assert_eq!(filled_quadrant_pixel(14, 8, Continuation::None).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn wrapped_leaves_can_fill_a_quadrant_batch_before_a_leading_pair() {
    assert_eq!(filled_tree_pixel((2, 14), (2, 2), Continuation::EmptySwitch, &[]).to_bits(), 0);
    assert_eq!(filled_tree_pixel((2, 14), (2, 2), Continuation::EmptySwitch, &[58]).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn filled_quadrant_batches_submit_the_middle_pair_before_the_upper_pair() {
    assert_eq!(filled_quadrant_pixel(10, 2, Continuation::None).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn a_full_batch_collects_separate_ranges_of_the_last_quadrant() {
    assert_eq!(filled_quadrant_pixel(15, 4, Continuation::None).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn a_filled_batch_can_start_in_the_first_branch_of_the_last_quadrant() {
    assert_eq!(filled_quadrant_pixel(12, 0, Continuation::None).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn a_closed_switch_below_the_final_quadrants_changes_the_next_full_batch() {
    let small = 2.0_f32.powi(-25).to_bits();
    assert_eq!(filled_quadrant_pixel(5, 0, Continuation::None).to_bits(), 0);
    assert_eq!(filled_quadrant_pixel(5, 0, Continuation::EmptySwitch).to_bits(), small);
    assert_eq!(filled_quadrant_pixel(0, 5, Continuation::EmptySwitch).to_bits(), 0);
}

#[test]
fn an_inactive_branch_contributes_its_connected_quadrant_depth() {
    assert_eq!(filled_quadrant_pixel(5, 0, Continuation::InactiveBranch(8)).to_bits(), 0);
    assert_eq!(filled_quadrant_pixel(5, 0, Continuation::InactiveBranch(9)).to_bits(), 2.0_f32.powi(-25).to_bits());
}
