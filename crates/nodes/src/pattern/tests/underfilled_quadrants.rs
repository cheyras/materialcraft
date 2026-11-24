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

fn filled_quadrant_pixel(negative: usize, positive: usize) -> f32 {
    let mut graph = PatternGraph::with_root(PatternNodeKind::Iterate);
    let iterate = graph.root.unwrap();
    let root = graph.add(PatternNodeKind::Quadrant).unwrap();
    graph.nodes[(iterate - 1) as usize].children[1] = Some(root);
    let faint = 2.0_f32.powi(-76);
    let small = 2.0_f32.powi(-25);
    let mut next = None;
    for _ in (3..=8).rev() {
        let id = graph.add(PatternNodeKind::Quadrant).unwrap();
        let node = &mut graph.nodes[(id - 1) as usize];
        node.parameters = additive_square(faint, [0.0; 2], [0.0; 2]);
        node.children.fill(next);
        next = Some(id);
    }
    for quadrant in 0..4 {
        let first = graph.add(PatternNodeKind::Quadrant).unwrap();
        graph.nodes[(root - 1) as usize].children[quadrant] = Some(first);
        for slot in 0..4 {
            let second = graph.add(PatternNodeKind::Quadrant).unwrap();
            graph.nodes[(first - 1) as usize].children[slot] = Some(second);
            let cell = [2 * (quadrant % 2) + slot % 2, 2 * (quadrant / 2) + slot / 2];
            let ordinal = 4 * quadrant + slot;
            let gain = if ordinal == negative {
                -1.0
            } else if ordinal == positive {
                small
            } else {
                faint
            };
            let shift = if gain == faint { [0.0; 2] } else { cell.map(|i| 0.265625 - (i as f32 + 0.5) * 0.25) };
            let node = &mut graph.nodes[(second - 1) as usize];
            node.parameters = additive_square(gain, [0.0; 2], shift);
            node.children.fill(next);
        }
    }
    let image = cook_in_regions(graph, &BTreeMap::new(), 41381, [32; 2], [[0.0, 1.0, 0.0, 1.0]; 2], 1.0).unwrap().remove(0);
    image.get_gray(8, 8, false)
}

#[test]
fn filled_quadrant_batches_submit_the_middle_pair_before_the_upper_pair() {
    assert_eq!(filled_quadrant_pixel(10, 2).to_bits(), 2.0_f32.powi(-25).to_bits());
}

#[test]
fn a_full_batch_collects_separate_ranges_of_the_last_quadrant() {
    assert_eq!(filled_quadrant_pixel(15, 4).to_bits(), 2.0_f32.powi(-25).to_bits());
}
