use super::*;

fn overlapping_subtree_pixel(repetitions: i32, negative_gain: f32) -> f32 {
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
    assert_eq!(overlapping_subtree_pixel(1, -1.0).to_bits(), small.to_bits());
    // The three negative draws reach 0.25 before the three small additions.
    assert_eq!(overlapping_subtree_pixel(3, -0.25).to_bits(), (0.25 + 3.0 * small).to_bits());
}
