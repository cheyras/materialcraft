use materialcraft_core::{FunctionGraph, Graph, OutputFormat, Package, PatternGraph, PatternNodeKind, Value, ValueType};
use materialcraft_eval::{Cache, Options, cook};
use std::sync::Mutex;

#[test]
fn bound_pattern_property_changes_the_first_local_read_at_equal_opacity() {
    let mut graph = Graph::new("First quadrant read");
    graph.size = [4; 2];
    graph.format = OutputFormat::F32;
    let stamp = graph.add_node("fx_map", [0.0; 2]);
    let node = graph.node_mut(stamp).unwrap();
    node.params.insert("color_mode".into(), Value::Bool(false));
    node.params.insert("roughness".into(), Value::Float(0.0));
    let mut pattern = PatternGraph::with_root(PatternNodeKind::Quadrant);
    let fields = &mut pattern.nodes[0].parameters;
    fields.insert("patterntype".into(), Value::Int(2));
    let mut tint = FunctionGraph::input("local", ValueType::Float);
    let read = tint.output.unwrap();
    let equal = tint.add("eq", &[read, read], None);
    let yes = tint.add_constant(Value::Float(0.375));
    let no = tint.add_constant(Value::Float(0.625));
    let choose = tint.add("ifelse", &[equal, yes, no], None);
    tint.output = Some(tint.add("set", &[choose], Some(Value::Text("local".into()))));
    fields.insert("opacity".into(), Value::Function(Box::new(tint)));
    node.params.insert("pattern".into(), Value::PatternGraph(Box::new(pattern)));

    let mut package = Package::new("Quadrant parameter origin");
    package.graphs.push(graph);
    let mut options = Options::new(&|_| None);
    options.overrides.insert("gain".into(), Value::Float(1.0));
    let cache = Mutex::new(Cache::default());
    for direct in [true, false] {
        let node = package.graphs[0].node_mut(stamp).unwrap();
        if direct {
            node.bindings.insert("opacity".into(), "gain".into());
        } else {
            node.bindings.clear();
            node.functions.insert("opacity".into(), FunctionGraph::input("gain", ValueType::Float));
        }
        for fixed in [true, false, true] {
            options.literal_inputs = if fixed { vec!["gain".into()] } else { Vec::new() };
            let result = cook(&package, 0, &options, &cache);
            assert_eq!(result.nodes[&stamp].error, None);
            let expected = if fixed { 0.375 } else { 0.625 };
            assert_eq!(result.image(stamp, 0).unwrap().get_gray(8, 8, false), expected, "direct {direct}, fixed {fixed}");
        }
    }

    let graph = &mut package.graphs[0];
    graph.node_mut(stamp).unwrap().functions.clear();
    graph.output_size_function = Some(FunctionGraph::input("offset", ValueType::Integer2));
    options.overrides.insert("offset".into(), Value::Int2([0; 2]));
    for (mode, expected) in [(materialcraft_core::SizeMode::RelativeToParent, 0.625), (materialcraft_core::SizeMode::Absolute, 0.375)] {
        let base = &mut package.graphs[0].node_mut(stamp).unwrap().base;
        base.size_mode = mode;
        base.size = if mode == materialcraft_core::SizeMode::Absolute { [4; 2] } else { [0; 2] };
        let result = cook(&package, 0, &options, &cache);
        assert_eq!(result.nodes[&stamp].error, None);
        assert_eq!(result.image(stamp, 0).unwrap().get_gray(0, 0, false), expected, "{mode:?}");
    }
}

#[test]
fn gray_pattern_opacity_keeps_signed_and_undefined_contributions_until_blending() {
    let mut graph = Graph::new("Signed stamp opacity");
    graph.size = [4; 2];
    graph.format = OutputFormat::F32;
    let stamp = graph.add_node("fx_map", [0.0; 2]);
    let node = graph.node_mut(stamp).unwrap();
    node.params.insert("color_mode".into(), Value::Bool(false));
    node.params.insert("roughness".into(), Value::Float(0.0));
    let mut opacity = FunctionGraph::input("%gain", ValueType::Float);
    let numerator = opacity.output.unwrap();
    let denominator = opacity.add_constant(Value::Float(0.0));
    opacity.output = Some(opacity.add("div", &[numerator, denominator], None));
    node.functions.insert("opacity".into(), opacity);

    let mut package = Package::new("Stamp storage");
    package.graphs.push(graph);
    let cache = Mutex::new(Cache::default());
    let mut options = Options::new(&|_| None);
    let mut pattern = PatternGraph::with_root(PatternNodeKind::Quadrant);
    let fields = &mut pattern.nodes[0].parameters;
    fields.insert("patterntype".into(), Value::Int(2));
    fields.insert("patternsize".into(), Value::Float2([0.5; 2]));
    fields.insert("opacity".into(), Value::Float4([0.25, 0.25, 0.25, 1.0]));
    for (mode, expected) in [(0, [f32::MAX.to_bits(), f32::MIN.to_bits(), 0x7fff_ffff]), (1, [f32::MAX.to_bits(), 0, 0])] {
        pattern.nodes[0].parameters.insert("blendingmode".into(), Value::Int(mode));
        package.graphs[0].node_mut(stamp).unwrap().params.insert("pattern".into(), Value::PatternGraph(Box::new(pattern.clone())));
        for (gain, bits) in [1.0, -1.0, 0.0].into_iter().zip(expected) {
            options.overrides.insert("gain".into(), Value::Float(gain));
            let result = cook(&package, 0, &options, &cache);
            assert_eq!(result.nodes[&stamp].error, None);
            let image = result.image(stamp, 0).unwrap();
            assert_eq!(image.get_gray(8, 8, false).to_bits(), bits);
            assert_eq!(image.get_gray(0, 0, false).to_bits(), 0);
        }
    }

    let child = pattern.add(PatternNodeKind::Quadrant).unwrap();
    pattern.nodes[0].children[0] = Some(child);
    pattern.nodes[0].parameters.insert("blendingmode".into(), Value::Int(0));
    pattern.nodes[0].parameters.insert("patternsize".into(), Value::Float2([1.0; 2]));
    pattern.nodes[1].parameters = pattern.nodes[0].parameters.clone();
    pattern.nodes[1].parameters.insert("opacity".into(), Value::Float4([-0.25, -0.25, -0.25, 1.0]));
    package.graphs[0].node_mut(stamp).unwrap().params.insert("pattern".into(), Value::PatternGraph(Box::new(pattern)));
    options.overrides.insert("gain".into(), Value::Float(1.0));
    let result = cook(&package, 0, &options, &cache);
    let image = result.image(stamp, 0).unwrap();
    assert_eq!(image.get_gray(4, 4, false), f32::MIN);
    assert_eq!(image.get_gray(12, 12, false), f32::MAX);
}
