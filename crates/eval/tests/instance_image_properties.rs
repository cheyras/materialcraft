use materialcraft_core::{FunctionGraph, Graph, OutputFormat, Package, ParameterInheritance, Value};
use materialcraft_eval::{Cache, Options, cook};
use std::sync::Mutex;

#[test]
fn each_instance_image_pin_preserves_its_own_sampling_properties() {
    let mut parent = Graph::new("Caller");
    let mut child = Graph::new("Two fields");
    for graph in [&mut parent, &mut child] {
        graph.size = [6; 2];
        graph.format = OutputFormat::U16;
    }
    let instance = parent.add_node("graph_instance", [0.0; 2]);
    parent.node_mut(instance).unwrap().params.insert("graph".into(), Value::Text("Two fields".into()));
    let mut filters = Vec::new();
    for (name, size, tiling, ratio) in [("first", [2.0, 0.5], [false, true], 0), ("second", [0.5, 2.0], [true, false], 1)] {
        let mut noise = FunctionGraph::default();
        let unit = noise.add_constant(Value::Float(1.0));
        noise.output = Some(noise.add("rand", &[unit], None));
        let source = parent.add_node("pixel_processor", [0.0; 2]);
        let field = parent.node_mut(source).unwrap();
        field.params.insert("function".into(), Value::Function(Box::new(noise)));
        field.params.insert("color_mode".into(), Value::Bool(false));
        field.base.pixel_size_mode = ParameterInheritance::Absolute;
        field.base.pixel_size = size;
        field.base.tiling_mode = ParameterInheritance::Absolute;
        field.base.tiling = tiling;
        field.base.pixel_ratio_mode = ParameterInheritance::Absolute;
        field.base.pixel_ratio = ratio;
        parent.connect(source, "output", instance, name).unwrap();

        let input = child.add_node("input_grayscale", [0.0; 2]);
        child.node_mut(input).unwrap().params.insert("identifier".into(), Value::Text(name.into()));
        let inside = child.add_node("blur", [0.0; 2]);
        child.node_mut(inside).unwrap().params.insert("intensity".into(), Value::Float(12.0));
        child.connect(input, "output", inside, "input").unwrap();
        let output = child.add_node("output", [0.0; 2]);
        child.node_mut(output).unwrap().params.insert("identifier".into(), Value::Text(name.into()));
        child.connect(inside, "output", output, "input").unwrap();

        let flat = parent.add_node("blur", [0.0; 2]);
        parent.node_mut(flat).unwrap().params.insert("intensity".into(), Value::Float(12.0));
        parent.connect(source, "output", flat, "input").unwrap();
        let receiver = parent.add_node("blur", [0.0; 2]);
        parent.node_mut(receiver).unwrap().params.insert("intensity".into(), Value::Float(0.0));
        parent.connect(instance, name, receiver, "input").unwrap();
        filters.push((receiver, flat, size, tiling, ratio));
    }
    let mut package = Package::new("Image pin properties");
    package.graphs.extend([parent, child]);
    let cache = Mutex::new(Cache::default());
    for cached in [false, true] {
        let cooked = cook(&package, 0, &Options::new(&|_| None), &cache);
        assert_eq!(cooked.nodes[&instance].error, None);
        assert_eq!(cooked.nodes[&instance].cached, cached);
        for &(receiver, flat, size, tiling, ratio) in &filters {
            let base = cooked.nodes[&receiver].base;
            assert_eq!((base.pixel_size, base.tiling, base.pixel_ratio), (size, tiling, ratio));
            assert_eq!(cooked.image(receiver, 0).unwrap().data(), cooked.image(flat, 0).unwrap().data());
        }
    }
}
