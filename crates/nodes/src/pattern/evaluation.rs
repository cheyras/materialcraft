use materialcraft_core::{PatternNode, PatternNodeKind, Value, ValueType};
use std::collections::{BTreeMap, BTreeSet};

use super::mixed_depth_runs::MixedDepthRuns;
use super::stamp_cache::{self, StampCache};
use super::{catalog, footprint::SampleGrid, pattern_stamp::PatternStamp, pattern_visit::PatternVisit, raster, variables::PatternVariables};
use crate::function::{EvaluationContext, EvaluationState, Program, compile_with_literal_inputs};
use crate::random::FunctionRandom;
use crate::{CookContext, CookResult};

// A 512-by-512 quadtree takes 349,525 visits. Retained, discarded and streamed
// draws have bounded allowances. At most four terminal draws share a parent's
// visit, and full property evaluations keep their five-allowance ceiling.
const MAX_VISITS: usize = 1_048_576;

// Overlapping squares establish the order of the sixteen grandchildren in a regular seven-level tree.
const SEVEN_LEVEL_RUN_ORDER: [usize; 16] = [12, 13, 8, 9, 14, 0, 1, 15, 2, 11, 6, 7, 10, 3, 4, 5];

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuadrantChildren {
    None,
    AllTerminal,
    Other,
}

type PatternNodes<'a> = BTreeMap<u32, (&'a PatternNode, bool, QuadrantChildren)>;
type PropertyPrograms<'a> = BTreeMap<(u32, &'a str), (Program, BTreeMap<String, ValueType>)>;

fn reusable_leaf(node: &PatternNode, programs: &PropertyPrograms, assigned: &BTreeSet<&str>, grayscale: bool) -> bool {
    let Some(shape) = node.parameters.get("patterntype").and_then(Value::as_i32) else { return false };
    let image = image_pattern(shape);
    if !(2..=18).contains(&shape) && !(grayscale && image) {
        return false;
    }
    let mut computed = false;
    let Some(definition) = catalog::definition(node.kind) else { return false };
    for name in definition.params.iter().map(|p| p.id).filter(|name| uses_property(node.kind, name, shape)) {
        if let Some((program, _)) = programs.get(&(node.id, name)) {
            // A fixed-image entry stores the branch offset, size and grayscale tint.
            // Every other field, including the selected slot, must be constant for this leaf.
            if image && !matches!(name, "branchoffset" | "patternsize" | "opacity") {
                return false;
            }
            if matches!(name, "patterntype" | "randomseed" | "randominherited") || !stamp_cache::reusable(program, assigned) {
                return false;
            }
            computed = true;
        }
    }
    computed
}

struct TerminalSiblings {
    stack: usize,
    discarded: bool,
}

fn uniform_deep_draws(root: Option<u32>, nodes: &PatternNodes) -> bool {
    let mut pending: Vec<_> = root.into_iter().map(|id| (id, 0_u8)).collect();
    let mut seen = BTreeSet::new();
    let mut order = None;
    while let Some((id, depth)) = pending.pop() {
        if !seen.insert((id, depth)) {
            continue;
        }
        let Some(&(node, _, _)) = nodes.get(&id) else { return false };
        let quadrant = node.kind == PatternNodeKind::Quadrant;
        if quadrant {
            let literal = |name: &str| node.parameters.get(name).map_or(Some(0), Value::as_i32);
            let Some(shape) = literal("patterntype") else { return false };
            if shape != 0 {
                // A drawing first-level quadrant can require reordering across outer iterations.
                let Some(blend @ 0..=2) = literal("blendingmode") else { return false };
                let (pattern, filter) = if image_pattern(shape) {
                    let Some(slot) = literal("imageindex").and_then(|index| fixed_image_slot(shape, index)) else { return false };
                    let Some(filter @ 0..=2) = literal("imagefiltering") else { return false };
                    if slot >= crate::MAX_IMAGE_INPUTS as i32 {
                        return false;
                    }
                    (256 + slot, filter)
                } else if (2..=16).contains(&shape) {
                    (shape, 0)
                } else {
                    return false;
                };
                let rank = (pattern, filter, blend != 0);
                if depth < 2 || *order.get_or_insert(rank) != rank {
                    return false;
                }
            }
        }
        let child_depth = (depth + u8::from(quadrant)).min(2);
        pending.extend(node.children.iter().flatten().map(|&child| (child, child_depth)));
    }
    order.is_some()
}

fn quadrant_mode(kind: PatternNodeKind, name: &str) -> bool {
    kind == PatternNodeKind::Quadrant && matches!(name, "patterntype" | "blendingmode" | "imagefiltering" | "imagepremul" | "randominherited")
}

fn image_pattern(pattern: i32) -> bool {
    pattern == 1 || (256..256 + crate::MAX_IMAGE_INPUTS as i32).contains(&pattern)
}

fn fixed_image_slot(pattern: i32, requested: i32) -> Option<i32> {
    // A fixed zero preserves the packed slot. The generic image pattern starts at slot zero.
    (requested >= 0).then_some(if requested == 0 { pattern.max(256) - 256 } else { requested })
}

fn uses_property(kind: PatternNodeKind, name: &str, pattern: i32) -> bool {
    match name {
        "randomseed" | "randominherited" | "patterntype" | "branchoffset" => true,
        "imageindex" | "imagefiltering" | "imagepremul" => image_pattern(pattern),
        _ => kind != PatternNodeKind::Quadrant || pattern != 0,
    }
}

fn has_bound_draw(root: Option<u32>, nodes: &PatternNodes, programs: &PropertyPrograms, context: &CookContext, modes: &BTreeMap<String, Value>) -> bool {
    // Both connected alternatives contribute, even when a switch or zero iteration skips one during traversal.
    let mut pending: Vec<_> = root.into_iter().collect();
    let mut seen = BTreeSet::new();
    let mut state = EvaluationState::default();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(&(node, _, _)) = nodes.get(&id) else { continue };
        pending.extend(node.children.iter().flatten().copied());
        let Some(definition) = catalog::definition(node.kind) else { continue };
        let mut pattern = None;
        for field in &definition.params {
            if field.id == "randominherited"
                || quadrant_mode(node.kind, field.id)
                || !programs
                    .get(&(id, field.id))
                    .is_some_and(|(program, locals)| program.reads_bound_input(|name| !locals.contains_key(name) && context.vars.contains_key(name)))
            {
                continue;
            }
            if uses_property(node.kind, field.id, 0) {
                return true;
            }
            let shape = pattern.get_or_insert_with(|| {
                if let Some((program, _)) = programs.get(&(id, "patterntype")) {
                    let evaluation = EvaluationContext {
                        image_formats: context.input_formats.get(1..).unwrap_or(&[]),
                        scope: crate::function::FunctionScope::FxMap,
                        position: [0.0; 2],
                        size: [context.w as f32, context.h as f32],
                        images: context.inputs.get(1..).unwrap_or(&[]),
                        tile: context.tile,
                        variables: modes,
                    };
                    program.evaluate_with_random(&evaluation, &mut state, &mut FunctionRandom::image(context.seed)).ok().and_then(|v| v.as_i32()).unwrap_or(0)
                } else {
                    node.parameters.get("patterntype").and_then(Value::as_i32).unwrap_or(0)
                }
            });
            if uses_property(node.kind, field.id, *shape) {
                return true;
            }
        }
    }
    false
}

fn has_uniform_quadrant_leaves(root: u32, depth: u32, nodes: &PatternNodes) -> bool {
    let mut level = BTreeSet::from([root]);
    for current in 0..=depth {
        let mut next = BTreeSet::new();
        for id in level {
            let Some(&(node, _, children)) = nodes.get(&id) else { return false };
            if node.kind != PatternNodeKind::Quadrant || (children == QuadrantChildren::None) != (current == depth) {
                return false;
            }
            next.extend(node.children.iter().flatten().copied());
        }
        level = next;
    }
    true
}

#[derive(Default)]
struct SevenLevelRuns {
    seen_root: bool,
    starts: [usize; 17],
    grandchildren: usize,
    child_counts: [Option<usize>; 2],
    parents_drawn: usize,
    leaves_drawn: usize,
}

impl SevenLevelRuns {
    fn visit(&mut self, node: &PatternNode, depth: u32, retained: usize) -> bool {
        if node.kind != PatternNodeKind::Quadrant {
            return true;
        }
        if depth == 0 {
            // A second quadrant tree interleaves its draws with the first tree's grandchildren.
            if self.seen_root {
                return false;
            }
            self.seen_root = true;
        }
        if depth == 2 {
            let Some(start) = self.starts.get_mut(self.grandchildren) else { return false };
            *start = retained;
            self.grandchildren += 1;
        }
        match depth {
            0..=4 => !node.children.contains(&None),
            5 | 6 => {
                let count = node.children.iter().flatten().count();
                count > 0 && *self.child_counts[(depth - 5) as usize].get_or_insert(count) == count
            }
            _ => true,
        }
    }

    fn draw(&mut self, stamp: &PatternStamp) -> bool {
        if stamp.blend != 0 || !(2..=18).contains(&stamp.pattern) {
            return false;
        }
        match stamp.depth {
            6 => self.parents_drawn += 1,
            7 => self.leaves_drawn += 1,
            _ => return false,
        }
        true
    }

    fn finish(mut self, retained: usize) -> Option<[usize; 17]> {
        let [Some(penultimate), Some(terminal)] = self.child_counts else { return None };
        // The first five full subdivisions give 1,024 parents of the penultimate level.
        let leaves = 1024 * penultimate * terminal;
        let measured =
            (self.parents_drawn == 0 && (penultimate == 4 || (penultimate == 3 && terminal == 4))) || (self.parents_drawn == 4096 && leaves == 16384);
        if self.grandchildren != SEVEN_LEVEL_RUN_ORDER.len() || self.leaves_drawn != leaves || !measured {
            return None;
        }
        self.starts[SEVEN_LEVEL_RUN_ORDER.len()] = retained;
        Some(self.starts)
    }
}

pub(crate) fn cook(context: &CookContext) -> CookResult {
    cook_with_base_level_inputs(context, &[])
}

/// Cook a pattern with the supplied base-level-only flags in image port order.
pub fn cook_with_base_level_inputs(context: &CookContext, base_level_only: &[bool]) -> CookResult {
    cook_with_bound_properties(context, base_level_only, false)
}

/// A live input to a drawing property affects reads of a local preceding its first assignment.
pub fn cook_with_bound_properties(context: &CookContext, base_level_only: &[bool], bound_property: bool) -> CookResult {
    cook_with_visit_limit(context, base_level_only, MAX_VISITS, bound_property)
}

/// Cook the image and report live drawing inputs for a downstream pattern.
pub fn cook_with_image_dependencies(context: &CookContext, base_level_only: &[bool], bound_property: bool) -> Result<(craft_texture::Image, bool), String> {
    evaluate(context, base_level_only, MAX_VISITS, bound_property)
}

pub(super) fn cook_with_visit_limit(context: &CookContext, base_level_only: &[bool], visit_limit: usize, bound_property: bool) -> CookResult {
    evaluate(context, base_level_only, visit_limit, bound_property).map(|(image, _)| vec![image])
}

fn evaluate(context: &CookContext, base_level_only: &[bool], visit_limit: usize, bound_property: bool) -> Result<(craft_texture::Image, bool), String> {
    let Some(Value::PatternGraph(graph)) = context.params.get("pattern") else {
        return Err("missing pattern graph".into());
    };

    let opacity_depth = graph.validated_depth()?;
    let terminal_nodes: BTreeSet<_> =
        graph.nodes.iter().filter(|node| node.kind == PatternNodeKind::Quadrant && node.children.iter().all(Option::is_none)).map(|node| node.id).collect();
    let mut nodes = BTreeMap::new();
    let mut functions = BTreeMap::new();
    let mut fixed_switches = BTreeSet::new();
    for node in &graph.nodes {
        let mut fixed = node.kind == PatternNodeKind::Switch && node.children.contains(&None);
        let mut literal_geometry = true;
        for (name, value) in &node.parameters {
            if let Value::Function(function) = value {
                let program = compile_with_literal_inputs(function, context.seed, context.vars.literal_inputs())?;
                if matches!(name.as_str(), "patternsize" | "patternrotation") {
                    literal_geometry &= program.output_is_literal();
                }
                fixed = fixed && name == "switch" && program.depends_only_on_image_size();
                let mut names = BTreeMap::new();
                super::variables::declarations(&program, &mut names);
                functions.insert((node.id, name.as_str()), (program, names));
            }
        }
        if fixed {
            fixed_switches.insert(node.id);
        }
        let quadrant_children = if terminal_nodes.contains(&node.id) {
            QuadrantChildren::None
        } else if node.kind == PatternNodeKind::Quadrant && node.children.iter().flatten().all(|id| terminal_nodes.contains(id)) {
            QuadrantChildren::AllTerminal
        } else {
            QuadrantChildren::Other
        };
        nodes.insert(node.id, (node, literal_geometry, quadrant_children));
    }
    let grayscale = !context.b("color_mode");
    let assigned = functions.values().flat_map(|(_, names)| names.keys().map(String::as_str)).collect();
    let reusable_leaves: BTreeSet<_> =
        graph.nodes.iter().filter(|n| terminal_nodes.contains(&n.id) && reusable_leaf(n, &functions, &assigned, grayscale)).map(|n| n.id).collect();
    let mut stamp_cache = StampCache::default();

    let image_slots: Vec<_> =
        context.inputs.iter().skip(1).take(crate::MAX_IMAGE_INPUTS).enumerate().filter_map(|(i, image)| image.as_ref().map(|_| i as i32)).collect();
    let mut pending = Vec::new();
    if let Some(root) = graph.root {
        pending.push((PatternVisit { node: root, center: [0.5; 2], extent: 1.0, depth: 0, number: 0, seed: context.seed }, 0_u32));
    }

    let mut stamps = Vec::new();
    let mut runtime_image_selector = false;
    let mut root_siblings = None;
    let mut root_children_pending = 0;
    let mut branch_starts = [None; 4];

    let mut variables = PatternVariables::new(context.vars.graph_values());
    let has_locals = functions.values().any(|(_, names)| !names.is_empty());
    variables.bound_draw =
        bound_property || (context.vars.keys().any(|name| !name.starts_with('$')) && has_bound_draw(graph.root, &nodes, &functions, context, &variables.modes));
    let separate_child_variables = opacity_depth == 6 && has_locals && graph.root.is_some_and(|root| has_uniform_quadrant_leaves(root, opacity_depth, &nodes));
    let mut max_depth = 0;
    let mut state = EvaluationState::default();
    let mut terminal_siblings: Option<TerminalSiblings> = None;
    let mut reused_terminal_misses = 0_usize;
    let mut remaining = visit_limit;
    let mut discard_allowance = visit_limit;
    let mut full_evaluations = visit_limit.saturating_mul(5);
    let mut cached_stream_allowance = visit_limit;
    let mut accounted_stamps = 0;
    let stream_chunk = (visit_limit / 4).clamp(1, 4_096);
    let mut uniform_order = None;
    let mut raster_work = raster::RasterWork::new();
    let mut repeated_roots = None;
    let mut repeated_six_level: Option<super::repeated_six_level::RepeatedSixLevelRuns> = None;
    let mut terminal_quadrants = (grayscale && opacity_depth == 8 && !context.b("cull_outside")).then(super::terminal_quadrants::TerminalQuadrants::default);
    let mut seven_level_runs = (grayscale && opacity_depth == 7).then(SevenLevelRuns::default);
    let mut underfilled =
        (grayscale && opacity_depth >= 8 && !context.b("cull_outside")).then(|| super::underfilled_quadrants::UnderfilledQuadrants::new(opacity_depth));
    let mut sparse_runs = if grayscale && opacity_depth == 8 { super::tile_runs::SparseQuadrantRuns::new(context) } else { None };
    let mut mixed_depth_runs = if grayscale && matches!(opacity_depth, 8..=10) { MixedDepthRuns::new(context, opacity_depth) } else { None };
    // A grayscale opacity expression can produce an infinity or NaN which participates in the blend.
    let opacity = context.value("opacity").and_then(Value::as_f32).unwrap_or(0.0);
    let nonfinite_gray_opacity = grayscale && !opacity.is_finite();
    let opacities = raster::depth_opacities(context.f("roughness"), opacity_depth, opacity);
    let grid = opacities.as_deref().ok().and_then(|weights| SampleGrid::new(context, weights));
    let mut parameter_values = Vec::new();
    let mut children = Vec::new();
    let mut empty_switches = BTreeSet::new();
    while let Some((mut visit, repeats_left)) = pending.pop() {
        // Returning below the previous stack size starts the next child of the root.
        if pending.len() < root_children_pending {
            root_children_pending = pending.len();
            if let Some(slot) = root_siblings.and_then(|siblings| pending.len().checked_sub(siblings)).and_then(|slot| branch_starts.get_mut(slot)) {
                *slot = Some(stamps.len());
            }
        }
        variables.resume(pending.len());
        if separate_child_variables && visit.depth == 1 {
            variables.start_root_child();
        }
        max_depth = max_depth.max(visit.depth);
        if empty_switches.contains(&visit.node) {
            continue;
        }
        remaining = remaining.checked_sub(1).ok_or("pattern graph exceeded its execution budget")?;
        if repeats_left != 0 {
            // Resume the previous draw's scope before scheduling its sibling. Descendants of
            // this draw run first and inherit its own number and seed.
            let next = PatternVisit { number: visit.number + 1, seed: crate::random::iteration_child(visit.seed), ..visit.clone() };
            pending.push((next, repeats_left - 1));
        }
        let &(node, literal_geometry, quadrant_children) = nodes.get(&visit.node).ok_or("missing pattern node")?;
        if underfilled.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len())) {
            underfilled = None;
        }
        if seven_level_runs.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len())) {
            seven_level_runs = None;
        }
        if terminal_quadrants.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len())) {
            terminal_quadrants = None;
        }
        if sparse_runs.as_ref().is_some_and(|runs| !runs.visit(node, visit.depth)) {
            sparse_runs = None;
        }
        if mixed_depth_runs.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len())) {
            mixed_depth_runs = None;
        }
        if node.kind == PatternNodeKind::Dot {
            if let Some(Some(child)) = node.children.first() {
                visit.node = *child;
                pending.push((visit, 0));
            }
            continue;
        }

        let definition = catalog::definition(node.kind).ok_or("missing pattern definition")?;
        for (name, value) in [("$depth", visit.depth as f32), ("$depthpow2", visit.extent), ("$number", visit.number as f32)] {
            if let Some(bound) = variables.drawing.get_mut(name) {
                *bound = Value::Float(value);
            } else {
                variables.drawing.insert(name.into(), Value::Float(value));
            }
        }
        let mut declared = Vec::new();

        let mut evaluate = |id: &str, random: &mut FunctionRandom| -> Result<Option<Value>, String> {
            let Some((program, names)) = functions.get(&(node.id, id)) else { return Ok(node.parameters.get(id).cloned()) };
            let mode = quadrant_mode(node.kind, id);
            if !mode {
                variables.enter(names, visit.depth, &mut declared);
            }
            let inputs = if mode { &variables.modes } else { &variables.drawing };
            // An unbound direct link leaves the field at its default. A getter inside an expression still evaluates to its typed zero.
            if program.input_identifier().is_some_and(|name| !name.starts_with('$') && !inputs.contains_key(name)) {
                return Ok(None);
            }
            let evaluation = EvaluationContext {
                image_formats: context.input_formats.get(1..).unwrap_or(&[]),
                scope: crate::function::FunctionScope::FxMap,
                position: if mode { [0.0; 2] } else { visit.center },
                size: [context.w as f32, context.h as f32],
                images: context.inputs.get(1..).unwrap_or(&[]),
                tile: context.tile,
                variables: inputs,
            };
            let value = if node.kind == PatternNodeKind::Quadrant && id == "opacity" {
                program.evaluate_pattern_tint(&evaluation, &mut state, random)?
            } else {
                program.evaluate_with_random(&evaluation, &mut state, random)?
            };
            // A quadrant's mode expressions keep their own writes. A later stamp property can read writes from the draw expressions.
            if !mode {
                for (name, value) in state.assignments() {
                    if let Some(bound) = variables.drawing.get_mut(name) {
                        *bound = value.clone();
                    }
                }
            }
            Ok(Some(value))
        };
        // The seed expression reads the incoming stream before its offset applies to this stamp and its descendants.
        let offset = evaluate("randomseed", &mut FunctionRandom::uniform(visit.seed))?.and_then(|v| v.as_i32()).unwrap_or(0) as u32;
        let inherits = evaluate("randominherited", &mut FunctionRandom::image(visit.seed))?.and_then(|v| v.as_bool()).unwrap_or(true);
        visit.seed = if inherits { visit.seed.wrapping_add(offset) } else { offset };
        parameter_values.clear();
        let mut random = FunctionRandom::image(visit.seed);
        let mut pattern = if node.kind == PatternNodeKind::Quadrant { evaluate("patterntype", &mut random)?.and_then(|v| v.as_i32()).unwrap_or(0) } else { 0 };
        let image_pattern = image_pattern(pattern);
        let cache_key = reusable_leaves.contains(&node.id).then_some(stamp_cache::Key { node: node.id, depth: visit.depth, seed: visit.seed });
        let cached = cache_key.and_then(|key| stamp_cache.get(key));
        let reused = cached.is_some();
        // An unused stamp property cannot publish a Set to a child. The branch offset still positions that child.
        let active = |id: &str| !matches!(id, "patterntype" | "randomseed" | "randominherited") && uses_property(node.kind, id, pattern);
        if !reused {
            full_evaluations = full_evaluations.checked_sub(1).ok_or("pattern graph exceeded its execution budget")?;
            for parameter in definition.params.iter().filter(|p| active(p.id)) {
                let mut stream = FunctionRandom::pattern(visit.seed, parameter.id);
                let value = evaluate(parameter.id, stream.as_mut().unwrap_or(&mut random))?.unwrap_or_else(|| parameter.default.clone());
                parameter_values.push((parameter.id, value));
            }
        }

        let field = |name: &str| parameter_values.iter().find_map(|(id, value)| (*id == name).then_some(value));
        let scalar = |name: &str| field(name).and_then(Value::as_f32).unwrap_or(0.0);
        let integer = |name: &str| field(name).and_then(Value::as_i32).unwrap_or(0);
        let vector = |name: &str| field(name).and_then(Value::as_vec4).map(|value| [value[0], value[1]]).unwrap_or([0.0; 2]);
        children.clear();
        let mut repeated = None;
        let mut discarded_draw = false;
        let track_root_children =
            grayscale && node.kind == PatternNodeKind::Quadrant && visit.depth == 0 && (Some(node.id) == graph.root || opacity_depth == 6);
        match node.kind {
            PatternNodeKind::Dot => {}
            PatternNodeKind::Quadrant => {
                if visit.depth == 0
                    && let Some(active) = &mut repeated_roots
                {
                    // A switch on the repeated branch may omit an entire quadrant tree.
                    *active += 1;
                }
                let offset = cached.as_ref().map_or_else(|| vector("branchoffset"), |properties| properties.branch_offset);
                visit.center = [visit.center[0] + offset[0], visit.center[1] + offset[1]];
                if let Some(properties) = &cached {
                    pattern = properties.stamp.pattern;
                } else if image_pattern {
                    let requested = integer("imageindex");
                    let slot = if functions.get(&(node.id, "imageindex")).is_some_and(|(program, _)| !program.uses_fixed_image_slot(&variables.drawing)) {
                        // Even a stamp with empty coverage establishes the wired order for the other image draws.
                        runtime_image_selector = true;
                        // A runtime selector advances across disconnected slots and stops at the last wired image.
                        let pos = image_slots.partition_point(|&i| i < requested);
                        image_slots.get(pos).or_else(|| image_slots.last()).copied()
                    } else {
                        fixed_image_slot(pattern, requested)
                    };
                    pattern = slot.map_or(0, |i| i.saturating_add(256));
                }
                if pattern != 0 {
                    let mut stamp = if let Some(properties) = cached {
                        properties.stamp
                    } else {
                        let tint = field("opacity").unwrap_or(&Value::Float4([1.0; 4]));
                        let rgba = tint.as_vec4().ok_or("pattern color must be a float or float4")?;
                        let computed_tint = functions.contains_key(&(node.id, "opacity"));
                        if rgba.iter().any(|v| v.is_infinite() || (v.is_nan() && !computed_tint)) {
                            return Err("pattern color must be finite".into());
                        }
                        let color = match tint {
                            Value::Float(luminosity) => [1.0, 1.0, 1.0, *luminosity],
                            _ if grayscale => {
                                // A color expression supplies its alpha. A literal swatch supplies its mean RGB.
                                let luminosity = if computed_tint { rgba[3] } else { (rgba[0] + rgba[1] + rgba[2]) / 3.0 };
                                [1.0, 1.0, 1.0, luminosity]
                            }
                            _ => rgba,
                        };

                        let stamp = PatternStamp {
                            center: vector("frameoffset"),
                            size: vector("patternsize").map(|size| size * visit.extent),
                            rotation: scalar("patternrotation"),
                            literal_geometry,
                            color,
                            depth: visit.depth,
                            pattern,
                            variation: scalar("patternsuppl"),
                            blend: integer("blendingmode"),
                            filtering: integer("imagefiltering"),
                            premultiplied: field("imagepremul").and_then(Value::as_bool).unwrap_or(false),
                        };
                        if let Some(key) = cache_key {
                            stamp_cache.insert(
                                key,
                                stamp_cache::Properties { branch_offset: offset, stamp: stamp.clone() },
                                image_pattern && raster_work.streaming,
                            );
                        }
                        stamp
                    };
                    stamp.center = [visit.center[0] + stamp.center[0], visit.center[1] + stamp.center[1]];
                    // A draw that misses the sample grid still belongs to its grandchild's run.
                    if seven_level_runs.as_mut().is_some_and(|runs| !runs.draw(&stamp)) {
                        seven_level_runs = None;
                    }
                    // Evaluate every property before discarding a stamp. Its assignments can reach
                    // later visits. Keep the first stamp so raster setup still validates depth weights.
                    let missed = (!stamps.is_empty() || raster_work.background.is_some()) && grid.as_ref().is_some_and(|grid| grid.misses(&stamp));
                    if underfilled
                        .as_mut()
                        .is_some_and(|runs| !runs.draw(&stamp, opacities.as_deref().unwrap_or_default(), grid.as_ref(), stamps.len(), !missed))
                    {
                        underfilled = None;
                    }
                    if sparse_runs.as_mut().is_some_and(|runs| runs.draw(&stamp, opacities.as_deref().unwrap_or_default(), !missed).is_none()) {
                        sparse_runs = None;
                    }
                    if mixed_depth_runs.as_mut().is_some_and(|runs| !runs.draw(&stamp, opacities.as_deref().unwrap_or_default(), !missed)) {
                        mixed_depth_runs = None;
                    }
                    if repeated_six_level
                        .as_mut()
                        .is_some_and(|runs| runs.record(&stamp, opacities.as_deref().unwrap_or_default(), grid.as_ref(), !missed).is_none())
                    {
                        repeated_six_level = None;
                    }
                    if terminal_quadrants.as_mut().is_some_and(|runs| !runs.draw(&stamp, opacities.as_deref().unwrap_or_default(), grid.as_ref(), !missed)) {
                        terminal_quadrants = None;
                    }
                    if missed {
                        // These siblings have no children. Each has already evaluated its own
                        // properties and failed the sample-grid test in the original draw order.
                        if let Some(siblings) = &mut terminal_siblings {
                            siblings.discarded = true;
                            reused_terminal_misses = reused_terminal_misses.saturating_add(usize::from(reused));
                            remaining += 1;
                        } else {
                            discarded_draw = true;
                        }
                    } else {
                        // A fully evaluated quadrant schedules at most four terminal children.
                        // Its visit bounds their cache reads when their retained draws are released.
                        accounted_stamps += usize::from(!reused || terminal_siblings.is_some());
                        stamps.push(stamp);
                    }
                }

                if quadrant_children == QuadrantChildren::AllTerminal {
                    terminal_siblings = Some(TerminalSiblings { stack: pending.len(), discarded: false });
                }
                if quadrant_children != QuadrantChildren::None {
                    let mut child_random = visit.seed;
                    for (child, center) in node.children.iter().zip(visit.quadrant_centers()) {
                        let child_seed = crate::random::quadrant_child(&mut child_random, visit.depth.saturating_add(1));
                        if let Some(child) = child {
                            let mut child_visit = visit.clone();
                            child_visit.node = *child;
                            child_visit.depth = visit.depth.saturating_add(1);
                            child_visit.extent *= 0.5;
                            child_visit.seed = child_seed;
                            child_visit.center = center;
                            children.push(child_visit);
                        }
                    }
                }
            }
            PatternNodeKind::Switch => {
                let slot = usize::from(field("switch").and_then(Value::as_bool).unwrap_or(false));
                if let Some(Some(child)) = node.children.get(slot) {
                    visit.node = *child;
                    children.push(visit);
                } else if fixed_switches.contains(&node.id) {
                    empty_switches.insert(node.id);
                }
            }
            PatternNodeKind::Iterate => {
                let count = u32::try_from(integer("numberadded")).map_err(|_| "iteration count cannot be negative")?;

                if grayscale
                    && Some(node.id) == graph.root
                    && node.children.first() == Some(&None)
                    && !graph.nodes.iter().any(|other| other.id != node.id && other.kind == PatternNodeKind::Iterate)
                {
                    repeated_roots = Some(0_usize);
                    if opacity_depth == 6 && count > 1 && !context.b("cull_outside") {
                        repeated_six_level = Some(super::repeated_six_level::RepeatedSixLevelRuns::default());
                    }
                }
                if let Some(Some(child)) = node.children.first() {
                    // The initial branch keeps the enclosing iteration's number and seed.
                    children.push(PatternVisit { node: *child, ..visit.clone() });
                }
                if let Some(repeats_left) = count.checked_sub(1)
                    && let Some(Some(child)) = node.children.get(1)
                    && !empty_switches.contains(child)
                {
                    if visit.depth > 0 && (repeats_left != 0 || !children.is_empty()) {
                        seven_level_runs = None;
                        if !children.is_empty() || mixed_depth_runs.as_mut().is_some_and(|runs| !runs.repeat_parent(visit.depth, repeats_left)) {
                            mixed_depth_runs = None;
                        }
                    }
                    // Each connected visit costs at least one unit, even if its stamp is discarded.
                    // A fixed switch can discover an empty branch on its first visit and skip the rest.
                    if count as usize > remaining + discard_allowance && !fixed_switches.contains(child) {
                        return Err("pattern iteration count exceeds the execution budget".into());
                    }
                    let first = PatternVisit { node: *child, number: 0, seed: crate::random::iteration_child(visit.seed), ..visit };
                    repeated = Some((first, repeats_left));
                }
            }
        }

        if !empty_switches.is_empty() {
            children.retain(|child| {
                if empty_switches.contains(&child.node) {
                    // A cached empty branch still contributes its visited depth to draw ordering.
                    max_depth = max_depth.max(child.depth);
                    false
                } else {
                    true
                }
            });
        }
        // A quadrant with no shape and no live children has finished evaluating its
        // properties. Charge it to the same bounded allowance as a discarded draw.
        let empty_quadrant = node.kind == PatternNodeKind::Quadrant && pattern == 0 && children.is_empty();
        if quadrant_children == QuadrantChildren::None
            && terminal_siblings.as_ref().is_some_and(|siblings| pending.len() == siblings.stack)
            && let Some(siblings) = terminal_siblings.take()
            && siblings.discarded
        {
            // Each skipped property evaluation pays for one group's first discarded draw.
            // Sibling groups can reuse different numbers of stamps. Parents and visible
            // stamps still pay for their own visits, and full evaluations stay bounded.
            if reused_terminal_misses > 0 {
                reused_terminal_misses -= 1;
            } else if discard_allowance > 0 {
                discard_allowance -= 1;
            } else {
                remaining = remaining.checked_sub(1).ok_or("pattern graph exceeded its execution budget")?;
            }
        }
        if discard_allowance > 0 && (discarded_draw || empty_quadrant) {
            discard_allowance -= 1;
            remaining += 1;
        }
        // Once every subtree ordering has been ruled out, equal-ranked stamps can be composited
        // immediately in traversal order. Reuse their storage and preserve all F32 adds.
        if stamps.len() >= stream_chunk
            && max_depth > 6
            && mixed_depth_runs.is_none()
            && terminal_quadrants.is_none()
            && sparse_runs.is_none()
            && underfilled.is_none()
            && seven_level_runs.is_none()
            && repeated_six_level.is_none()
            && *uniform_order.get_or_insert_with(|| uniform_deep_draws(graph.root, &nodes))
        {
            // Other cached draws, including direct repetitions, have a separate bounded allowance.
            let cached_credit = (stamps.len() - accounted_stamps).min(cached_stream_allowance);
            raster_work.streaming = true;
            let group = [super::pattern_stamp::DrawRun::whole(&stamps)];
            let image = raster::render(context, &group, &opacities, nonfinite_gray_opacity, base_level_only, runtime_image_selector, &mut raster_work)?;
            raster_work.background = Some(image);
            cached_stream_allowance -= cached_credit;
            remaining = remaining.saturating_add(accounted_stamps + cached_credit);
            accounted_stamps = 0;
            stamps.clear();
        }
        if children.len() + pending.len() + usize::from(repeated.is_some()) > remaining {
            return Err("pattern graph exceeded its execution budget".into());
        }

        variables.keep_for_children(pending.len(), declared);
        if track_root_children {
            if root_siblings.is_some() {
                // Multiple emitted trees can interleave. A single tree's four child ranges no longer describe the draw.
                branch_starts.fill(None);
                root_children_pending = 0;
            } else {
                // Outer repetitions can remain on the stack while the only emitted tree visits its four children.
                root_siblings = Some(pending.len());
                root_children_pending = pending.len() + children.len();
            }
        }
        pending.extend(repeated);
        pending.extend(children.drain(..).rev().map(|visit| (visit, 0)));
    }

    let mut groups = [&[][..]; SEVEN_LEVEL_RUN_ORDER.len()];
    if let Some(run_starts) = seven_level_runs.and_then(|runs| runs.finish(stamps.len())) {
        for (group, run) in groups.iter_mut().zip(SEVEN_LEVEL_RUN_ORDER) {
            *group = stamps.get(run_starts[run]..run_starts[run + 1]).ok_or("invalid pattern draw range")?;
        }
    } else {
        let mut prefix = stamps.as_slice();
        if max_depth == 6 {
            // At six subdivisions the root draws first, then its child streams composite
            // in reverse slot order. Within a stream the original draw order is retained.
            for (group, start) in groups[1..].iter_mut().zip(branch_starts.into_iter().flatten()) {
                let (earlier, branch) = prefix.split_at(start);
                *group = branch;
                prefix = earlier;
            }
        }
        groups[0] = prefix;
    }
    let groups = groups.map(super::pattern_stamp::DrawRun::whole);
    let mixed_groups = mixed_depth_runs
        .and_then(|runs| runs.finish(&stamps))
        .or_else(|| terminal_quadrants.and_then(|runs| runs.finish(&stamps)))
        .or_else(|| sparse_runs.and_then(|runs| runs.finish(&stamps)))
        .or_else(|| underfilled.and_then(|runs| runs.finish(&stamps)))
        .or_else(|| repeated_six_level.and_then(|runs| runs.finish(&stamps, repeated_roots.unwrap_or_default())))
        .or_else(|| repeated_roots.and_then(|repeats| super::repeated_quadrants::runs(context, &stamps, repeats, max_depth)));
    let image = raster::render(
        context,
        mixed_groups.as_deref().unwrap_or(&groups),
        &opacities,
        nonfinite_gray_opacity,
        base_level_only,
        runtime_image_selector,
        &mut raster_work,
    )?;
    Ok((image, variables.bound_draw))
}
