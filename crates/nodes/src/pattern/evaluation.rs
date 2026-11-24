use materialcraft_core::{PatternNode, PatternNodeKind, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::mixed_depth_runs::MixedDepthRuns;
use super::{catalog, footprint::SampleGrid, pattern_stamp::PatternStamp, pattern_visit::PatternVisit, raster, variables::PatternVariables};
use crate::function::{EvaluationContext, EvaluationState, compile_with_literal_inputs};
use crate::random::FunctionRandom;
use crate::{CookContext, CookResult};

// A 512-by-512 quadtree takes 349,525 visits. This limit bounds retained stamps and
// provides a second allowance for discarded draws. A terminal sibling group has at most
// three additional discards, and needs two ordinary visits (its parent and first discard).
// Thus at most five times this limit can execute, including the discarded stamps.
const MAX_VISITS: usize = 1_048_576;

// Overlapping squares establish the order of the sixteen grandchildren in a regular seven-level tree.
const SEVEN_LEVEL_RUN_ORDER: [usize; 16] = [12, 13, 8, 9, 14, 0, 1, 15, 2, 11, 6, 7, 10, 3, 4, 5];

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuadrantChildren {
    None,
    AllTerminal,
    Other,
}

fn has_uniform_quadrant_leaves(root: u32, depth: u32, nodes: &BTreeMap<u32, (&PatternNode, bool, QuadrantChildren)>) -> bool {
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
    cook_with_visit_limit(context, base_level_only, MAX_VISITS)
}

pub(super) fn cook_with_visit_limit(context: &CookContext, base_level_only: &[bool], visit_limit: usize) -> CookResult {
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
    let separate_child_variables = opacity_depth == 6
        && functions.values().any(|(_, names)| !names.is_empty())
        && graph.root.is_some_and(|root| has_uniform_quadrant_leaves(root, opacity_depth, &nodes));
    let mut max_depth = 0;
    let mut state = EvaluationState::default();
    let mut terminal_siblings: Option<(usize, bool)> = None;
    let mut remaining = visit_limit;
    let mut discard_allowance = visit_limit;
    let grayscale = !context.b("color_mode");
    let mut repeated_roots = None;
    let mut repeated_six_level: Option<super::repeated_six_level::RepeatedSixLevelRuns> = None;
    let mut terminal_quadrants = (grayscale && opacity_depth == 8 && !context.b("cull_outside")).then(super::terminal_quadrants::TerminalQuadrants::default);
    let mut seven_level_runs = (grayscale && opacity_depth == 7).then(SevenLevelRuns::default);
    let mut underfilled = (grayscale && opacity_depth >= 8 && !context.b("cull_outside")).then(super::underfilled_quadrants::UnderfilledQuadrants::default);
    let mut sparse_runs = if grayscale && opacity_depth == 8 { super::tile_runs::SparseQuadrantRuns::new(context) } else { None };
    let mut mixed_depth_runs = if grayscale && matches!(opacity_depth, 8 | 9) { MixedDepthRuns::new(context, opacity_depth) } else { None };
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
        if terminal_siblings.is_some_and(|(start, _)| pending.len() < start) {
            terminal_siblings = None;
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
        if underfilled.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len(), Some(node.id) == graph.root)) {
            underfilled = None;
        }
        if seven_level_runs.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len())) {
            seven_level_runs = None;
        }
        if terminal_quadrants.as_mut().is_some_and(|runs| !runs.visit(node, visit.depth, stamps.len(), Some(node.id) == graph.root)) {
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
            let mode =
                node.kind == PatternNodeKind::Quadrant && matches!(id, "patterntype" | "blendingmode" | "imagefiltering" | "imagepremul" | "randominherited");
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
        let image_pattern = pattern == 1 || (256..256 + crate::MAX_IMAGE_INPUTS as i32).contains(&pattern);
        // An unused stamp property cannot publish a Set to a child. The branch offset still positions that child.
        let active = |id: &str| match id {
            "patterntype" | "randomseed" | "randominherited" => false,
            "branchoffset" => true,
            "imageindex" | "imagefiltering" | "imagepremul" => image_pattern,
            _ => node.kind != PatternNodeKind::Quadrant || pattern != 0,
        };
        for parameter in definition.params.iter().filter(|p| active(p.id)) {
            let mut stream = FunctionRandom::pattern(visit.seed, parameter.id);
            let value = evaluate(parameter.id, stream.as_mut().unwrap_or(&mut random))?.unwrap_or_else(|| parameter.default.clone());
            parameter_values.push((parameter.id, value));
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
                let offset = vector("branchoffset");
                visit.center = [visit.center[0] + offset[0], visit.center[1] + offset[1]];
                if image_pattern {
                    let requested = integer("imageindex");
                    let slot = if functions.get(&(node.id, "imageindex")).is_some_and(|(program, _)| !program.uses_fixed_image_slot(&variables.drawing)) {
                        // Even a stamp with empty coverage establishes the wired order for the other image draws.
                        runtime_image_selector = true;
                        // A runtime selector advances across disconnected slots and stops at the last wired image.
                        let pos = image_slots.partition_point(|&i| i < requested);
                        image_slots.get(pos).or_else(|| image_slots.last()).copied()
                    } else {
                        // A fixed zero preserves the packed slot. The generic image pattern starts at slot zero.
                        (requested >= 0).then_some(if requested == 0 { pattern.max(256) - 256 } else { requested })
                    };
                    pattern = slot.map_or(0, |i| i.saturating_add(256));
                }
                if pattern != 0 {
                    let offset = vector("frameoffset");
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
                        center: [visit.center[0] + offset[0], visit.center[1] + offset[1]],
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
                    // A draw that misses the sample grid still belongs to its grandchild's run.
                    if seven_level_runs.as_mut().is_some_and(|runs| !runs.draw(&stamp)) {
                        seven_level_runs = None;
                    }
                    // Evaluate every property before discarding a stamp. Its assignments can reach
                    // later visits. Keep the first stamp so raster setup still validates depth weights.
                    let missed = !stamps.is_empty() && grid.as_ref().is_some_and(|grid| grid.misses(&stamp));
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
                        let later_discard = terminal_siblings.as_mut().is_some_and(|(_, seen)| std::mem::replace(seen, true));
                        if later_discard {
                            remaining += 1;
                        } else {
                            discarded_draw = true;
                        }
                    } else {
                        stamps.push(stamp);
                    }
                }

                if quadrant_children == QuadrantChildren::AllTerminal {
                    terminal_siblings = Some((pending.len(), false));
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
                    // An inner iterator that submits multiple branches breaks the complete quadrant layout.
                    if visit.depth > 0 && (repeats_left != 0 || !children.is_empty()) {
                        seven_level_runs = None;
                        mixed_depth_runs = None;
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
        if discard_allowance > 0 && (discarded_draw || empty_quadrant) {
            discard_allowance -= 1;
            remaining += 1;
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
        .or_else(|| terminal_quadrants.and_then(|runs| runs.finish(&stamps, repeated_roots.unwrap_or(1))))
        .or_else(|| sparse_runs.and_then(|runs| runs.finish(&stamps)))
        .or_else(|| underfilled.and_then(|runs| runs.finish(&stamps)))
        .or_else(|| repeated_six_level.and_then(|runs| runs.finish(&stamps, repeated_roots.unwrap_or_default())))
        .or_else(|| repeated_roots.and_then(|repeats| super::repeated_quadrants::runs(context, &stamps, repeats, max_depth)));
    Ok(vec![raster::render(context, mixed_groups.as_deref().unwrap_or(&groups), opacities, nonfinite_gray_opacity, base_level_only, runtime_image_selector)?])
}
