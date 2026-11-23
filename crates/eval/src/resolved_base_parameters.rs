use materialcraft_core::{BaseParameter, BaseParameters, Graph, OutputFormat, ParameterInheritance, Value};
use materialcraft_nodes::function::compile_with_literal_inputs;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedBaseParameters {
    pub size: [u32; 2],
    /// This texture inherits or evaluates a size controlled by a runtime parameter.
    #[serde(skip)]
    pub dynamic_size: bool,
    /// Negative requested exponents on axes whose allocated texture is one texel.
    #[serde(skip)]
    pub size_underflow: [i32; 2],
    pub seed: u32,
    pub format: OutputFormat,
    pub tiling: [bool; 2],
    pub pixel_ratio: i32,
    pub pixel_size: [f32; 2],
}

// An image carries no random stream. Both relative modes add to the parent seed.
fn resolve_seed(mode: ParameterInheritance, offset: i32, parent: u32) -> u32 {
    mode.resolve(0, None, parent).wrapping_add(offset as u32)
}

impl ResolvedBaseParameters {
    pub(crate) fn graph(graph: &Graph, options: &crate::Options<'_>) -> Self {
        let parent = options.parent_base;
        let parent_ratio = options.parent_pixel_ratio.or_else(|| parent.map(|base| base.pixel_ratio));
        Self {
            size: parent.map_or_else(|| graph.size_px(), |base| graph.output_size_px(base.size)),
            dynamic_size: graph.absolute_output_size.is_none() && parent.is_some_and(|base| base.dynamic_size),
            size_underflow: parent.filter(|_| graph.absolute_output_size.is_none()).map_or([0; 2], |base| base.size_underflow),
            seed: resolve_seed(graph.seed_mode, graph.seed, parent.map_or(0, |base| base.seed)),
            format: parent.map_or(graph.format, |base| base.format),
            tiling: graph.tiling_mode.resolve(graph.tiling, options.parent_tiling, options.parent_tiling.unwrap_or(graph.tiling)),
            pixel_ratio: graph.pixel_ratio_mode.resolve(graph.pixel_ratio, parent_ratio, parent_ratio.unwrap_or(1)),
            pixel_size: parent.map_or([1.0; 2], |base| base.pixel_size),
        }
    }

    pub(crate) fn resolve(base: &BaseParameters, input: Option<Self>, parent: Self) -> Self {
        let dimensions = |base: Self| (base.size, base.size_underflow);
        let size_mode = BaseParameter::OutputSize.inheritance(base);
        let (size_base, underflow) = size_mode.resolve(([1; 2], [0; 2]), input.map(dimensions), dimensions(parent));
        let shift: [i32; 2] = std::array::from_fn(|axis| underflow[axis].saturating_add(base.size[axis].clamp(-12, 13)));

        // Internal images may be one texel; the editor minimum would break size functions
        let size = std::array::from_fn(|axis| (f64::from(size_base[axis]) * 2f64.powi(shift[axis])).round().clamp(1.0, 8192.0) as u32);
        let size_underflow =
            std::array::from_fn(|axis| if size_base[axis].is_power_of_two() { shift[axis].saturating_add(size_base[axis].ilog2() as i32).min(0) } else { 0 });

        let inherited_pixel_size = base.pixel_size_mode.resolve([1.0; 2], input.map(|base| base.pixel_size), parent.pixel_size);
        Self {
            size,
            size_underflow,
            dynamic_size: size_mode.resolve(false, input.map(|base| base.dynamic_size), parent.dynamic_size),
            seed: resolve_seed(base.seed_mode, base.seed, parent.seed),
            format: BaseParameter::Format.inheritance(base).resolve(base.format, input.map(|base| base.format), parent.format),
            tiling: base.tiling_mode.resolve(base.tiling, input.map(|base| base.tiling), parent.tiling),
            pixel_ratio: base.pixel_ratio_mode.resolve(base.pixel_ratio, input.map(|base| base.pixel_ratio), parent.pixel_ratio),
            pixel_size: std::array::from_fn(|axis| (inherited_pixel_size[axis] * base.pixel_size[axis]).clamp(-8192.0, 8192.0)),
        }
    }

    pub(crate) fn variables(self, variables: &mut BTreeMap<String, Value>) {
        self.size_variables(variables);
        variables.insert("$tiling".into(), Value::Int(i32::from(self.tiling[0]) | (i32::from(self.tiling[1]) << 1)));
        variables.insert("$pixelratio".into(), Value::Int(self.pixel_ratio));
    }

    fn size_variables(self, variables: &mut BTreeMap<String, Value>) {
        if self.size_underflow == [0; 2] {
            variables.remove("$sizelog2");
        } else {
            let exponents =
                std::array::from_fn(|axis| if self.size_underflow[axis] < 0 { self.size_underflow[axis] as f32 } else { (self.size[axis] as f32).log2() });
            variables.insert("$sizelog2".into(), Value::Float2(exponents));
        }
    }
}

pub(crate) fn evaluate(
    base: &BaseParameters,
    input: Option<ResolvedBaseParameters>,
    parent: ResolvedBaseParameters,
    variables: &mut std::borrow::Cow<'_, BTreeMap<String, Value>>,
    literal_inputs: &[String],
) -> Result<(ResolvedBaseParameters, BTreeMap<BaseParameter, Value>), String> {
    if base.functions.is_empty() {
        return Ok((ResolvedBaseParameters::resolve(base, input, parent), BTreeMap::new()));
    }

    // Evaluation mutates literals while borrowing the attached function graphs
    let mut evaluated = BaseParameters {
        size_mode: base.size_mode,
        size: base.size,
        format: base.format,
        format_mode: base.format_mode,
        seed: base.seed,
        seed_mode: base.seed_mode,
        pixel_size: base.pixel_size,
        pixel_size_mode: base.pixel_size_mode,
        functions: BTreeMap::new(),
        tiling: base.tiling,
        tiling_mode: base.tiling_mode,
        pixel_ratio: base.pixel_ratio,
        pixel_ratio_mode: base.pixel_ratio_mode,
    };
    let mut values = BTreeMap::new();
    let mut computed_size = false;
    let mut resolved = ResolvedBaseParameters::resolve(&evaluated, input, parent);

    // Size and seed establish the context used by the remaining parameter functions
    for parameter in [
        BaseParameter::RandomSeed,
        BaseParameter::OutputSize,
        BaseParameter::Format,
        BaseParameter::PixelSize,
        BaseParameter::PixelRatio,
        BaseParameter::Tiling,
    ] {
        if let Some(function) = base.functions.get(&parameter) {
            resolved.variables(variables.to_mut());
            let size = if parameter == BaseParameter::OutputSize && parameter.inheritance(base) != ParameterInheritance::Absolute {
                let inherited = parameter.inheritance(base).resolve(parent, input, parent);
                inherited.size_variables(variables.to_mut());
                inherited.size
            } else {
                resolved.size
            };

            let (value, _) = compile_with_literal_inputs(function, resolved.seed, literal_inputs)
                .and_then(|program| {
                    if parameter == BaseParameter::OutputSize {
                        computed_size = !program.output_is_literal() && !program.depends_only_on_image_size();
                    }
                    crate::parameter_function::evaluate(
                        &program,
                        parameter.value_type(),
                        variables.to_mut(),
                        size,
                        crate::parameter_function::ChannelMode::Opaque,
                        None,
                    )
                })
                .map_err(|error| format!("function of `{}`: {error}", parameter.identifier()))?;
            parameter.set_value(&mut evaluated, &value)?;
            parameter.set_inheritance(&mut evaluated, parameter.inheritance(base));
            resolved = ResolvedBaseParameters::resolve(&evaluated, input, parent);
        }

        values.insert(parameter, parameter.value(&evaluated));
    }

    resolved.dynamic_size |= computed_size;
    Ok((resolved, values))
}
