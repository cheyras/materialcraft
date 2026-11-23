/// The caller determines which context-dependent variables a function can read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FunctionScope {
    #[default]
    Node,
    /// A graph or node's base random seed, evaluated as a 32-bit word.
    RandomSeed,
    ValueProcessor,
    PixelProcessor,
    FxMap,
}
