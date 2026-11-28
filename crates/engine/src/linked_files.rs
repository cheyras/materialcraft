use crate::document_path;
use materialcraft_core::Package;
use std::{collections::BTreeSet, sync::Arc};

pub(crate) fn normalize_references(package: &mut Package, path: &str) {
    for resource in &mut package.resources {
        if !resource.source.is_empty() {
            resource.source = document_path::resolve(path, &resource.source);
        }
    }

    for instance in package.graphs.iter_mut().flat_map(|graph| graph.instances_mut()) {
        if !instance.package.is_empty() {
            instance.package = document_path::resolve(path, &instance.package);
        }
    }
}

impl crate::Session {
    pub(crate) fn load_file_links(&mut self, package: &mut Package) -> Vec<String> {
        for (path, source) in &self.pkg.dependencies {
            package.dependencies.entry(path.clone()).or_insert_with(|| source.clone());
        }
        let pending = references(package);
        self.load_sources(package, pending)
    }

    fn load_sources(&mut self, package: &mut Package, pending: Vec<String>) -> Vec<String> {
        read_references(package, pending, |uri, source| {
            // Only aliases actually read from the catalog are owned by it. A caller can supply its own snapshot.
            self.library_sources.insert(uri.to_owned(), Arc::clone(source));
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn refresh_saved_links(&mut self, path: &str) {
        use materialcraft_formats::sbs::library;
        use std::path::Path;

        let file = Path::new(path);
        let mut changed = BTreeSet::new();
        if let Ok(root) = library::root().canonicalize()
            && file.starts_with(&root)
        {
            // The saved path is canonical. The same file may also be reached through a library symlink.
            let aliases: Vec<_> = self
                .library_sources
                .iter()
                .filter_map(|(uri, owned)| {
                    let relative = library::relative_path(uri).ok()?;
                    (self.linked_sources.get(uri).is_some_and(|source| Arc::ptr_eq(source, owned))
                        && root.join(relative).canonicalize().ok()?.as_path() == file)
                        .then(|| uri.clone())
                })
                .collect();
            if !aliases.is_empty() {
                let mut registry = Package { dependencies: self.linked_sources.clone(), ..Default::default() };
                for uri in &aliases {
                    registry.dependencies.remove(uri);
                }
                // Read the saved bytes in each alias's own directory so nested links and resources keep their base.
                for warning in self.load_sources(&mut registry, aliases.clone()) {
                    self.warn(warning);
                }
                changed.extend(
                    aliases.into_iter().filter(|uri| registry.dependencies.get(uri).is_some_and(|source| self.linked_sources.get(uri) != Some(source))),
                );
                self.linked_sources.extend(registry.dependencies);
            }
        }
        self.refresh_links_from(changed);
    }

    pub(crate) fn discard_source_changes(&mut self, path: &str) {
        self.linked_sources.remove(path);
        self.pkg.dependencies.remove(path);
        for document in self.documents.values_mut() {
            document.pkg.dependencies.remove(path);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut package = self.pkg.clone();

            // Include inactive dependents so closing a source also refreshes their references
            package.graphs.extend(self.documents.values().flat_map(|document| document.pkg.graphs.iter().cloned()));
            package.dependencies = self.linked_sources.clone();
            let pending = references(&package);
            for warning in self.load_sources(&mut package, pending) {
                self.warn(warning);
            }

            self.linked_sources.extend(package.dependencies);
        }

        self.refresh_links_from(BTreeSet::from([path.to_string()]));
    }
}

fn references(package: &Package) -> Vec<String> {
    package.graphs.iter().flat_map(|graph| graph.instances()).filter(|instance| !instance.package.is_empty()).map(|instance| instance.package.clone()).collect()
}

/// Seed the registry with already parsed library packages before traversing dependencies. The
/// registry stays flat so shared dependencies are parsed once and cannot form Arc ownership cycles.
pub(crate) fn load_references(package: &mut Package, pending: Vec<String>) -> Vec<String> {
    read_references(package, pending, |_, _| {})
}

fn read_references(package: &mut Package, mut pending: Vec<String>, mut catalog_read: impl FnMut(&str, &Arc<Package>)) -> Vec<String> {
    let mut visited = BTreeSet::new();
    let mut warnings = Vec::new();
    while let Some(path) = pending.pop() {
        if !visited.insert(path.clone()) {
            continue;
        }

        if let Some(source) = package.dependencies.get(&path) {
            pending.extend(references(source));
            continue;
        }

        let mut read = || -> std::result::Result<Package, String> {
            if path.starts_with("sbs://") {
                let bytes = materialcraft_formats::sbs::library::read(&path)?;
                let read = materialcraft_formats::sbs::read_sbs(&bytes, &path, &crate::file_cmds::linked_files(&path)).map_err(|error| error.to_string())?;
                warnings.extend(read.warnings);
                return Ok(read.pkg);
            }

            #[cfg(target_arch = "wasm32")]
            return Err("external file is unavailable in the browser".into());
            #[cfg(not(target_arch = "wasm32"))]
            {
                let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
                let read = materialcraft_formats::sbs::read_sbs(&bytes, &path, &crate::file_cmds::linked_files(&path)).map_err(|error| error.to_string())?;
                warnings.extend(read.warnings);
                Ok(read.pkg)
            }
        };
        match read() {
            Ok(mut source) => {
                normalize_references(&mut source, &path);
                pending.extend(references(&source));
                let source = Arc::new(source);
                if path.starts_with("sbs://") {
                    catalog_read(&path, &source);
                }
                package.dependencies.insert(path, source);
            }
            Err(error) => warnings.push(format!("Could not resolve {path}: {error}")),
        }
    }

    warnings
}
