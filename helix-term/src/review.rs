//! Code review mode: shows the changes between a base revision and the working
//! copy inline in the editor.
//!
//! * Documents keep their working copy contents, so editing, saving and
//!   language servers work exactly as usual.
//! * Lines removed since the base are shown as subdued virtual lines (no line
//!   numbers, not selectable) with a red gutter marker.
//! * Added/changed lines get a green marker in the `marks` gutter, which
//!   replaces the `diff` gutter in existing views while reviewing.
//! * In normal mode `n`/`p` jump to the next/previous change (shadowing
//!   search-next and paste while reviewing) and `M` marks the changes under the
//!   cursor as reviewed (shown with a `✓` next to the change marker). Reviewed
//!   changes are remembered per branch in the repository's git directory.
//!
//! The base is the merge base of the requested revision and `HEAD`, which is
//! what a pull request against that revision shows.
//!
//! This module is intentionally written like an extension: it only uses
//! generic, public editor primitives (document decorations from
//! `helix_view::annotations::decorations`, the `marks` gutter, editor events
//! and typed commands) and keeps all of its state to itself. It should be
//! straightforward to port to a plugin once the plugin system can call those
//! primitives.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, bail, Context as _};
use helix_core::syntax::{HighlightEvent, Loader, Syntax};
use helix_core::{line_ending::line_end_char_index, Language, Rope, Selection};

use helix_event::register_hook;
use helix_loader::workspace_trust::TrustQuery;
use helix_view::annotations::decorations::{
    ColorOverlay, DecorationLayer, GutterMark, Marker, TextHighlight, VirtualLine, VirtualLines,
};
use helix_view::editor::{ConfigEvent, GutterType};
use helix_view::events::{ConfigDidChange, DocumentDidChange, DocumentDidClose, DocumentDidOpen};
use helix_view::graphics::Style;
use helix_view::{Document, DocumentId, Editor};
use tui::text::{Span, Spans};

use crate::compositor::{self, Compositor};
use crate::job::{self, Callback, Jobs};
use crate::ui::{overlay::overlaid, Picker, PickerColumn, PromptEvent};

/// Name of the decoration layer owned by review mode.
const LAYER: &str = "review";

/// Theme scopes. They fall back to `diff.plus`/`diff.minus` and `ui.virtual`.
const SCOPE_ADDED_MARKER: &str = "diff.plus.gutter";
const SCOPE_REMOVED_MARKER: &str = "diff.minus.gutter";
const SCOPE_REMOVED_TEXT: &str = "ui.virtual.review.removed";
/// Removed lines keep their syntax highlighting, subdued by blending it toward
/// this scope's color (falls back to the removed text color).
const SCOPE_REMOVED_OVERLAY: &str = "ui.virtual.review.removed.overlay";
const REMOVED_OVERLAY_OPACITY: u8 = 50;
const SCOPE_ADDED_COUNT: &str = "diff.plus";
const SCOPE_REMOVED_COUNT: &str = "diff.minus";
const MARKER: &str = "▍";
const REVIEWED_MARKER: &str = "✓";
const SCOPE_REVIEWED: &str = "ui.virtual.review.reviewed";

/// Keybindings that are active while reviewing (same format as the `[keys]` config).
const KEYS: &str = r#"
[normal]
n = ":review-next"
p = ":review-prev"
M = ":review-mark"
"#;

/// Don't diff huge files on every keystroke.
const MAX_DIFF_LINES: usize = 100_000;

struct Review {
    /// What the user asked to review against, for display.
    base_name: String,
    /// The merge base commit.
    base: String,
    /// Root of the git work tree.
    root: PathBuf,
    /// Base versions of the reviewed documents. Documents that are not part of
    /// the repository (or are ignored) are not tracked.
    bases: HashMap<DocumentId, Base>,
    /// The changes of each reviewed document.
    hunks: HashMap<DocumentId, Vec<Hunk>>,
    /// Keys of the changes marked as reviewed.
    reviewed: HashSet<String>,
    /// Where `reviewed` is saved, and the branch it is saved for.
    store: PathBuf,
    branch: String,
    /// Whether the `marks` gutter replaced a `diff` gutter (or was appended).
    replaced_diff_gutter: bool,
}

/// The base version of a document.
struct Base {
    text: Rope,
    /// Syntax highlighting of each line, computed once since the base never changes.
    highlights: Vec<Vec<TextHighlight>>,
}

static STATE: Mutex<Option<Review>> = Mutex::new(None);

fn state() -> MutexGuard<'static, Option<Review>> {
    // the state stays consistent even if a holder panicked
    STATE.lock().unwrap_or_else(|err| err.into_inner())
}

pub(crate) fn register_hooks() {
    register_hook!(move |event: &mut DocumentDidOpen<'_>| {
        let mut state = state();
        if let Some(review) = state.as_mut() {
            let loader = event.editor.syn_loader.load_full();
            if let Some(doc) = event.editor.documents.get_mut(&event.doc) {
                review.track(doc, &loader);
            }
        }
        Ok(())
    });

    register_hook!(move |event: &mut DocumentDidChange<'_>| {
        let mut state = state();
        if let Some(review) = state.as_mut() {
            review.refresh(event.doc);
        }
        Ok(())
    });

    // e.g. `:config-reload` restores the user's gutter layout
    register_hook!(move |event: &mut ConfigDidChange<'_>| {
        let mut state = state();
        if let Some(review) = state.as_mut() {
            review.use_marks_gutter(event.editor);
        }
        Ok(())
    });

    register_hook!(move |event: &mut DocumentDidClose<'_>| {
        let mut state = state();
        if let Some(review) = state.as_mut() {
            review.bases.remove(&event.doc.id());
            review.hunks.remove(&event.doc.id());
        }
        Ok(())
    });
}

impl Review {
    fn start(editor: &mut Editor, requested_base: Option<&str>) -> anyhow::Result<Self> {
        let cwd = helix_stdx::env::current_working_dir();
        let workspace = helix_loader::find_workspace_in(&cwd).0;
        if !editor
            .workspace_trust
            .query(&workspace, TrustQuery::Git)
            .is_trusted()
        {
            bail!(
                "review mode runs git, which requires a trusted workspace (see :workspace-trust)"
            );
        }

        let root = PathBuf::from(git_string(&cwd, &["rev-parse", "--show-toplevel"])?);
        let root = helix_stdx::path::canonicalize(root);
        let base_name = match requested_base {
            Some(base) => base.to_string(),
            None => default_base(&root)?,
        };
        let base = git_string(&root, &["merge-base", &base_name, "HEAD"])
            .with_context(|| format!("no merge base between {base_name} and HEAD"))?;

        let store = review_store_path(&root)?;
        let branch = git_string(&root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
        let mut review = Review {
            base_name,
            base,
            bases: HashMap::new(),
            hunks: HashMap::new(),
            reviewed: load_reviewed(&store, &branch),
            store,
            branch,
            root,
            replaced_diff_gutter: false,
        };
        let loader = editor.syn_loader.load_full();
        for doc in editor.documents.values_mut() {
            review.track(doc, &loader);
        }
        crate::keymap::set_keymap_layer(LAYER, KEYS)?;
        review.use_marks_gutter(editor);
        Ok(review)
    }

    /// Shows the `marks` gutter instead of the `diff` gutter.
    ///
    /// This changes the editor configuration rather than individual views,
    /// which would get reset on every resize or split.
    fn use_marks_gutter(&mut self, editor: &mut Editor) {
        let mut config = editor.config().clone();
        let layout = &mut config.gutters.layout;
        if layout.contains(&GutterType::Marks) {
            return;
        }
        match layout.iter().position(|gutter| *gutter == GutterType::Diff) {
            Some(idx) => {
                layout[idx] = GutterType::Marks;
                self.replaced_diff_gutter = true;
            }
            None => {
                // with a spacer so the markers don't touch the text
                layout.extend([GutterType::Marks, GutterType::Spacer]);
                self.replaced_diff_gutter = false;
            }
        }
        update_config(editor, config);
    }

    fn stop(self, editor: &mut Editor) {
        crate::keymap::remove_keymap_layer(LAYER);
        for doc in editor.documents.values_mut() {
            doc.decorations_mut().remove_layer(LAYER);
        }
        let mut config = editor.config().clone();
        let layout = &mut config.gutters.layout;
        if self.replaced_diff_gutter {
            for gutter in layout.iter_mut().filter(|g| **g == GutterType::Marks) {
                *gutter = GutterType::Diff;
            }
        } else if let Some(idx) = layout.iter().position(|g| *g == GutterType::Marks) {
            layout.remove(idx);
            if layout.get(idx) == Some(&GutterType::Spacer) {
                layout.remove(idx);
            }
        }
        update_config(editor, config);
    }

    /// Loads the base version of `doc` and decorates it.
    fn track(&mut self, doc: &mut Document, loader: &Loader) {
        let Some(text) = doc.path().and_then(|path| self.load_base(path)) else {
            return;
        };
        let language = doc
            .syntax()
            .map(|syntax| syntax.root_language())
            .or_else(|| {
                doc.language_name()
                    .and_then(|name| loader.language_for_name(name))
            });
        let highlights = language
            .map(|language| highlight_lines(&text, language, loader))
            .unwrap_or_default();
        self.bases.insert(doc.id(), Base { text, highlights });
        self.refresh(doc);
    }

    /// Returns the contents of `path` at the base revision, an empty rope for
    /// files added since then and `None` for files that aren't part of the review.
    fn load_base(&self, path: &Path) -> Option<Rope> {
        load_base(&self.root, &self.base, path)
    }

    /// Recomputes the decorations of `doc` from its base contents.
    fn refresh(&mut self, doc: &mut Document) {
        let Some(Base {
            text: base,
            highlights,
        }) = self.bases.get(&doc.id())
        else {
            return;
        };
        let text = doc.text();
        if base.len_lines() > MAX_DIFF_LINES || text.len_lines() > MAX_DIFF_LINES {
            doc.decorations_mut().remove_layer(LAYER);
            self.hunks.remove(&doc.id());
            return;
        }
        let rel = doc
            .path()
            .and_then(|path| path.strip_prefix(&self.root).ok())
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();

        let (mut added_lines, mut removed_lines, mut reviewed_hunks) = (0, 0, 0);
        let mut layer = DecorationLayer::default();
        let mut hunks = Vec::new();
        for (before, after, key) in keyed_hunks(&rel, base, text) {
            let reviewed = self.reviewed.contains(&key);
            reviewed_hunks += usize::from(reviewed);
            added_lines += after.len();
            removed_lines += before.len();

            // the reviewed marker goes next to the first row of the hunk: the
            // first removed line if there is one, otherwise the first added line
            let reviewed_marker =
                || Marker::new(if reviewed { REVIEWED_MARKER } else { " " }, SCOPE_REVIEWED);
            if !before.is_empty() {
                // Virtual lines can only go *below* a line, so removals at the
                // very top of the file are shown below the first line.
                layer.virtual_lines.push(VirtualLines {
                    after_line: after.start.saturating_sub(1),
                    lines: before
                        .clone()
                        .map(|line| {
                            let mut markers = vec![Marker::new(MARKER, SCOPE_REMOVED_MARKER)];
                            if line == before.start {
                                markers.push(reviewed_marker());
                            }
                            VirtualLine {
                                text: line_text(base, line),
                                scope: SCOPE_REMOVED_TEXT.into(),
                                highlights: highlights.get(line).cloned().unwrap_or_default(),
                                overlay: Some(ColorOverlay {
                                    scope: SCOPE_REMOVED_OVERLAY.into(),
                                    opacity: REMOVED_OVERLAY_OPACITY,
                                }),
                                markers,
                            }
                        })
                        .collect(),
                });
            }
            for line in after.clone() {
                layer.gutter_marks.push(GutterMark {
                    line,
                    marker: Marker::new(MARKER, SCOPE_ADDED_MARKER),
                });
                if before.is_empty() && line == after.start {
                    layer.gutter_marks.push(GutterMark {
                        line,
                        marker: reviewed_marker(),
                    });
                }
            }
            hunks.push(Hunk { after, key });
        }
        if !hunks.is_empty() {
            layer.status = vec![
                Marker::new(format!("+{added_lines}"), SCOPE_ADDED_COUNT),
                Marker::new(format!("-{removed_lines}"), SCOPE_REMOVED_COUNT),
                Marker::new(
                    format!("{REVIEWED_MARKER}{reviewed_hunks}/{}", hunks.len()),
                    SCOPE_REVIEWED,
                ),
            ];
        }
        doc.decorations_mut().set_layer(LAYER, layer);
        self.hunks.insert(doc.id(), hunks);
    }

    /// Marks the hunks with the given keys as reviewed (or not) and saves that.
    fn set_reviewed(&mut self, keys: Vec<String>, reviewed: bool) -> anyhow::Result<()> {
        for key in keys {
            if reviewed {
                self.reviewed.insert(key);
            } else {
                self.reviewed.remove(&key);
            }
        }
        save_reviewed(&self.store, &self.branch, &self.reviewed)
    }

    /// Files changed between the base and the working tree (including untracked files).
    fn changed_files(&self) -> anyhow::Result<Vec<ChangedFile>> {
        let mut files = Vec::new();
        let status = git(
            &self.root,
            &["diff", "--name-status", "--no-renames", "-z", &self.base],
        )?;
        let numstat = git(
            &self.root,
            &["diff", "--numstat", "--no-renames", "-z", &self.base],
        )?;
        // `<added>\t<removed>\t<path>\0`, with `-` counts for binary files
        let mut line_counts = HashMap::new();
        for entry in numstat.split(|&b| b == 0).filter(|entry| !entry.is_empty()) {
            let mut fields = entry.splitn(3, |&b| b == b'\t');
            if let (Some(added), Some(removed), Some(path)) =
                (fields.next(), fields.next(), fields.next())
            {
                let parse = |n: &[u8]| std::str::from_utf8(n).ok()?.parse::<usize>().ok();
                line_counts.insert(path.to_vec(), parse(added).zip(parse(removed)));
            }
        }

        let mut fields = status.split(|&b| b == 0).filter(|field| !field.is_empty());
        while let (Some(status), Some(path)) = (fields.next(), fields.next()) {
            files.push(ChangedFile {
                status: match status.first() {
                    Some(b'A') => FileStatus::Added,
                    Some(b'D') => FileStatus::Deleted,
                    _ => FileStatus::Modified,
                },
                line_counts: line_counts.get(path).copied().flatten(),
                progress: None,
                path: self.root.join(String::from_utf8_lossy(path).as_ref()),
            });
        }
        let untracked = git(
            &self.root,
            &["ls-files", "--others", "--exclude-standard", "-z"],
        )?;
        files.extend(
            untracked
                .split(|&b| b == 0)
                .filter(|path| !path.is_empty())
                .map(|path| {
                    let path = self.root.join(String::from_utf8_lossy(path).as_ref());
                    ChangedFile {
                        status: FileStatus::Added,
                        line_counts: count_lines(&path).map(|lines| (lines, 0)),
                        progress: None,
                        path,
                    }
                }),
        );
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(files)
    }
}

/// Returns the contents of `path` at revision `base`, an empty rope for files
/// added since then and `None` for files that aren't part of the review.
fn load_base(root: &Path, base: &str, path: &Path) -> Option<Rope> {
    let rel = path.strip_prefix(root).ok()?;
    let rel = rel.to_str()?;
    match git(root, &["show", &format!("{base}:{rel}")]) {
        Ok(contents) => Some(Rope::from(String::from_utf8_lossy(&contents).as_ref())),
        // not in the base revision: a new file, unless git ignores it
        Err(_) if git(root, &["check-ignore", "-q", rel]).is_err() => Some(Rope::new()),
        Err(_) => None,
    }
}

/// Syntax highlights of each line of `text`, with ranges relative to the line start.
fn highlight_lines(text: &Rope, language: Language, loader: &Loader) -> Vec<Vec<TextHighlight>> {
    let mut lines = vec![Vec::new(); text.len_lines()];
    let source = text.slice(..);
    let Ok(syntax) = Syntax::new(source, language, loader) else {
        return lines;
    };
    let scope_names: Vec<Arc<str>> = loader.scopes().iter().map(|s| s.as_str().into()).collect();
    let mut highlighter = syntax.highlighter(source, loader, ..);
    let mut scopes: Arc<[Arc<str>]> = Arc::new([]);
    let mut pos = 0;
    loop {
        let next = highlighter.next_event_offset();
        let end = if next == u32::MAX {
            text.len_bytes()
        } else {
            next as usize
        };
        // split the highlighted range at line boundaries
        let mut start = pos;
        while !scopes.is_empty() && start < end {
            let line = text.byte_to_line(start);
            let line_start = text.line_to_byte(line);
            let line_end = text.line_to_byte(line + 1).min(end);
            lines[line].push(TextHighlight {
                range: start - line_start..line_end - line_start,
                scopes: scopes.clone(),
            });
            start = line_end;
        }
        if next == u32::MAX {
            break;
        }
        let (event, highlights) = highlighter.advance();
        let mut stack = match event {
            HighlightEvent::Refresh => Vec::new(),
            HighlightEvent::Push => scopes.to_vec(),
        };
        stack.extend(highlights.map(|highlight| scope_names[highlight.idx()].clone()));
        scopes = stack.into();
        pos = end;
    }
    lines
}

/// Number of lines in a text file, `None` for binary or unreadable files.
fn count_lines(path: &Path) -> Option<usize> {
    let contents = std::fs::read(path).ok()?;
    if contents.contains(&0) {
        return None;
    }
    let newlines = contents.iter().filter(|&&b| b == b'\n').count();
    Some(newlines + usize::from(contents.last().is_some_and(|&b| b != b'\n')))
}

/// The branch a review compares against when none is given: the remote's
/// default branch if known, otherwise `main` or `master`.
fn default_base(root: &Path) -> anyhow::Result<String> {
    if let Ok(remote_head) = git_string(
        root,
        &[
            "rev-parse",
            "--abbrev-ref",
            "--verify",
            "--quiet",
            "origin/HEAD",
        ],
    ) {
        return Ok(remote_head);
    }
    ["main", "master"]
        .into_iter()
        .find(|branch| git(root, &["rev-parse", "--verify", "--quiet", branch]).is_ok())
        .map(String::from)
        .ok_or_else(|| anyhow!("could not determine the base branch, pass one to :review"))
}

fn update_config(editor: &Editor, config: helix_view::editor::Config) {
    // applied (and views refreshed) by the application on the next event loop iteration
    let _ = editor
        .config_events
        .0
        .send(ConfigEvent::Update(Box::new(config)));
}

fn git(dir: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .context("failed to run git")?;
    if !output.status.success() {
        bail!(
            "git {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn git_string(dir: &Path, args: &[&str]) -> anyhow::Result<String> {
    Ok(String::from_utf8(git(dir, args)?)?.trim().to_string())
}

#[derive(Debug, Clone, Copy)]
enum FileStatus {
    Added,
    Modified,
    Deleted,
}

struct ChangedFile {
    status: FileStatus,
    /// Added and removed lines, `None` for binary files.
    line_counts: Option<(usize, usize)>,
    /// Reviewed and total changes (hunks), if known.
    progress: Option<(usize, usize)>,
    path: PathBuf,
}

struct PickerData {
    root: PathBuf,
    added: Style,
    modified: Style,
    deleted: Style,
    reviewed: Style,
}

fn changed_files_picker(
    editor: &Editor,
    files: Vec<ChangedFile>,
    root: PathBuf,
    base: String,
) -> Picker<ChangedFile, PickerData> {
    let preview_root = root.clone();
    let preview_cache: RefCell<HashMap<PathBuf, Option<Arc<Vec<Spans<'static>>>>>> =
        RefCell::default();
    let columns = [
        PickerColumn::new("change", |file: &ChangedFile, data: &PickerData| {
            match file.status {
                FileStatus::Added => Span::styled("+ added", data.added),
                FileStatus::Modified => Span::styled("~ modified", data.modified),
                FileStatus::Deleted => Span::styled("- deleted", data.deleted),
            }
            .into()
        }),
        PickerColumn::new("lines", |file: &ChangedFile, data: &PickerData| {
            match file.line_counts {
                Some((added, removed)) => Spans::from(vec![
                    Span::styled(format!("+{added}"), data.added),
                    Span::raw(" "),
                    Span::styled(format!("-{removed}"), data.deleted),
                ]),
                None => Spans::from("binary"),
            }
            .into()
        }),
        PickerColumn::new("reviewed", |file: &ChangedFile, data: &PickerData| {
            match file.progress {
                Some((done, total)) if total > 0 => {
                    let percent = done * 100 / total;
                    let style = if done == total {
                        data.added
                    } else {
                        data.reviewed
                    };
                    let check = if done > 0 { REVIEWED_MARKER } else { " " };
                    Span::styled(format!("{check} {done}/{total} {percent}%"), style)
                }
                _ => Span::raw(""),
            }
            .into()
        })
        .without_filtering(),
        PickerColumn::new("path", |file: &ChangedFile, data: &PickerData| {
            file.path
                .strip_prefix(&data.root)
                .unwrap_or(&file.path)
                .display()
                .to_string()
                .into()
        }),
    ];
    let data = PickerData {
        root,
        added: editor.theme.get("diff.plus"),
        modified: editor.theme.get("diff.delta"),
        deleted: editor.theme.get("diff.minus"),
        reviewed: editor.theme.get(SCOPE_REVIEWED),
    };
    Picker::new(columns, 3, files, data, |cx, file: &ChangedFile, action| {
        // Deleted files open as an empty buffer, which shows the whole file as removed.
        if let Err(err) = cx.editor.open(&file.path, action) {
            cx.editor.set_error(format!("{err}"));
        }
    })
    .with_preview(|_editor, file| Some((file.path.as_path().into(), None)))
    .with_text_preview(move |editor, file| {
        preview_cache
            .borrow_mut()
            .entry(file.path.clone())
            .or_insert_with(|| {
                file_diff_preview(editor, &preview_root, &base, &file.path).map(Arc::new)
            })
            .clone()
    })
}

/// Lines of unchanged context shown around each change in the picker preview.
const PREVIEW_CONTEXT: usize = 3;

/// A compact diff of `path` (unchanged regions collapsed), styled like the
/// editor shows it while reviewing. `None` for binary or unreadable files.
fn file_diff_preview(
    editor: &Editor,
    root: &Path,
    base: &str,
    path: &Path,
) -> Option<Vec<Spans<'static>>> {
    file_diff_rows(editor, root, base, path).map(|diff| diff.rows)
}

/// The rows of a compact diff, and which line each row shows.
struct DiffRows {
    rows: Vec<Spans<'static>>,
    /// `None` for separators, otherwise the kind of row and its line (in the
    /// base for removed lines, in the working copy otherwise)
    keys: Vec<Option<(Row, usize)>>,
}

/// The working copy contents of `path`: the open buffer if there is one (with
/// unsaved changes), otherwise the file on disk. Empty for deleted files and
/// `None` for binary files.
fn working_copy(open: Option<&Rope>, path: &Path) -> Option<Rope> {
    if let Some(text) = open {
        return Some(text.clone());
    }
    match std::fs::read(path) {
        Ok(contents) if contents.contains(&0) => None,
        Ok(contents) => Some(Rope::from(String::from_utf8_lossy(&contents).as_ref())),
        Err(_) => Some(Rope::new()),
    }
}

fn line_text(text: &Rope, line: usize) -> String {
    text.slice(text.line_to_char(line)..line_end_char_index(&text.slice(..), line))
        .to_string()
}

fn file_diff_rows(editor: &Editor, root: &Path, base: &str, path: &Path) -> Option<DiffRows> {
    let old = load_base(root, base, path)?;
    let new = working_copy(editor.document_by_path(path).map(|doc| doc.text()), path)?;
    let loader = editor.syn_loader.load();
    let language = loader.language_for_filename(path);
    let highlight = |text: &Rope| {
        language
            .map(|language| highlight_lines(text, language, &loader))
            .unwrap_or_default()
    };
    let (old_highlights, new_highlights) = (highlight(&old), highlight(&new));

    let theme = &editor.theme;
    let styles = PreviewStyles {
        text: theme.get("ui.text"),
        linenr: theme.get("ui.linenr"),
        added_marker: theme.get(SCOPE_ADDED_MARKER),
        removed_marker: theme.get(SCOPE_REMOVED_MARKER),
        removed_text: theme.get("ui.text").patch(theme.get(SCOPE_REMOVED_TEXT)),
        removed_overlay: ColorOverlay {
            scope: SCOPE_REMOVED_OVERLAY.into(),
            opacity: REMOVED_OVERLAY_OPACITY,
        },
    };
    let new_lines = visible_line_count(&new);
    let mut preview = DiffPreview {
        rows: Vec::new(),
        keys: Vec::new(),
        width: new_lines.max(1).to_string().len(),
        styles,
        theme,
    };

    let hunks = helix_core::diff::line_hunks(&old, &new);
    let mut shown = 0; // lines of `new` shown (or skipped) so far
    for (i, (before, after)) in hunks.iter().enumerate() {
        let before = before.start as usize..before.end as usize;
        let mut after = after.start as usize..after.end as usize;
        if new.len_chars() == 0 {
            after = 0..0; // a deleted file has no (empty) added line
        }
        let start = after.start.saturating_sub(PREVIEW_CONTEXT).max(shown);
        preview.separator(start - shown);
        for line in start..after.start {
            preview.line(Some(line), Row::Context, &new, &new_highlights, line);
        }
        for line in before {
            preview.line(None, Row::Removed, &old, &old_highlights, line);
        }
        for line in after.clone() {
            preview.line(Some(line), Row::Added, &new, &new_highlights, line);
        }
        let next_start = hunks
            .get(i + 1)
            .map_or(new_lines, |(_, after)| after.start as usize);
        let end = (after.end + PREVIEW_CONTEXT).min(next_start).min(new_lines);
        for line in after.end..end {
            preview.line(Some(line), Row::Context, &new, &new_highlights, line);
        }
        shown = end.max(after.end);
    }
    if preview.rows.is_empty() {
        let linenr = preview.styles.linenr;
        preview
            .rows
            .push(Spans::from(Span::styled("No changes", linenr)));
        preview.keys.push(None);
    } else {
        preview.separator(new_lines.saturating_sub(shown));
    }
    Some(DiffRows {
        rows: preview.rows,
        keys: preview.keys,
    })
}

struct PreviewStyles {
    text: Style,
    linenr: Style,
    added_marker: Style,
    removed_marker: Style,
    removed_text: Style,
    removed_overlay: ColorOverlay,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Row {
    Context,
    Added,
    Removed,
}

struct DiffPreview<'a> {
    rows: Vec<Spans<'static>>,
    keys: Vec<Option<(Row, usize)>>,
    /// width of the line number column
    width: usize,
    styles: PreviewStyles,
    theme: &'a helix_view::Theme,
}

impl DiffPreview<'_> {
    fn separator(&mut self, skipped: usize) {
        if skipped == 0 {
            return;
        }
        let plural = if skipped == 1 { "" } else { "s" };
        self.keys.push(None);
        self.rows.push(Spans::from(vec![
            Span::styled(
                format!("{:>width$}   ", "⋯", width = self.width),
                self.styles.linenr,
            ),
            Span::styled(
                format!("{skipped} unchanged line{plural}"),
                self.styles.linenr,
            ),
        ]));
    }

    fn line(
        &mut self,
        number: Option<usize>,
        row: Row,
        text: &Rope,
        highlights: &[Vec<TextHighlight>],
        line: usize,
    ) {
        let styles = &self.styles;
        let number = number.map_or(String::new(), |n| (n + 1).to_string());
        let mut spans = vec![
            Span::styled(
                format!("{number:>width$} ", width = self.width),
                styles.linenr,
            ),
            match row {
                Row::Context => Span::raw(" "),
                Row::Added => Span::styled(MARKER, styles.added_marker),
                Row::Removed => Span::styled(MARKER, styles.removed_marker),
            },
            Span::raw(" "),
        ];

        let removed = row == Row::Removed;
        let base_style = if removed {
            styles.removed_text
        } else {
            styles.text
        };
        let finish = |style: Style| {
            if removed {
                styles.removed_overlay.apply(style, self.theme)
            } else {
                style
            }
        };
        let line_text = line_text(text, line);
        let mut push = |range: std::ops::Range<usize>, style: Style| {
            if let Some(part) = line_text.get(range).filter(|part| !part.is_empty()) {
                spans.push(Span::styled(part.replace('\t', "    "), finish(style)));
            }
        };
        let mut pos = 0;
        for highlight in highlights.get(line).into_iter().flatten() {
            let start = highlight.range.start.clamp(pos, line_text.len());
            let end = highlight.range.end.clamp(start, line_text.len());
            push(pos..start, base_style);
            let style = highlight.scopes.iter().fold(base_style, |style, scope| {
                style.patch(self.theme.get(scope))
            });
            push(start..end, style);
            pos = end;
        }
        push(pos..line_text.len(), base_style);
        self.rows.push(spans.into());
        self.keys.push(Some((row, line)));
    }
}

/// Number of lines, not counting the empty "line" after a trailing line break.
fn visible_line_count(text: &Rope) -> usize {
    let lines = text.len_lines();
    if text.len_chars() > 0 && text.line(lines - 1).len_chars() == 0 {
        lines - 1
    } else {
        lines
    }
}

/// Computes the review progress of `files` (in the background) and opens the
/// changed files picker.
fn push_files_picker(
    editor: &Editor,
    jobs: &mut Jobs,
    mut files: Vec<ChangedFile>,
    root: PathBuf,
    base: String,
    reviewed: HashSet<String>,
) {
    // buffers may have unsaved changes
    let open_texts: HashMap<PathBuf, Rope> = editor
        .documents()
        .filter_map(|doc| Some((doc.path()?.to_path_buf(), doc.text().clone())))
        .collect();
    jobs.callback(async move {
        let (files, root, base) = tokio::task::spawn_blocking(move || {
            for file in &mut files {
                file.progress = review_progress(&root, &base, &file.path, &open_texts, &reviewed);
            }
            (files, root, base)
        })
        .await?;
        // the picker isn't `Send`, so it is created in the callback
        let call: job::Callback = Callback::EditorCompositor(Box::new(
            move |editor: &mut Editor, compositor: &mut Compositor| {
                let picker = changed_files_picker(editor, files, root, base);
                compositor.push(Box::new(overlaid(picker)));
            },
        ));
        Ok(call)
    });
}

/// Reviewed and total changes of the file at `path`.
fn review_progress(
    root: &Path,
    base: &str,
    path: &Path,
    open_texts: &HashMap<PathBuf, Rope>,
    reviewed: &HashSet<String>,
) -> Option<(usize, usize)> {
    let old = load_base(root, base, path)?;
    let new = working_copy(open_texts.get(path), path)?;
    if old.len_lines() > MAX_DIFF_LINES || new.len_lines() > MAX_DIFF_LINES {
        return None;
    }
    let rel = path
        .strip_prefix(root)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    let hunks = keyed_hunks(&rel, &old, &new);
    let done = hunks
        .iter()
        .filter(|(_, _, key)| reviewed.contains(key))
        .count();
    Some((done, hunks.len()))
}

/// Starts (or restarts) a review against `base` and opens the changed files picker.
fn start_review(editor: &mut Editor, jobs: &mut Jobs, base: Option<&str>) -> anyhow::Result<()> {
    let (files, root, base) = start_review_quietly(editor, base)?;
    let reviewed = state()
        .as_ref()
        .map(|review| review.reviewed.clone())
        .unwrap_or_default();
    push_files_picker(editor, jobs, files, root, base, reviewed);
    Ok(())
}

/// Starts (or restarts) a review, returning the changed files, the repository
/// root and the base commit.
fn start_review_quietly(
    editor: &mut Editor,
    base: Option<&str>,
) -> anyhow::Result<(Vec<ChangedFile>, PathBuf, String)> {
    let mut state = state();
    if let Some(old) = state.take() {
        old.stop(editor);
    }
    let review = Review::start(editor, base)?;
    let files = review.changed_files().unwrap_or_default();
    editor.set_status(format!(
        "Reviewing {} changed files against {} ({})",
        files.len(),
        review.base_name,
        &review.base[..review.base.len().min(10)],
    ));
    let (root, base) = (review.root.clone(), review.base.clone());
    *state = Some(review);
    Ok((files, root, base))
}

/// Opens the changed files picker of the current review, starting a review
/// against the default base if there is none.
pub(crate) fn open_picker(editor: &mut Editor, jobs: &mut Jobs) -> anyhow::Result<()> {
    let state = state();
    let Some(review) = state.as_ref() else {
        drop(state);
        return start_review(editor, jobs, None);
    };
    let (files, root, base, reviewed) = (
        review.changed_files()?,
        review.root.clone(),
        review.base.clone(),
        review.reviewed.clone(),
    );
    drop(state);
    push_files_picker(editor, jobs, files, root, base, reviewed);
    Ok(())
}

// Diff search

/// An added or removed line of the review.
struct DiffLine {
    path: PathBuf,
    /// `Row::Added` or `Row::Removed`
    row: Row,
    /// The line in its version of the file (the base for removed lines).
    line: usize,
    /// The line of the working copy to jump to. Removed lines are shown just
    /// above this line.
    target: usize,
    text: String,
}

/// Collects the added and removed lines of `files` (blank lines are skipped).
fn collect_diff_lines(
    root: &Path,
    base: &str,
    files: &[ChangedFile],
    open_texts: &HashMap<PathBuf, Rope>,
) -> Vec<DiffLine> {
    let mut lines = Vec::new();
    for file in files {
        let path = &file.path;
        let Some(old) = load_base(root, base, path) else {
            continue;
        };
        let Some(new) = working_copy(open_texts.get(path), path) else {
            continue;
        };
        let last_line = visible_line_count(&new).saturating_sub(1);
        for (before, after) in helix_core::diff::line_hunks(&old, &new) {
            let target = (after.start as usize).min(last_line);
            let mut push = |row: Row, text: &Rope, line: usize, target: usize| {
                let text = line_text(text, line);
                if !text.trim().is_empty() {
                    lines.push(DiffLine {
                        path: path.clone(),
                        row,
                        line,
                        target,
                        text,
                    });
                }
            };
            for line in before.start as usize..before.end as usize {
                push(Row::Removed, &old, line, target);
            }
            if new.len_chars() > 0 {
                for line in after.start as usize..after.end as usize {
                    push(Row::Added, &new, line, line);
                }
            }
        }
    }
    lines
}

struct SearchData {
    root: PathBuf,
    added: Style,
    removed: Style,
}

fn diff_search_picker(
    editor: &Editor,
    lines: Vec<DiffLine>,
    root: PathBuf,
    base: String,
    query: String,
) -> Picker<DiffLine, SearchData> {
    let columns = [
        PickerColumn::new("change", |line: &DiffLine, data: &SearchData| {
            match line.row {
                Row::Removed => Span::styled("-", data.removed),
                _ => Span::styled("+", data.added),
            }
            .into()
        })
        .without_filtering(),
        PickerColumn::new("path", |line: &DiffLine, data: &SearchData| {
            let path = line.path.strip_prefix(&data.root).unwrap_or(&line.path);
            format!("{}:{}", path.display(), line.line + 1).into()
        }),
        PickerColumn::new("code", |line: &DiffLine, _: &SearchData| {
            line.text.trim().replace('\t', " ").into()
        }),
    ];
    let data = SearchData {
        root: root.clone(),
        added: editor.theme.get("diff.plus"),
        removed: editor.theme.get("diff.minus"),
    };

    // compact diffs of the files, built on first use
    let diffs: RefCell<HashMap<PathBuf, Option<Arc<DiffRows>>>> = RefCell::default();
    let highlight = editor
        .theme
        .try_get("ui.highlight")
        .unwrap_or_else(|| editor.theme.get("ui.selection"));

    Picker::new(columns, 2, lines, data, |cx, line: &DiffLine, action| {
        let doc = match cx.editor.open(&line.path, action) {
            Ok(id) => doc_mut!(cx.editor, &id),
            Err(err) => {
                cx.editor.set_error(format!("{err}"));
                return;
            }
        };
        let view = view_mut!(cx.editor);
        let text = doc.text();
        let pos = text.line_to_char(line.target.min(text.len_lines().saturating_sub(1)));
        doc.set_selection(view.id, Selection::point(pos));
        if action.align_view(view, doc.id()) {
            helix_view::align_view(doc, view, helix_view::Align::Center);
        }
    })
    .with_text_preview(move |editor, line| {
        let diff = diffs
            .borrow_mut()
            .entry(line.path.clone())
            .or_insert_with(|| file_diff_rows(editor, &root, &base, &line.path).map(Arc::new))
            .clone()?;
        // show the matching row (highlighted) with some context above it
        let idx = diff
            .keys
            .iter()
            .position(|key| *key == Some((line.row, line.line)))?;
        let start = idx.saturating_sub(PREVIEW_CONTEXT + 2);
        let mut rows = diff.rows[start..].to_vec();
        for span in &mut rows[idx - start].0 {
            span.style = span.style.patch(highlight);
        }
        Some(Arc::new(rows))
    })
    .with_initial_query(query, editor)
    .truncate_end()
}

/// `:review-search [query]`: search the added and removed lines of the review.
pub(crate) fn review_search(
    cx: &mut compositor::Context,
    args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event != PromptEvent::Validate {
        return Ok(());
    }
    let query = args.into_iter().collect::<Vec<_>>().join(" ");

    open_search(cx.editor, cx.jobs, query)
}

/// Opens the diff search picker with `query`, starting a review against the
/// default base if there is none.
pub(crate) fn open_search(
    editor: &mut Editor,
    jobs: &mut Jobs,
    query: String,
) -> anyhow::Result<()> {
    let state = state();
    let (files, root, base) = match state.as_ref() {
        Some(review) => (
            review.changed_files()?,
            review.root.clone(),
            review.base.clone(),
        ),
        None => {
            drop(state);
            start_review_quietly(editor, None)?
        }
    };
    // buffers may have unsaved changes
    let open_texts: HashMap<PathBuf, Rope> = editor
        .documents()
        .filter_map(|doc| Some((doc.path()?.to_path_buf(), doc.text().clone())))
        .collect();

    jobs.callback(async move {
        let (lines, root, base) = tokio::task::spawn_blocking(move || {
            let lines = collect_diff_lines(&root, &base, &files, &open_texts);
            (lines, root, base)
        })
        .await?;
        let call: job::Callback = Callback::EditorCompositor(Box::new(
            move |editor: &mut Editor, compositor: &mut Compositor| {
                let picker = diff_search_picker(editor, lines, root, base, query);
                compositor.push(Box::new(overlaid(picker)));
            },
        ));
        Ok(call)
    });
    Ok(())
}

// Reviewed state

/// A change in a reviewed document.
struct Hunk {
    /// Lines in the working copy (empty for pure removals, which are shown
    /// just above `after.start`).
    after: std::ops::Range<usize>,
    /// Identifies the change across sessions: a hash of the file path and the
    /// removed and added text, so it survives unrelated edits (but not edits
    /// of the change itself).
    key: String,
}

impl Hunk {
    /// Whether the hunk is shown on (or, for removals, right next to) `line`.
    fn touches(&self, lines: std::ops::RangeInclusive<usize>) -> bool {
        if self.after.is_empty() {
            // removed lines are shown between these two lines
            let (above, below) = (self.after.start.saturating_sub(1), self.after.start);
            lines.contains(&above) || lines.contains(&below)
        } else {
            *lines.start() < self.after.end && self.after.start <= *lines.end()
        }
    }
}

/// The changes between `base` and `text` (of the file at repository path
/// `rel`) as `(base lines, working copy lines, key)`. The key identifies the
/// change for remembering whether it was reviewed, see [`Hunk::key`].
fn keyed_hunks(
    rel: &str,
    base: &Rope,
    text: &Rope,
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>, String)> {
    // identical hunks in a file are told apart by their occurrence
    let mut occurrences: HashMap<String, usize> = HashMap::new();
    helix_core::diff::line_hunks(base, text)
        .into_iter()
        .map(|(before, after)| {
            let before = before.start as usize..before.end as usize;
            let mut after = after.start as usize..after.end as usize;
            if text.len_chars() == 0 {
                // an empty document (e.g. a deleted file) has one empty line that isn't an addition
                after.end = after.start;
            }
            let content_hash = sha256_hex(&format!(
                "{rel}\0{}\0{}",
                line_range_text(base, &before),
                line_range_text(text, &after)
            ));
            let occurrence = occurrences.entry(content_hash.clone()).or_default();
            let key = if *occurrence == 0 {
                content_hash
            } else {
                sha256_hex(&format!("{content_hash}\0{occurrence}"))
            };
            *occurrence += 1;
            (before, after, key)
        })
        .collect()
}

/// The text of a range of lines.
fn line_range_text(text: &Rope, lines: &std::ops::Range<usize>) -> String {
    let end = lines.end.min(text.len_lines());
    let start = lines.start.min(end);
    text.slice(text.line_to_char(start)..text.line_to_char(end))
        .to_string()
}

/// Where reviewed changes are stored: a file in the repository's git directory
/// (shared between worktrees, never committed), keyed by branch.
fn review_store_path(root: &Path) -> anyhow::Result<PathBuf> {
    let git_dir = git_string(root, &["rev-parse", "--git-common-dir"])?;
    Ok(root.join(git_dir).join("helix-review.json"))
}

type ReviewStore = std::collections::BTreeMap<String, std::collections::BTreeSet<String>>;

fn load_store(path: &Path) -> ReviewStore {
    std::fs::read(path)
        .ok()
        .and_then(|contents| serde_json::from_slice(&contents).ok())
        .unwrap_or_default()
}

fn load_reviewed(path: &Path, branch: &str) -> HashSet<String> {
    load_store(path)
        .remove(branch)
        .map(|keys| keys.into_iter().collect())
        .unwrap_or_default()
}

fn save_reviewed(path: &Path, branch: &str, reviewed: &HashSet<String>) -> anyhow::Result<()> {
    // other branches may have been reviewed by other instances meanwhile
    let mut store = load_store(path);
    if reviewed.is_empty() {
        store.remove(branch);
    } else {
        store.insert(branch.to_string(), reviewed.iter().cloned().collect());
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&store)?)?;
    std::fs::rename(&tmp, path).context("failed to save the review state")?;
    Ok(())
}

/// Toggles the reviewed state of the changes under the selections (or of all
/// changes in the file). Marks them all as reviewed unless they all already are.
fn toggle_reviewed(editor: &mut Editor, whole_file: bool) -> anyhow::Result<()> {
    let mut state = state();
    let review = state
        .as_mut()
        .context("not reviewing, start with :review")?;
    let (view, doc) = current!(editor);
    let hunks = review
        .hunks
        .get(&doc.id())
        .filter(|hunks| !hunks.is_empty())
        .context("no changes in this file")?;
    let text = doc.text().slice(..);
    let selected: Vec<std::ops::RangeInclusive<usize>> = doc
        .selection(view.id)
        .iter()
        .map(|range| {
            let (start, end) = range.line_range(text);
            start..=end
        })
        .collect();
    let keys: Vec<String> = hunks
        .iter()
        .filter(|hunk| whole_file || selected.iter().any(|lines| hunk.touches(lines.clone())))
        .map(|hunk| hunk.key.clone())
        .collect();
    if keys.is_empty() {
        bail!("no change under the cursor");
    }
    let reviewed = !keys.iter().all(|key| review.reviewed.contains(key));
    let count = keys.len();
    review.set_reviewed(keys, reviewed)?;
    review.refresh(doc);

    let hunks = &review.hunks[&doc.id()];
    let done = hunks
        .iter()
        .filter(|hunk| review.reviewed.contains(&hunk.key))
        .count();
    let changes = if count == 1 { "change" } else { "changes" };
    let marked = if reviewed { "reviewed" } else { "not reviewed" };
    editor.set_status(format!(
        "Marked {count} {changes} as {marked} ({done}/{} in this file)",
        hunks.len()
    ));
    Ok(())
}

/// `:review-mark`: toggle the reviewed state of the changes under the selections.
pub(crate) fn review_mark(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event == PromptEvent::Validate {
        toggle_reviewed(cx.editor, false)?;
    }
    Ok(())
}

/// `:review-mark-file`: toggle the reviewed state of all changes in the file.
pub(crate) fn review_mark_file(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event == PromptEvent::Validate {
        toggle_reviewed(cx.editor, true)?;
    }
    Ok(())
}

// Typed commands

/// `:review [base]`: start reviewing the changes since `base` (or restart against a new base).
pub(crate) fn review(
    cx: &mut compositor::Context,
    args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event != PromptEvent::Validate {
        return Ok(());
    }
    start_review(cx.editor, cx.jobs, args.first())
}

/// `:review-files`: pick one of the files changed in the current review.
pub(crate) fn review_files(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event != PromptEvent::Validate {
        return Ok(());
    }
    open_picker(cx.editor, cx.jobs)
}

/// `:review-close`: leave review mode.
pub(crate) fn review_close(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event != PromptEvent::Validate {
        return Ok(());
    }
    let review = state().take().context("not reviewing")?;
    review.stop(cx.editor);
    Ok(())
}

pub(crate) fn review_next(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event == PromptEvent::Validate {
        goto_hunk(cx.editor, true)?;
    }
    Ok(())
}

pub(crate) fn review_prev(
    cx: &mut compositor::Context,
    _args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event == PromptEvent::Validate {
        goto_hunk(cx.editor, false)?;
    }
    Ok(())
}

/// `:review-line [pr]`: opens (or, over SSH, copies) a GitHub link to the diff
/// at the line under the cursor.
///
/// The link points at the pull request of the current branch, whose number is
/// taken from the argument (and remembered for the branch) or looked up with
/// the GitHub CLI when it is installed. Without either, it points at GitHub's
/// compare view of the review's base and `HEAD`, which only needs the remote.
pub(crate) fn review_line(
    cx: &mut compositor::Context,
    args: helix_core::command_line::Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    if event != PromptEvent::Validate {
        return Ok(());
    }
    let pr_number = args
        .first()
        .map(|arg| {
            arg.trim_start_matches('#')
                .parse::<u64>()
                .with_context(|| format!("not a pull request number: {arg}"))
        })
        .transpose()?;
    let (view, doc) = current_ref!(cx.editor);
    let path = doc.path().context("buffer has no file")?.to_path_buf();
    let line = doc
        .selection(view.id)
        .primary()
        .cursor_line(doc.text().slice(..));
    let unsaved = doc.is_modified();

    let dir = path.parent().context("file has no parent directory")?;
    let root = helix_stdx::path::canonicalize(PathBuf::from(git_string(
        dir,
        &["rev-parse", "--show-toplevel"],
    )?));
    if !cx
        .editor
        .workspace_trust
        .query(&helix_loader::find_workspace_in(&root).0, TrustQuery::Git)
        .is_trusted()
    {
        bail!("this runs git, which requires a trusted workspace (see :workspace-trust)");
    }
    let rel = path
        .strip_prefix(&root)
        .context("file is not in the repository")?
        .to_str()
        .context("path is not valid UTF-8")?
        .replace('\\', "/");
    // GitHub anchors a diff line by the SHA-256 of the file path and the line
    // number on the right (new) side.
    let anchor = format!("diff-{}R{}", sha256_hex(&rel), line + 1);
    let over_ssh = ["SSH_CONNECTION", "SSH_TTY"]
        .iter()
        .any(|var| std::env::var_os(var).is_some());
    // compare against the review's base if reviewing this repository
    let review_base = state()
        .as_ref()
        .filter(|review| review.root == root)
        .map(|review| review.base.clone());

    cx.jobs.callback(async move {
        let (url, mut caveats, note) = tokio::task::spawn_blocking(move || {
            let mut caveats = Vec::new();
            if unsaved {
                caveats.push("unsaved changes".to_string());
            }
            if git(&root, &["diff", "--quiet", "HEAD", "--", &rel]).is_err() {
                caveats.push("uncommitted changes".to_string());
            }
            let head = git_string(&root, &["rev-parse", "HEAD"])?;
            let link = diff_link(&root, pr_number, review_base)?;
            let (url, note) = match &link {
                DiffLink::PullRequest { url, head: pr_head } => {
                    match pr_head {
                        Some(pr_head) if *pr_head != head => {
                            caveats.push("local HEAD differs from the pull request".to_string())
                        }
                        Some(_) => (),
                        None => caveats.extend(unpushed(&root, &head)),
                    }
                    (format!("{url}/changes?diff=unified#{anchor}"), None)
                }
                DiffLink::Compare { repo, base } => {
                    caveats.extend(unpushed(&root, &head));
                    (
                        format!("{repo}/compare/{base}...{head}?diff=unified#{anchor}"),
                        Some("compare view; pass the pull request number to link to it"),
                    )
                }
            };
            anyhow::Ok((url, caveats, note))
        })
        .await??;

        let opened = !over_ssh && open_in_browser(&url).await;
        if !opened && !over_ssh {
            caveats.push("could not open a browser".to_string());
        }

        let call: job::Callback = Callback::Editor(Box::new(move |editor: &mut Editor| {
            let copied = editor.registers.write('+', vec![url.clone()]).is_ok();
            let action = match (opened, copied) {
                (true, _) => "Opened",
                (false, true) => "Copied",
                (false, false) => "Link:",
            };
            let note = note.map_or(String::new(), |note| format!(" ({note})"));
            if caveats.is_empty() {
                editor.set_status(format!("{action} {url}{note}"));
            } else {
                editor.set_warning(format!(
                    "{action} {url}{note} (line may be off: {})",
                    caveats.join(", ")
                ));
            }
        }));
        Ok(call)
    });
    Ok(())
}

/// Opens `url` with the system's URL opener, returning whether that worked.
async fn open_in_browser(url: &str) -> bool {
    for cmd in open::commands(url) {
        let mut command: tokio::process::Command = cmd.into();
        command.stdin(std::process::Stdio::null());
        if command
            .output()
            .await
            .is_ok_and(|output| output.status.success())
        {
            return true;
        }
    }
    false
}

/// What a `:review-line` link points at.
#[derive(Clone)]
enum DiffLink {
    /// A pull request page, and the commit it points at if known.
    PullRequest { url: String, head: Option<String> },
    /// GitHub's compare view of `base...HEAD` in `repo`.
    Compare { repo: String, base: String },
}

/// Pull requests of checked out branches, by repository root and branch.
static PULL_REQUESTS: Mutex<Option<HashMap<(PathBuf, String), DiffLink>>> = Mutex::new(None);

/// How to link to the diff of the branch checked out in `root`: the pull
/// request `pr_number` (remembered for the branch), a previously remembered or
/// GitHub CLI provided pull request, or else the compare view against `base`
/// (by default the merge base with the default branch).
fn diff_link(
    root: &Path,
    pr_number: Option<u64>,
    base: Option<String>,
) -> anyhow::Result<DiffLink> {
    let branch = git_string(root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    let key = (root.to_path_buf(), branch);
    let mut cache = PULL_REQUESTS.lock().unwrap_or_else(|err| err.into_inner());
    let cache = cache.get_or_insert_default();

    if let Some(number) = pr_number {
        let link = DiffLink::PullRequest {
            url: format!("{}/pull/{number}", github_repo_url(root)?),
            head: None,
        };
        cache.insert(key, link.clone());
        return Ok(link);
    }
    if let Some(link) = cache.get(&key) {
        let mut link = link.clone();
        if let DiffLink::PullRequest {
            head: Some(head), ..
        } = &mut link
        {
            // the pull request moves when new commits are pushed
            if let Ok(pr) = gh_pull_request(root) {
                *head = pr.1;
            }
        }
        return Ok(link);
    }
    if helix_stdx::env::binary_exists("gh") {
        if let Ok((url, head)) = gh_pull_request(root) {
            let link = DiffLink::PullRequest {
                url,
                head: Some(head),
            };
            cache.insert(key, link.clone());
            return Ok(link);
        }
    }
    let base = match base {
        Some(base) => base,
        None => {
            let default = default_base(root)?;
            git_string(root, &["merge-base", &default, "HEAD"])?
        }
    };
    Ok(DiffLink::Compare {
        repo: github_repo_url(root)?,
        base,
    })
}

/// The pull request of the current branch according to the GitHub CLI, as
/// `(url, head commit)`.
fn gh_pull_request(root: &Path) -> anyhow::Result<(String, String)> {
    let output = Command::new("gh")
        .args(["pr", "view", "--json", "url,headRefOid"])
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .output()
        .context("failed to run gh")?;
    if !output.status.success() {
        bail!(
            "gh pr view: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let field = |name: &str| {
        json[name]
            .as_str()
            .map(String::from)
            .with_context(|| format!("gh pr view did not return {name}"))
    };
    Ok((field("url")?, field("headRefOid")?))
}

/// The web URL of the GitHub repository the current branch is pushed to (its
/// upstream's remote, or `origin`).
fn github_repo_url(root: &Path) -> anyhow::Result<String> {
    let remote = git_string(
        root,
        &[
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ],
    )
    .ok()
    .and_then(|upstream| {
        upstream
            .split_once('/')
            .map(|(remote, _)| remote.to_string())
    })
    .unwrap_or_else(|| "origin".to_string());
    let url = git_string(root, &["remote", "get-url", &remote])
        .with_context(|| format!("no remote named {remote}"))?;
    remote_web_url(&url).with_context(|| format!("can't make a web link from remote {url}"))
}

/// Turns a git remote URL (`git@host:owner/repo.git`, `ssh://git@host/owner/repo`,
/// `https://host/owner/repo.git`) into the repository's web URL.
fn remote_web_url(remote: &str) -> Option<String> {
    let remote = remote.trim().trim_end_matches('/');
    let (host, path) = if let Some(rest) = remote
        .strip_prefix("https://")
        .or_else(|| remote.strip_prefix("http://"))
        .or_else(|| remote.strip_prefix("ssh://"))
        .or_else(|| remote.strip_prefix("git://"))
    {
        let rest = rest.rsplit_once('@').map_or(rest, |(_, rest)| rest);
        rest.split_once('/')?
    } else {
        // scp-like syntax: [user@]host:owner/repo
        let rest = remote.rsplit_once('@').map_or(remote, |(_, rest)| rest);
        rest.split_once(':')?
    };
    // drop a port
    let host = host.split_once(':').map_or(host, |(host, _)| host);
    let path = path.trim_end_matches(".git");
    (!host.is_empty() && path.contains('/')).then(|| format!("https://{host}/{path}"))
}

/// A caveat if `head` hasn't been pushed (GitHub can't show it then).
fn unpushed(root: &Path, head: &str) -> Option<String> {
    match git_string(root, &["rev-parse", "@{upstream}"]) {
        Ok(upstream) if upstream == head => None,
        Ok(_) => Some("local HEAD differs from the pushed branch".to_string()),
        Err(_) => Some("branch has no upstream".to_string()),
    }
}

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(text.as_bytes())
        .iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

fn goto_hunk(editor: &mut Editor, forward: bool) -> anyhow::Result<()> {
    let state = state();
    let review = state
        .as_ref()
        .context("not reviewing, start with :review")?;
    let (view, doc) = current!(editor);
    let Some(hunks) = review.hunks.get(&doc.id()) else {
        bail!("no changes in this file");
    };
    let text = doc.text().slice(..);
    let last_line = text.len_lines().saturating_sub(1);
    let lines: Vec<usize> = hunks
        .iter()
        .map(|hunk| hunk.after.start.min(last_line))
        .collect();
    let cursor_line = doc.selection(view.id).primary().cursor_line(text);
    let target = if forward {
        lines.iter().find(|&&line| line > cursor_line)
    } else {
        lines.iter().rev().find(|&&line| line < cursor_line)
    };
    let Some(&line) = target else {
        bail!("no more changes in this file");
    };
    let pos = text.line_to_char(line);
    view.push_jump(doc);
    doc.set_selection(view.id, Selection::point(pos));
    helix_view::align_view(doc, view, helix_view::Align::Center);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::KEYS;
    use crate::keymap::KeyTrie;
    use helix_view::document::Mode;
    use std::collections::HashMap;

    #[test]
    fn hunk_touches() {
        let hunk = |after| super::Hunk {
            after,
            key: String::new(),
        };
        // added lines 3..5
        assert!(hunk(3..5).touches(4..=4));
        assert!(hunk(3..5).touches(0..=3));
        assert!(!hunk(3..5).touches(5..=6));
        // removal shown between lines 2 and 3
        assert!(hunk(3..3).touches(2..=2));
        assert!(hunk(3..3).touches(3..=3));
        assert!(!hunk(3..3).touches(4..=4));
    }

    #[test]
    fn reviewed_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("helix-review.json");
        assert!(super::load_reviewed(&path, "a").is_empty());
        let keys: std::collections::HashSet<String> = ["x".to_string()].into();
        super::save_reviewed(&path, "a", &keys).unwrap();
        super::save_reviewed(&path, "b", &["y".to_string()].into()).unwrap();
        assert_eq!(super::load_reviewed(&path, "a"), keys);
        super::save_reviewed(&path, "a", &Default::default()).unwrap();
        assert!(super::load_reviewed(&path, "a").is_empty());
        assert_eq!(super::load_reviewed(&path, "b").len(), 1);
    }

    #[test]
    fn remote_web_urls() {
        use super::remote_web_url;
        let expected = Some("https://github.com/oxidecomputer/opte".to_string());
        assert_eq!(
            remote_web_url("git@github.com:oxidecomputer/opte.git"),
            expected
        );
        assert_eq!(
            remote_web_url("https://github.com/oxidecomputer/opte.git"),
            expected
        );
        assert_eq!(
            remote_web_url("https://github.com/oxidecomputer/opte"),
            expected
        );
        assert_eq!(
            remote_web_url("ssh://git@github.com/oxidecomputer/opte"),
            expected
        );
        assert_eq!(
            remote_web_url("ssh://git@github.com:22/oxidecomputer/opte.git"),
            expected
        );
        assert_eq!(
            remote_web_url("https://user@github.com/oxidecomputer/opte/"),
            expected
        );
        assert_eq!(remote_web_url("/some/local/path"), None);
    }

    #[test]
    fn github_anchor_hash() {
        assert_eq!(
            super::sha256_hex("lib/opte/src/engine/flow_table.rs"),
            "d771e039225dde12ed95eb7abf0cb5eedffd61bd5942fecef1fed866e5dccd15"
        );
    }

    #[test]
    fn keys_parse() {
        let keys: HashMap<Mode, KeyTrie> = toml::from_str(KEYS).unwrap();
        assert!(keys.contains_key(&Mode::Normal));
    }
}
