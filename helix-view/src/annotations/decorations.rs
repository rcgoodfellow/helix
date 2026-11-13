//! Externally supplied, namespaced decorations for a document.
//!
//! These are a small set of generic, data-only primitives that let code outside
//! of the renderer (built-in modes today, plugins in the future) change how a
//! document is *displayed* without changing its text:
//!
//! * [`VirtualLines`]: whole lines of text shown **below** a document line. They
//!   are not part of the document: they cannot be selected or edited, vertical
//!   motions skip them and they get no line number.
//! * [`GutterMark`]: a short marker shown in the `marks` gutter next to a
//!   document line.
//! * [`LineStyle`]: a theme scope applied to every visual row of a range of
//!   document lines (typically to tint the background).
//!
//! Decorations are grouped into named layers so that independent producers do
//! not clobber each other. A producer owns its layer and replaces it wholesale
//! with [`DocumentDecorations::set_layer`]. Line indices are **not** mapped
//! through edits: producers that want their decorations to follow edits must
//! recompute them (for example from a `DocumentDidChange` hook). Out of range
//! lines are ignored, so stale decorations are never a correctness problem.
//!
//! Styles are theme scope names (for example `"diff.plus.gutter"`) that are
//! resolved at render time with the usual dot-separated fallback, so producers
//! don't need access to the theme and decorations follow theme changes.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use helix_core::text_annotations::LineAnnotation;
use helix_core::{Position, RopeSlice};

use crate::graphics::{Color, Modifier, Style};
use crate::Theme;

/// A short piece of text (usually a single cell wide) and the theme scope used to draw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Marker {
    pub text: String,
    pub scope: String,
}

impl Marker {
    pub fn new(text: impl Into<String>, scope: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            scope: scope.into(),
        }
    }
}

/// A marker displayed in the `marks` gutter for a document line. A line can
/// have several marks (also from several layers); they are drawn next to each
/// other, in layer order and then in the order they were added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GutterMark {
    pub line: usize,
    pub marker: Marker,
}

/// A single line of virtual text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualLine {
    /// The text to display. Should not contain line breaks. Lines wider than
    /// the view are truncated rather than soft-wrapped.
    pub text: String,
    /// Theme scope used for the text (and the rest of the row).
    pub scope: String,
    /// Optional highlighting (usually syntax highlighting) patched on top of `scope`.
    /// Must be sorted by `range.start` and not overlap.
    pub highlights: Vec<TextHighlight>,
    /// Optional translucent color layer drawn over the text (after highlighting),
    /// e.g. to subdue it while keeping the highlighting recognizable.
    pub overlay: Option<ColorOverlay>,
    /// Markers drawn in the gutter of this row, next to each other like
    /// [`GutterMark`]s. They are placed in the column of the `marks` gutter (or
    /// the `diff` gutter when there is no `marks` gutter). No other gutter is
    /// drawn for virtual lines.
    pub markers: Vec<Marker>,
}

/// A translucent color layer over text: text colors are blended toward the
/// foreground color of `scope` (with the usual theme fallback) by `opacity`
/// percent. Where colors can't be blended (not true color) the text is dimmed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorOverlay {
    pub scope: String,
    /// 0 (no effect) to 100 (text takes the overlay color).
    pub opacity: u8,
}

impl ColorOverlay {
    /// Applies the overlay to `style`.
    pub fn apply(&self, style: Style, theme: &Theme) -> Style {
        overlaid(style, theme.get(&self.scope).fg, self.opacity)
    }
}

/// Blends the foreground of `style` toward `color` by `opacity` percent, or dims
/// it when either color isn't a true color.
fn overlaid(mut style: Style, color: Option<Color>, opacity: u8) -> Style {
    let opacity = u16::from(opacity.min(100));
    if opacity == 0 {
        return style;
    }
    match (style.fg, color) {
        (Some(Color::Rgb(r, g, b)), Some(Color::Rgb(or, og, ob))) => {
            let blend = |c: u8, o: u8| {
                ((u16::from(c) * (100 - opacity) + u16::from(o) * opacity + 50) / 100) as u8
            };
            style.fg = Some(Color::Rgb(blend(r, or), blend(g, og), blend(b, ob)));
            style
        }
        _ => style.add_modifier(Modifier::DIM),
    }
}

/// Theme scopes applied to a byte range of a [`VirtualLine`]'s text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextHighlight {
    pub range: Range<usize>,
    /// Applied in order, later scopes take precedence (like nested syntax highlights).
    /// Shared because the same list is usually used by many highlights.
    pub scopes: Arc<[Arc<str>]>,
}

/// A block of virtual lines displayed directly below document line `after_line`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualLines {
    pub after_line: usize,
    pub lines: Vec<VirtualLine>,
}

/// A theme scope applied to every visual row of the document lines in `lines`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineStyle {
    pub lines: Range<usize>,
    pub scope: String,
}

/// All decorations produced by one producer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecorationLayer {
    pub virtual_lines: Vec<VirtualLines>,
    pub gutter_marks: Vec<GutterMark>,
    pub line_styles: Vec<LineStyle>,
    /// Short pieces of text shown by the `decoration-status` statusline element.
    pub status: Vec<Marker>,
}

impl DecorationLayer {
    pub fn is_empty(&self) -> bool {
        self.virtual_lines.is_empty()
            && self.gutter_marks.is_empty()
            && self.line_styles.is_empty()
            && self.status.is_empty()
    }

    fn sort(&mut self) {
        self.virtual_lines.sort_by_key(|block| block.after_line);
        self.gutter_marks.sort_by_key(|mark| mark.line);
        self.line_styles.sort_by_key(|style| style.lines.start);
    }

    fn virtual_lines_at(&self, line: usize) -> impl Iterator<Item = &VirtualLine> {
        let start = self
            .virtual_lines
            .partition_point(|block| block.after_line < line);
        self.virtual_lines[start..]
            .iter()
            .take_while(move |block| block.after_line == line)
            .flat_map(|block| &block.lines)
    }

    fn gutter_marks(&self, line: usize) -> impl Iterator<Item = &Marker> {
        let idx = self.gutter_marks.partition_point(|mark| mark.line < line);
        self.gutter_marks[idx..]
            .iter()
            .take_while(move |mark| mark.line == line)
            .map(|mark| &mark.marker)
    }

    /// The widest set of markers on a single row of this layer.
    fn marks_width(&self) -> usize {
        let width = |markers: &mut dyn Iterator<Item = &Marker>| -> usize {
            markers.map(|marker| marker.text.chars().count()).sum()
        };
        let mut max = 0;
        for group in self.gutter_marks.chunk_by(|a, b| a.line == b.line) {
            max = max.max(width(&mut group.iter().map(|mark| &mark.marker)));
        }
        for block in &self.virtual_lines {
            for line in &block.lines {
                max = max.max(width(&mut line.markers.iter()));
            }
        }
        max
    }

    fn line_style(&self, line: usize) -> Option<&str> {
        // styles are sorted by start; the candidates are the ones starting at or before `line`
        let end = self
            .line_styles
            .partition_point(|style| style.lines.start <= line);
        self.line_styles[..end]
            .iter()
            .rev()
            .find(|style| style.lines.contains(&line))
            .map(|style| style.scope.as_str())
    }
}

/// The decorations of a document, keyed by layer name.
#[derive(Debug, Clone, Default)]
pub struct DocumentDecorations {
    layers: BTreeMap<String, DecorationLayer>,
}

impl DocumentDecorations {
    /// Replaces (or creates) the layer `name`. An empty layer removes it.
    pub fn set_layer(&mut self, name: impl Into<String>, mut layer: DecorationLayer) {
        let name = name.into();
        if layer.is_empty() {
            self.layers.remove(&name);
        } else {
            layer.sort();
            self.layers.insert(name, layer);
        }
    }

    pub fn remove_layer(&mut self, name: &str) -> Option<DecorationLayer> {
        self.layers.remove(name)
    }

    pub fn layer(&self, name: &str) -> Option<&DecorationLayer> {
        self.layers.get(name)
    }

    pub fn layers(&self) -> impl Iterator<Item = (&str, &DecorationLayer)> {
        self.layers
            .iter()
            .map(|(name, layer)| (name.as_str(), layer))
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn has_virtual_lines(&self) -> bool {
        self.layers
            .values()
            .any(|layer| !layer.virtual_lines.is_empty())
    }

    /// The virtual lines shown below document line `line`, in layer order.
    pub fn virtual_lines_at(&self, line: usize) -> impl Iterator<Item = &VirtualLine> {
        self.layers
            .values()
            .flat_map(move |layer| layer.virtual_lines_at(line))
    }

    /// The statusline segments of all layers, in layer order.
    pub fn status(&self) -> impl Iterator<Item = &Marker> {
        self.layers.values().flat_map(|layer| &layer.status)
    }

    /// The gutter markers for document line `line`, in drawing order.
    pub fn gutter_marks(&self, line: usize) -> impl Iterator<Item = &Marker> {
        self.layers
            .values()
            .flat_map(move |layer| layer.gutter_marks(line))
    }

    /// The width the `marks` gutter needs (at least 1 cell).
    pub fn marks_width(&self) -> usize {
        self.layers
            .values()
            .map(DecorationLayer::marks_width)
            .sum::<usize>()
            .max(1)
    }

    /// The line style scope for document line `line`. Later layers take precedence.
    pub fn line_style(&self, line: usize) -> Option<&str> {
        self.layers
            .values()
            .rev()
            .find_map(|layer| layer.line_style(line))
    }
}

/// Reserves the vertical space for [`VirtualLines`] while a document is formatted.
///
/// Virtual lines are inserted after the *last* visual line of their document
/// line so that they appear below soft-wrapped lines rather than inside them.
pub(crate) struct VirtualLinesAnnotation<'a> {
    text: RopeSlice<'a>,
    decorations: &'a DocumentDecorations,
}

impl<'a> VirtualLinesAnnotation<'a> {
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new(
        text: RopeSlice<'a>,
        decorations: &'a DocumentDecorations,
    ) -> Box<dyn LineAnnotation + 'a> {
        Box::new(Self { text, decorations })
    }
}

impl LineAnnotation for VirtualLinesAnnotation<'_> {
    fn insert_virtual_lines(
        &mut self,
        line_end_char_idx: usize,
        _line_end_visual_pos: Position,
        doc_line: usize,
    ) -> Position {
        // This is called at the end of every *visual* line. The end of a
        // document line is where the next document line starts (or the end of
        // the text for the last line).
        let is_line_end = doc_line < self.text.len_lines()
            && line_end_char_idx == self.text.line_to_char(doc_line + 1);
        if !is_line_end {
            return Position::new(0, 0);
        }
        Position::new(self.decorations.virtual_lines_at(doc_line).count(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vline(text: &str) -> VirtualLine {
        VirtualLine {
            text: text.into(),
            scope: "ui.virtual".into(),
            highlights: Vec::new(),
            overlay: None,
            markers: Vec::new(),
        }
    }

    #[test]
    fn overlay_blends_or_dims() {
        let style = Style::default().fg(Color::Rgb(200, 100, 0));
        let blue = Some(Color::Rgb(0, 0, 100));
        assert_eq!(overlaid(style, blue, 50).fg, Some(Color::Rgb(100, 50, 50)));
        assert_eq!(overlaid(style, blue, 0), style);
        let dimmed = overlaid(Style::default().fg(Color::Red), blue, 50);
        assert!(dimmed.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn lookups() {
        let mut decorations = DocumentDecorations::default();
        decorations.set_layer(
            "b",
            DecorationLayer {
                virtual_lines: vec![
                    VirtualLines {
                        after_line: 5,
                        lines: vec![vline("x")],
                    },
                    VirtualLines {
                        after_line: 2,
                        lines: vec![vline("y"), vline("z")],
                    },
                ],
                gutter_marks: vec![GutterMark {
                    line: 3,
                    marker: Marker::new("b", "b"),
                }],
                line_styles: vec![LineStyle {
                    lines: 1..4,
                    scope: "b".into(),
                }],
                status: vec![Marker::new("+1", "diff.plus")],
            },
        );
        decorations.set_layer(
            "a",
            DecorationLayer {
                virtual_lines: vec![VirtualLines {
                    after_line: 2,
                    lines: vec![vline("w")],
                }],
                gutter_marks: vec![GutterMark {
                    line: 3,
                    marker: Marker::new("a", "a"),
                }],
                ..Default::default()
            },
        );

        let texts: Vec<_> = decorations
            .virtual_lines_at(2)
            .map(|line| line.text.as_str())
            .collect();
        assert_eq!(texts, ["w", "y", "z"]);
        assert_eq!(decorations.virtual_lines_at(4).count(), 0);
        let marks: Vec<_> = decorations
            .gutter_marks(3)
            .map(|m| m.text.as_str())
            .collect();
        assert_eq!(marks, ["a", "b"]);
        assert_eq!(decorations.gutter_marks(2).count(), 0);
        assert_eq!(decorations.marks_width(), 2);
        assert_eq!(decorations.line_style(0), None);
        assert_eq!(decorations.line_style(3), Some("b"));
        assert_eq!(decorations.line_style(4), None);

        let status: Vec<_> = decorations.status().map(|m| m.text.as_str()).collect();
        assert_eq!(status, ["+1"]);

        decorations.set_layer("b", DecorationLayer::default());
        assert_eq!(decorations.layers().count(), 1);
    }
}
