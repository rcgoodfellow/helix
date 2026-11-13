//! Rendering for [`helix_view::annotations::decorations`]: virtual lines and line styles.
//! (Gutter marks are rendered by the `marks` gutter.)

use helix_core::unicode::width::UnicodeWidthChar;
use helix_core::Position;
use helix_view::annotations::decorations::{DocumentDecorations, VirtualLine};
use helix_view::graphics::Rect;
use helix_view::theme::Style;
use helix_view::Theme;

use crate::ui::document::{LinePos, TextRenderer};
use crate::ui::text_decorations::Decoration;

/// Where the gutter of the view is, so virtual lines can draw their own gutter.
#[derive(Debug, Clone, Copy)]
pub struct GutterArea {
    pub x: u16,
    pub width: u16,
    /// Column of the gutter that virtual line markers are drawn in.
    pub marker_x: u16,
}

pub struct DocumentDecorationsRenderer<'a> {
    decorations: &'a DocumentDecorations,
    theme: &'a Theme,
    gutter: Option<GutterArea>,
    gutter_style: Style,
    tab_width: usize,
    /// reused buffers for the visible part of a virtual line and, for each of
    /// its bytes, the byte offset in the original text
    buf: String,
    src_offsets: Vec<usize>,
    styles: Vec<(std::ops::Range<usize>, Style)>,
}

impl<'a> DocumentDecorationsRenderer<'a> {
    pub fn new(
        decorations: &'a DocumentDecorations,
        theme: &'a Theme,
        gutter: Option<GutterArea>,
        tab_width: usize,
    ) -> Self {
        Self {
            decorations,
            theme,
            gutter,
            gutter_style: theme.get("ui.gutter.virtual"),
            tab_width: tab_width.max(1),
            buf: String::new(),
            src_offsets: Vec::new(),
            styles: Vec::new(),
        }
    }

    fn draw_virtual_line(&mut self, renderer: &mut TextRenderer, row: u16, line: &VirtualLine) {
        let viewport = renderer.viewport;
        let style = self.theme.get(&line.scope);

        if let Some(gutter) = self.gutter {
            renderer.set_style(Rect::new(gutter.x, row, gutter.width, 1), self.gutter_style);
            let mut x = gutter.marker_x;
            let gutter_end = gutter.x + gutter.width;
            for marker in &line.markers {
                if x >= gutter_end {
                    break;
                }
                let marker_style = self.gutter_style.patch(self.theme.get(&marker.scope));
                renderer.set_stringn(
                    x,
                    row,
                    &marker.text,
                    (gutter_end - x) as usize,
                    marker_style,
                );
                x += marker.text.chars().count() as u16;
            }
        }

        // Fill the whole row so a background in `scope` covers the line.
        renderer.set_style(Rect::new(viewport.x, row, viewport.width, 1), style);

        visible_text(
            &line.text,
            self.tab_width,
            renderer.offset.col,
            &mut self.buf,
            &mut self.src_offsets,
        );
        let theme = self.theme;
        let apply_overlay = |style: Style| match &line.overlay {
            Some(overlay) => overlay.apply(style, theme),
            None => style,
        };
        let base_style = renderer.text_style.patch(style);
        let text_style = apply_overlay(base_style);
        self.styles.clear();
        self.styles.extend(line.highlights.iter().map(|highlight| {
            let style = highlight.scopes.iter().fold(base_style, |style, scope| {
                style.patch(self.theme.get(scope))
            });
            (highlight.range.clone(), apply_overlay(style))
        }));
        let (styles, src_offsets) = (&self.styles, &self.src_offsets);
        renderer.set_string_truncated(
            viewport.x,
            row,
            &self.buf,
            viewport.width as usize,
            |offset| {
                let Some(&src) = src_offsets.get(offset) else {
                    return text_style;
                };
                // highlights are sorted and don't overlap
                let idx = styles.partition_point(|(range, _)| range.end <= src);
                match styles.get(idx) {
                    Some((range, style)) if range.start <= src => *style,
                    _ => text_style,
                }
            },
            false,
            false,
        );
    }
}

impl Decoration for DocumentDecorationsRenderer<'_> {
    fn decorate_line(&mut self, renderer: &mut TextRenderer, pos: LinePos) {
        if let Some(scope) = self.decorations.line_style(pos.doc_line) {
            let viewport = renderer.viewport;
            renderer.set_style(
                Rect::new(viewport.x, pos.visual_line, viewport.width, 1),
                self.theme.get(scope),
            );
        }
    }

    fn render_virt_lines(
        &mut self,
        renderer: &mut TextRenderer,
        pos: LinePos,
        virt_off: Position,
    ) -> Position {
        // Must mirror `VirtualLinesAnnotation`, which only reserves space after
        // the last visual line of a document line.
        if !pos.last_visual_line {
            return Position::new(0, 0);
        }
        let decorations = self.decorations;
        let bottom = renderer.offset.row + renderer.viewport.height as usize;
        let mut count = 0;
        for line in decorations.virtual_lines_at(pos.doc_line) {
            let row = pos.visual_line as usize + virt_off.row + count;
            count += 1;
            // rows above the view are skipped by the renderer itself
            if row < bottom {
                self.draw_virtual_line(renderer, row as u16, line);
            }
        }
        Position::new(count, 0)
    }
}

/// Expands tabs and drops the first `skip_cols` columns (horizontal scrolling) of `text`.
/// `src_offsets` receives, for every byte of `out`, the offset of the char it came from in `text`.
fn visible_text(
    text: &str,
    tab_width: usize,
    skip_cols: usize,
    out: &mut String,
    src_offsets: &mut Vec<usize>,
) {
    out.clear();
    src_offsets.clear();
    let mut col = 0;
    for (offset, ch) in text.char_indices() {
        if ch == '\n' || ch == '\r' {
            break;
        }
        let width = if ch == '\t' {
            tab_width - col % tab_width
        } else {
            ch.width().unwrap_or(0)
        };
        let end = col + width;
        if end > skip_cols {
            let len = out.len();
            if ch == '\t' || col < skip_cols {
                // a tab or wide char partially scrolled out of view
                out.extend(std::iter::repeat_n(' ', end - col.max(skip_cols)));
            } else {
                out.push(ch);
            }
            src_offsets.resize(out.len(), offset);
            debug_assert!(src_offsets.len() > len);
        }
        col = end;
    }
}

#[cfg(test)]
mod tests {
    use super::visible_text;

    #[test]
    fn visible_text_expands_tabs_and_scrolls() {
        let mut out = String::new();
        let mut offsets = Vec::new();
        visible_text("x\tab", 4, 2, &mut out, &mut offsets);
        assert_eq!(offsets, [1, 1, 2, 3]);
        visible_text("\tab", 4, 0, &mut out, &mut offsets);
        assert_eq!(out, "    ab");
        visible_text("x\tab", 4, 0, &mut out, &mut offsets);
        assert_eq!(out, "x   ab");
        visible_text("x\tab", 4, 2, &mut out, &mut offsets);
        assert_eq!(out, "  ab");
        visible_text("abcdef", 4, 4, &mut out, &mut offsets);
        assert_eq!(out, "ef");
        visible_text("a\r\n", 4, 0, &mut out, &mut offsets);
        assert_eq!(out, "a");
    }
}
