use pulldown_cmark::{html, CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

/// Renders Markdown into HTML for display inside the web UI.
///
/// Additionally, converts fenced code blocks with language `mermaid` into
/// `<div class="mermaid">...</div>` nodes so Mermaid can render them.
pub fn render_markdown_with_mermaid(md: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_SMART_PUNCTUATION);

    let parser = Parser::new_ext(md, options);

    let mut in_mermaid_block = false;
    let mut events: Vec<Event<'_>> = Vec::new();

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(lang)))
                if lang.trim().eq_ignore_ascii_case("mermaid") =>
            {
                in_mermaid_block = true;
                events.push(Event::Html("<div class=\"mermaid\">".into()));
            }
            Event::End(TagEnd::CodeBlock) if in_mermaid_block => {
                in_mermaid_block = false;
                events.push(Event::Html("</div>".into()));
            }
            other => {
                if in_mermaid_block {
                    // Drop any nested tags that would introduce <pre><code> wrappers.
                    match other {
                        Event::Start(Tag::CodeBlock(_)) | Event::End(TagEnd::CodeBlock) => {}
                        _ => events.push(other),
                    }
                } else {
                    events.push(other);
                }
            }
        }
    }

    let mut out = String::new();
    html::push_html(&mut out, events.into_iter());

    // For doc pages, we want each H2 section to read like a separate "card".
    // We do this as a post-process on the rendered HTML to avoid complicating
    // the pulldown-cmark event stream.
    wrap_h2_sections(&out)
}

fn wrap_h2_sections(html: &str) -> String {
    let first = match html.find("<h2") {
        Some(i) => i,
        None => return html.to_string(),
    };

    let mut out = String::with_capacity(html.len() + 64);
    out.push_str(&html[..first]);

    let mut rest = &html[first..];
    while let Some(next) = rest[3..].find("<h2").map(|i| i + 3) {
        let (section, tail) = rest.split_at(next);
        out.push_str("<section class=\"docs-section\">");
        out.push_str(section);
        out.push_str("</section>");
        rest = tail;
    }

    out.push_str("<section class=\"docs-section\">");
    out.push_str(rest);
    out.push_str("</section>");
    out
}
