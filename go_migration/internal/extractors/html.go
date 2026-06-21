package extractors

import (
	"bytes"
	"strings"

	"github.com/kennethstott/chonk/internal/models"
	"golang.org/x/net/html"
)

// HTMLExtractor converts HTML to plain Markdown-ish text.
// Python source: chonk/extractors/_html.py
// MIGRATION MARKER: Status COMPLETE — Python BeautifulSoup get_text() replaced
// by golang.org/x/net/html tree walker with heading promotion.
type HTMLExtractor struct{}

// CanHandle returns true for html and htm.
func (e HTMLExtractor) CanHandle(docType string) bool {
	return docType == "html" || docType == "htm" || docType == "text/html"
}

// Extract parses HTML and returns readable plain text with Markdown headings.
func (e HTMLExtractor) Extract(data []byte, _ string) (string, error) {
	if len(data) == 0 {
		return "", nil
	}
	doc, err := html.Parse(bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	walkHTML(doc, &sb)
	return strings.TrimSpace(sb.String()), nil
}

// Annotate is a no-op for HTML.
func (e HTMLExtractor) Annotate(chunks []models.DocumentChunk, _ []byte, _ string) []models.DocumentChunk {
	return chunks
}

// ── HTML tree walker ──────────────────────────────────────────────────────────

// skipTags lists tags whose content is not part of visible body text.
var skipTags = map[string]bool{
	"script": true, "style": true, "noscript": true,
	"head": true, "meta": true, "link": true, "template": true,
}

// headingPrefix maps HTML heading tags to Markdown prefixes.
var headingPrefix = map[string]string{
	"h1": "# ", "h2": "## ", "h3": "### ",
	"h4": "#### ", "h5": "##### ", "h6": "###### ",
}

func walkHTML(n *html.Node, sb *strings.Builder) {
	if n.Type == html.TextNode {
		text := strings.TrimSpace(n.Data)
		if text != "" {
			sb.WriteString(text)
			sb.WriteByte(' ')
		}
		return
	}
	if n.Type == html.ElementNode {
		tag := strings.ToLower(n.Data)
		if skipTags[tag] {
			return
		}
		if prefix, ok := headingPrefix[tag]; ok {
			sb.WriteByte('\n')
			sb.WriteString(prefix)
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				walkHTML(c, sb)
			}
			sb.WriteByte('\n')
			return
		}
		if tag == "p" || tag == "div" || tag == "section" || tag == "article" ||
			tag == "li" || tag == "blockquote" || tag == "tr" {
			sb.WriteByte('\n')
		}
		if tag == "a" {
			// Render links as Markdown: [text](href)
			var href string
			for _, attr := range n.Attr {
				if attr.Key == "href" {
					href = attr.Val
					break
				}
			}
			var linkText strings.Builder
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				walkHTML(c, &linkText)
			}
			text := strings.TrimSpace(linkText.String())
			if text != "" && href != "" {
				sb.WriteString("[")
				sb.WriteString(text)
				sb.WriteString("](")
				sb.WriteString(href)
				sb.WriteString(")")
			} else {
				sb.WriteString(text)
			}
			return
		}
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkHTML(c, sb)
	}
}
