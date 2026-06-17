package extractors

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"text/template"
	"unicode/utf8"

	"github.com/andybalholm/cascadia"
	"github.com/gorilla/html"
)

// HTML extractor — converts HTML to Markdown using stdlib html.parser.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// _stripHtmlChrome removes navigation chrome tags from raw HTML.
// Removes <nav>, <aside>, <header>, <footer>, <noscript>, <script>, <style>
// and elements with navigation-related CSS classes/IDs.
func _stripHtmlChrome(html string) string {
	// Remove common navigation tags
	html = regexp.MustCompile(`<nav[^>]*>.*?</nav>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<aside[^>]*>.*?</aside>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<header[^>]*>.*?</header>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<footer[^>]*>.*?</footer>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<noscript[^>]*>.*?</noscript>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<script[^>]*>.*?</script>`).ReplaceAllString(html, "")
	html = regexp.MustCompile(`<style[^>]*>.*?</style>`).ReplaceAllString(html, "")

	// Remove elements with navigation-related CSS classes/IDs
	// Using a more comprehensive regex to match various navigation patterns
	navAttrRE := regexp.MustCompile(`<(div|section|ul|table)[^>]*?` +
		`(?:class|id|role)\s*=\s*['"][^'"]*?` +
		`(?:sidebar|navbox|navbar|navigation|toc\b|catlinks` +
		`|mw-panel|mw-head|mw-editsection` +
		`|menu|breadcrumb|noprint` +
		`|portal|sister-?project|interlanguage|authority-control` +
		`|reflist|references|footnotes|mw-references-wrap|citation)` +
		`[^'"]*?['"][^>]*>.*?</\1>`)
	html = navAttrRE.ReplaceAllString(html, "")

	// Remove reference superscripts
	html = regexp.MustCompile(`<sup[^>]*class=['"][^'"]*reference[^'"]*['"][^>]*>.*?</sup>`).ReplaceAllString(html, "")

	return html
}

// _convertHtmlToMarkdown converts HTML to Markdown, preserving heading structure.
// Uses stdlib html.parser — no external dependencies.
// Handles: headings, paragraphs, lists (ul/ol/li), <br>, <pre>/<code>,
// bold, italic, links, and tables.
// Strips navigation chrome (nav, aside, sidebar, navbox, etc.).
func _convertHtmlToMarkdown(html string) string {
	html = _stripHtmlChrome(html)

	// Create a new converter
	converter := &markdownConverter{
		output:     []string{},
		tagStack:   []string{},
		listStack:  []string{},
		olCounters: []int{},
		inPre:      false,
		href:       nil,
		linkText:   []string{},
		inLink:     false,
		inCell:     false,
		cellBuf:    []string{},
		currentRowCells: []string{},
		lastRowColCount: 0,
	}

	// Parse the HTML
	converter.feed(html)

	// Get the markdown output
	return converter.getMarkdown()
}

// markdownConverter converts HTML to Markdown.
type markdownConverter struct {
	output     []string
	tagStack   []string
	listStack  []string
	olCounters []int
	inPre      bool
	href       *string
	linkText   []string
	inLink     bool
	inCell     bool
	cellBuf    []string
	currentRowCells []string
	lastRowColCount int
}

// _append appends text to the output or cell buffer.
func (mc *markdownConverter) _append(text string) {
	if mc.inCell {
		mc.cellBuf = append(mc.cellBuf, text)
	} else {
		mc.output = append(mc.output, text)
	}
}

// handleStartTag handles HTML start tags.
func (mc *markdownConverter) handleStartTag(tag string, attrs []html.Attribute) {
	tag = strings.ToLower(tag)
	mc.tagStack = append(mc.tagStack, tag)

	switch tag {
	case "h1", "h2", "h3", "h4", "h5", "h6":
		mc.output = append(mc.output, "\n\n")
	case "p":
		if !mc.inCell {
			mc.output = append(mc.output, "\n\n")
		}
	case "br":
		if mc.inCell {
			mc.cellBuf = append(mc.cellBuf, " ")
		} else {
			mc.output = append(mc.output, "\n")
		}
	case "pre":
		mc.inPre = true
		mc.output = append(mc.output, "\n\n```
")
	case "ul":
		mc.listStack = append(mc.listStack, "ul")
	case "ol":
		mc.listStack = append(mc.listStack, "ol")
		mc.olCounters = append(mc.olCounters, 0)
	case "li":
		indent := strings.Repeat("  ", len(mc.listStack)-1)
		if len(mc.listStack) > 0 && mc.listStack[len(mc.listStack)-1] == "ol" {
			mc.olCounters[len(mc.olCounters)-1]++
			mc.output = append(mc.output, fmt.Sprintf("\n%s%d. ", indent, mc.olCounters[len(mc.olCounters)-1]))
		} else {
			mc.output = append(mc.output, fmt.Sprintf("\n%s- ", indent))
		}
	case "strong", "b":
		mc._append("**")
	case "em", "i":
		mc._append("*")
	case "a":
		// Extract href attribute
		var href string
		for _, attr := range attrs {
			if attr.Key == "href" {
				href = attr.Val
				break
			}
		}
		mc.href = &href
		mc.inLink = true
		mc.linkText = []string{}
	case "tr":
		mc.currentRowCells = []string{}
	case "td", "th":
		mc.inCell = true
		mc.cellBuf = []string{}
	}
}

// handleEndTag handles HTML end tags.
func (mc *markdownConverter) handleEndTag(tag string) {
	tag = strings.ToLower(tag)

	// Pop the tag stack if it matches
	if len(mc.tagStack) > 0 && mc.tagStack[len(mc.tagStack)-1] == tag {
		mc.tagStack = mc.tagStack[:len(mc.tagStack)-1]
	}

	switch tag {
	case "h1", "h2", "h3", "h4", "h5", "h6":
		level := int(tag[1] - '0')
		prefix := strings.Repeat("#", level) + " "
		textParts := []string{}
		for len(mc.output) > 0 && mc.output[len(mc.output)-1] != "\n\n" {
			textParts = append(textParts, mc.output[len(mc.output)-1])
			mc.output = mc.output[:len(mc.output)-1]
		}
		text := strings.Join(reverseStringSlice(textParts), "")
		mc.output = append(mc.output, fmt.Sprintf("%s%s\n\n", prefix, text))
	case "p":
		if !mc.inCell {
			mc.output = append(mc.output, "\n")
		}
	case "pre":
		mc.inPre = false
		mc.output = append(mc.output, "\n```
\n")
	case "ul":
		if len(mc.listStack) > 0 {
			mc.listStack = mc.listStack[:len(mc.listStack)-1]
		}
		mc.output = append(mc.output, "\n")
	case "ol":
		if len(mc.listStack) > 0 {
			mc.listStack = mc.listStack[:len(mc.listStack)-1]
		}
		if len(mc.olCounters) > 0 {
			mc.olCounters = mc.olCounters[:len(mc.olCounters)-1]
		}
		mc.output = append(mc.output, "\n")
	case "strong", "b":
		mc._append("**")
	case "em", "i":
		mc._append("*")
	case "a":
		linkText := strings.Join(mc.linkText, "")
		if mc.href != nil && linkText != "" {
			formatted := fmt.Sprintf("[%s](%s)", linkText, *mc.href)
			mc._append(formatted)
		} else {
			mc._append(linkText)
		}
		mc.inLink = false
		mc.href = nil
		mc.linkText = []string{}
	case "td", "th":
		cellText := strings.Join(mc.cellBuf, "")
		cellText = strings.Join(strings.Fields(cellText), " ")
		mc.currentRowCells = append(mc.currentRowCells, cellText)
		mc.cellBuf = []string{}
		mc.inCell = false
	case "tr":
		if len(mc.currentRowCells) > 0 {
			row := "| " + strings.Join(mc.currentRowCells, " | ") + " |"
			mc.output = append(mc.output, fmt.Sprintf("\n%s", row))
			mc.lastRowColCount = len(mc.currentRowCells)
		}
	case "thead":
		if mc.lastRowColCount > 0 {
			sep := "| " + strings.Join(strings.Repeat("---", mc.lastRowColCount), " | ") + " |"
			mc.output = append(mc.output, fmt.Sprintf("\n%s", sep))
		}
	}
}

// handleData handles HTML text data.
func (mc *markdownConverter) handleData(data string) {
	if mc.inLink {
		mc.linkText = append(mc.linkText, data)
		return
	}

	// Skip empty lines in non-pre sections
	if !mc.inPre && strings.Contains(data, "\n") && strings.TrimSpace(data) == "" {
		return
	}

	mc._append(data)
}

// getMarkdown returns the final markdown output.
func (mc *markdownConverter) getMarkdown() string {
	text := strings.Join(mc.output, "")
	text = regexp.MustCompile(`\n{3,}`).ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}

// _headingScanner collects heading level, id-attribute, and text from HTML.
type headingScanner struct {
	records []struct {
		level int
		anchor string
		text string
	}
	level *int
	anchor *string
	buf []string
}

// handleStartTag handles HTML start tags.
func (hs *headingScanner) handleStartTag(tag string, attrs []html.Attribute) {
	tag = strings.ToLower(tag)
	if len(tag) == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '6' {
		level := int(tag[1] - '0')
		hs.level = &level
		// Extract id attribute
		var anchor string
		for _, attr := range attrs {
			if attr.Key == "id" {
				hs.anchor = &attr.Val
				break
			}
		}
		hs.buf = []string{}
	}
}

// handleEndTag handles HTML end tags.
func (hs *headingScanner) handleEndTag(tag string) {
	tag = strings.ToLower(tag)
	if len(tag) == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '6' && hs.level != nil {
		text := strings.Join(hs.buf, "")
		text = strings.TrimSpace(text)
		if text != "" {
			hs.records = append(hs.records, struct {
				level int
				anchor string
				text string
			}{
				level:  *hs.level,
				anchor: *hs.anchor,
				text:   text,
			})
		}
		hs.level = nil
		hs.anchor = nil
		hs.buf = []string{}
	}
}

// handleData handles HTML text data.
func (hs *headingScanner) handleData(data string) {
	if hs.level != nil {
		hs.buf = append(hs.buf, data)
	}
}

// feed parses the HTML string.
func (hs *headingScanner) feed(html string) {
	html = _stripHtmlChrome(html)

	// Parse the HTML
	parser := html.NewTokenizer(strings.NewReader(html))
	for {
		tokenType := parser.Next()
		switch tokenType {
		case html.ErrorToken:
			return
		case html.StartTagToken, html.EndTagToken:
			token := parser.Token()
			tag := string(token.Data)
			attrs := token.Attr
			if tokenType == html.StartTagToken {
				hs.handleStartTag(tag, attrs)
			} else {
				hs.handleEndTag(tag)
			}
		}
	}
}

// reverseStringSlice reverses a slice of strings.
func reverseStringSlice(s []string) []string {
	reversed := make([]string, len(s))
	for i, j := 0, len(s)-1; i < len(s); i, j = i+1, j-1 {
		reversed[i] = s[j]
	}
	return reversed
}

// HtmlExtractor extracts plain text (as Markdown) from HTML documents.
type HtmlExtractor struct {
	// canHandle returns true if the extractor can handle the document type.
	// Returns true for "html" and "htm" document types.
	//
	// Parameters:
	// - docType: The document type to check
	//
	// Returns:
	// - bool: True if the extractor can handle the document type
	canHandle func(docType string) bool

	// extract converts HTML to Markdown.
	//
	// Parameters:
	// - data: The HTML data to convert
	// - sourcePath: The source path of the document (optional)
	//
	// Returns:
	// - string: The converted Markdown text
	extract func(data []byte, sourcePath string) string

	// annotate adds heading information to chunks.
	//
	// Parameters:
	// - chunks: The list of chunks to annotate
	// - data: The HTML data to parse
	// - sourcePath: The source path of the document (optional)
	//
	// Returns:
	// - []Chunk: The annotated chunks
	annotate func(chunks []Chunk, data []byte, sourcePath string) []Chunk
}

// NewHtmlExtractor creates a new HTML extractor.
func NewHtmlExtractor() *HtmlExtractor {
	return &HtmlExtractor{
		canHandle: func(docType string) bool {
			return docType == "html" || docType == "htm"
		},
		extract: func(data []byte, sourcePath string) string {
			text := string(data)
			return _convertHtmlToMarkdown(text)
		},
						annotate: func(chunks []Chunk, data []byte, sourcePath string) []Chunk {
			html := string(data)

			// Scan for headings
			scanner := &headingScanner{}
			scanner.feed(html)
			if len(scanner.records) == 0 {
				return chunks
			}

			// Build (anchor, heading_path) for each heading in document order
			headingStack := []struct {
				level int
				anchor string
				text string
			}{}
			sectionAnchors := []struct {
				anchor string
				path []string
			}{}
			for _, record := range scanner.records {
				// Pop headings with level >= current level
				for len(headingStack) > 0 && headingStack[len(headingStack)-1].level >= record.level {
					headingStack = headingStack[:len(headingStack)-1]
				}
				// Add current heading
				headingStack = append(headingStack, struct {
					level int
					anchor string
					text string
				}{
					level:  record.level,
					anchor: record.anchor,
					text:   record.text,
				})
				// Build path from stack
				path := []string{}
				for _, h := range headingStack {
					path = append(path, h.text)
				}
				// Find top anchor
				var topAnchor string
				for i := len(headingStack) - 1; i >= 0; i-- {
					if headingStack[i].anchor != "" {
						topAnchor = headingStack[i].anchor
						break
					}
				}
				sectionAnchors = append(sectionAnchors, struct {
					anchor string
					path []string
				}{
					anchor: topAnchor,
					path:   path,
				})
			}

			// Convert to markdown and split into segments keyed by heading
			markdown := _convertHtmlToMarkdown(html)
			headingIdx := 0
			segments := []struct {
				anchor string
				path []string
				text string
			}{}
			var currentAnchor string
			var currentPath []string
			var currentBuf []string

			for _, line := range strings.Split(markdown, "\n") {
				// Check if line is a heading
				if headingIdx < len(sectionAnchors) {
					m := regexp.MustCompile(`^(#{1,6})\s+(.*)`).FindStringSubmatch(line)
					if m != nil {
						if len(currentBuf) > 0 {
							segments = append(segments, struct {
								anchor string
								path []string
								text string
							}{
								anchor: currentAnchor,
								path:   currentPath,
								text:   strings.Join(currentBuf, "\n"),
							})
						}
						currentAnchor = sectionAnchors[headingIdx].anchor
						currentPath = sectionAnchors[headingIdx].path
						headingIdx++
						currentBuf = []string{line}
						continue
					}
				}
				// Add line to current buffer
				currentBuf = append(currentBuf, line)
			}
			// Add last segment
			if len(currentBuf) > 0 {
				segments = append(segments, struct {
					anchor string
					path []string
					text string
				}{
					anchor: currentAnchor,
					path:   currentPath,
					text:   strings.Join(currentBuf, "\n"),
				})
			}

			// Annotate chunks with heading information
			for i := range chunks {
				content := chunks[i].Content
				for _, segment := range segments {
					// Check if chunk content contains text from segment
					// We check for fragments longer than 20 characters
					for _, frag := range strings.Split(content, "\n") {
						if len(strings.TrimSpace(frag)) > 20 && strings.Contains(segment.text, strings.TrimSpace(frag)) {
							// Add heading path and anchor to chunk
							chunks[i].SourceDetail = map[string]interface{}{
								"heading_path": segment.path,
							}
							if segment.anchor != "" {
								chunks[i].SourceDetail["anchor"] = segment.anchor
							}
							break
						}
					}
				}
			}

			return chunks
		},
	
				return chunks
			}

			// Build (anchor, heading_path) for each heading in document order
			headingStack := []struct {
				level int
				anchor string
				text string
			}{}
			sectionAnchors := []struct {
				anchor string
				path []string
			}{}
			for _, record := range scanner.records {
				// Pop headings with level >= current level
				for len(headingStack) > 0 && headingStack[len(headingStack)-1].level >= record.level {
					headingStack = headingStack[:len(headingStack)-1]
				}
				// Add current heading
				headingStack = append(headingStack, struct {
					level int
					anchor string
					text string
				}{
					level:  record.level,
					anchor: record.anchor,
					text:   record.text,
				})
				// Build path from stack
				path := []string{}
				for _, h := range headingStack {
					path = append(path, h.text)
				}
				// Find top anchor
				var topAnchor string
				for i := len(headingStack) - 1; i >= 0; i-- {
					if headingStack[i].anchor != "" {
						topAnchor = headingStack[i].anchor
						break
					}
				}
				sectionAnchors = append(sectionAnchors, struct {
					anchor string
					path []string
				}{
					anchor: topAnchor,
					path:   path,
				})
			}

			// Convert to markdown and split into segments keyed by heading
			markdown := _convertHtmlToMarkdown(html)
			headingIdx := 0
			segments := []struct {
				anchor string
				path []string
				text string
			}{}
			var currentAnchor string
			var currentPath []string
			var currentBuf []string

			for _, line := range strings.Split(markdown, "\n") {
				// Check if line is a heading
				if headingIdx < len(sectionAnchors) {
					m := regexp.MustCompile(`^(#{1,6})\s+(.*)`).FindStringSubmatch(line)
					if m != nil {
						if len(currentBuf) > 0 {
							segments = append(segments, struct {
								anchor string
								path []string
								text string
							}{
								anchor: currentAnchor,
								path:   currentPath,
								text:   strings.Join(currentBuf, "\n"),
							})
						}
						currentAnchor = sectionAnchors[headingIdx].anchor
						currentPath = sectionAnchors[headingIdx].path
						headingIdx++
						currentBuf = []string{line}
						continue
					}
				}
				// Add line to current buffer
				currentBuf = append(currentBuf, line)
			}
			// Add last segment
			if len(currentBuf) > 0 {
				segments = append(segments, struct {
					anchor string
					path []string
					text string
				}{
					anchor: currentAnchor,
					path:   currentPath,
					text:   strings.Join(currentBuf, "\n"),
				})
			}

			// Annotate chunks with heading information
			for i := range chunks {
				content := chunks[i].Content
				for _, segment := range segments {
					// Check if chunk content contains text from segment
					// We check for fragments longer than 20 characters
					for _, frag := range strings.Split(content, "\n") {
						if len(strings.TrimSpace(frag)) > 20 && strings.Contains(segment.text, strings.TrimSpace(frag)) {
							// Add heading path and anchor to chunk
							chunks[i].SourceDetail = map[string]interface{}{
								"heading_path": segment.path,
							}
							if segment.anchor != "" {
								chunks[i].SourceDetail["anchor"] = segment.anchor
							}
							break
						}
					}
				}
			}

			return chunks
		},
	
		