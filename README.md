# PDF Data Extraction Project Documentation

## What I Have Done

### Architecture

Built PDF processing tools using CrewAI framework with three separate extraction pipelines:

- **Image extraction** (PDFImageTextExtractionTool) - for UI images, logos, diagrams with structured text extraction from UI elements
- **Table extraction** (PDFTableExtractionTool) - for structured tabular data
- **Text extraction** (PDFTextExtractionTool) - for body text, paragraphs, headers

### Technical Implementation

- PDF processing using pdfplumber
- Image OCR using OpenAI's GPT-4o-mini vision API
- Text-to-JSON conversion via LLM prompting
- Schema validation using jsonschema
- Base64 image encoding for vision models
- Table boundary detection to filter text
- Document format support: PDF and DOC/DOCX

### Current Capabilities

- Extract images from PDFs (UI elements, diagrams, logos)
- Extract text from UI elements in a structured manner (buttons, labels, form fields, menus)
- Parse UI component hierarchy and relationships
- Extract text while ignoring tables and images
- Extract table data as structured arrays
- Map extracted content to predefined JSON schemas
- Validate output against JSON schemas
- Handle single-content-type documents (text-only, table-only, image-only)

### Known Limitations

- No unified pipeline - tools run independently
- No output merging - results remain separate
- Validation detects failures but doesn't trigger retries
- No reflection or self-correction mechanisms
- No content-type detection or routing logic
- Returns invalid data when validation fails

## What I Am Planning

### Phase 1: Unified Multi-Content Pipeline

**Content detection module:** Analyze PDF to identify what it contains
- Detect presence of: UI images with text elements, body text, tables, headers, shapes, UI components
- Return content type flags: `{has_images: true, has_ui_elements: true, has_tables: true, has_text: true}`

**Routing logic:** Invoke appropriate tools based on detection
- All three tools if PDF has mixed content
- Single tool if document is homogeneous

**Sequential output merging:** Combine results in structured format

```json
{
  "images": [...],
  "ui_elements": [
    {
      "type": "button",
      "text": "Submit",
      "position": {"x": 100, "y": 200}
    },
    {
      "type": "label",
      "text": "Username:",
      "associated_field": "input_1"
    }
  ],
  "tables": [...],
  "text": [...],
  "merged": true
}
```

### Phase 2: Document Format Handler

- Support for DOC/DOCX with image-only content
  - Convert DOC/DOCX to images
  - Route to Image Text Extraction Tool
- Format-agnostic processing pipeline

### Phase 3: Intelligent Tool Selection

- Orchestrator agent decides which tools to use
- Parallel execution for independent content types
- Smart merging preserving document structure order

### Phase 4: Add Self-Correction

- Validation feedback loops (3 retry attempts per tool)
- Agent analyzes validation errors and refines prompts
- Cross-tool consistency checks

### Phase 5: Memory & Learning

- Remember successful extraction patterns per document type
- Learn optimal tool combinations
- Track which schemas work for which content layouts

## Progress Made (Local)

### Working Features

- Three independent extraction tools functional
- Each tool produces valid JSON output
- Schema validation logic present
- Individual content type extraction reliable

### Tested Scenarios

- PDFs with only images
- PDFs with only tables
- PDFs with only text
- Mixed-content PDFs (manual tool invocation)

### Current Gaps

- No automatic content detection
- No unified entry point
- No output merging implemented
- Tools must be called manually for mixed PDFs
- Results stored separately, not combined

## Expected Final Output (Minimum Viable)

### Core Deliverable

Intelligent Multi-Content PDF Extraction System that:

**Accepts:**
- PDF/DOC/DOCX file path
- Target JSON schema (optional, per content type)
- Auto-detection mode (default: ON)

**Processes:**
- Analyzes document to detect content types
- Routes to appropriate extraction tool(s)
- Extracts all content types present
- Merges outputs sequentially (preserving document order)
- Validates merged output against schema

**Returns:**

```json
{
  "success": true,
  "document_type": "mixed_content",
  "content_detected": {
    "images": true,
    "ui_elements": true,
    "tables": true,
    "text": true,
    "headers": true,
    "shapes": false
  },
  "data": {
    "images": [
      {"page": 1, "content": "..."}
    ],
    "ui_elements": [
      {
        "page": 1,
        "type": "form",
        "components": [
          {"type": "label", "text": "Email Address:", "position": "top-left"},
          {"type": "input", "placeholder": "Enter email", "field_name": "email"},
          {"type": "button", "text": "Sign Up", "action": "submit"}
        ]
      }
    ],
    "tables": [
      {"page": 2, "content": [...]}
    ],
    "text": {
      "headers": ["..."],
      "body": "..."
    }
  },
  "metadata": {
    "tools_used": ["image_extraction", "table_extraction", "text_extraction"],
    "extraction_order": ["text", "tables", "images"],
    "total_pages": 5,
    "validation_status": "passed"
  }
}
```

### Use Cases

- **Mixed PDF:** Invoice with logo (image) + line items (table) + terms (text) → all three tools → merged output
- **UI Mockup PDF:** Design mockup with buttons, labels, form fields → structured UI element extraction with text, positions, and relationships
- **Text-only DOC:** Contract document → text extraction tool only
- **UI screenshot in DOCX:** Design mockup → convert to image → image extraction tool with structured UI text extraction
- **Data report PDF:** Only tables → table extraction tool only
- **Mobile App UI PDF:** App screen with navigation menus, buttons, input fields → structured extraction of all UI text components

### Success Metrics

- Content detection accuracy: 95%+
- Correct tool selection: 100%
- Output merging success: 90%+ maintaining document structure
- Processing time: <45 seconds for mixed 10-page PDF

### Iteration Plan

1. Build content detection module first
2. Implement routing logic
3. Create output merger with configurable ordering
4. Test on diverse document types
5. Add cross-content validation (e.g., table references in text match extracted tables)
6. Optimize parallel vs sequential execution

## Architecture Roadmap

### Current State (v0.1)
```
PDF → Manual Tool Selection → Single Output → No Merging
```

### Target State (v1.0)
```
PDF/DOC → Content Detector → Orchestrator → [Tool 1, Tool 2, Tool 3] → Output Merger → Unified JSON
```

### Future State (v2.0)
```
Document → Analyzer → Planner → Multi-Agent Execution → Validator → Self-Corrector → Final Output
```

---

**Note:** Current implementation is 12-14% agentic. Target v1.0 aims for 40-50% agentic with intelligent routing and merging. Target v2.0 aims for 70-80% agentic with full autonomy, reflection, and learning.
