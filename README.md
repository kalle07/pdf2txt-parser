# parsing / pdf parser
-> with a nice old style GUI -> have FUN !!!<br>
-> update: 08/26 <b>PDF_Parser-by-Kalle07-v8a</b><br>
-> on right side releases: windows exe available ! (no install - its selfrunning) <br>
or: https://github.com/kalle07/pdf2txt-parser/releases/download/V8a/PDFParser-by-kalle07.exe
<br>
<br>
<b>The PDF Parser is a high-performance desktop application for extracting text, images, drawings, and metadata from PDF documents. Built for speed with multi-core processing and batch conversion, it helps researchers, engineers, businesses, and developers convert large PDF collections into clean, searchable text while preserving valuable document information.<br>

The parser can also save extracted images and vector drawings as separate files. Using an external application (eg: small VL model like LFM25), these images and drawings can be automatically described, and the generated descriptions can then be injected back into the main extracted text file. This creates enriched, AI-ready documents that combine the original PDF text with meaningful descriptions of visual content, making them ideal for search, indexing, accessibility, and retrieval-augmented generation (RAG) workflows. </b><br><br>

Check the PDF before converting it to text: go to any page, ideally one at the beginning and one at the end, select the text with the mouse and copy it into an editor (can you see what you copied?)... if that doesn't work, this parser won't work and neither will any other simple program! To do this, you must remove the copy protection, or the page is just an image and you must use OCR first.<br><br>

• The generated TXT file has the same name as the PDF file.<br>
• The TXT file and a (optional) media folder with images/drawings are created in the source directory.<br>
• Two common types of tables are converted to JSON format (embedder readable)<br>
• Older TXT and images files will be overwritten without prompting.<br>
• When selecting a folder, all .pdf files inside it (non-hidden) are processed.<br>
• Instant text preview<br>
• Progress bar<br>

# <b>PDF to TXT converter ready to chunk for your RAG</b>
<b>EXE - ONLY WINDOWS</b><br>
<b>python install available, should be run everywhere</b><br>
<br>

<b>&#x21e8;</b> give me a ❤️ or ⭐, if you like  ;)<br><br>

newest: <b>PDF Parser by Kall07 </b>
<br>

<img width="1179" height="888" alt="grafik" src="https://github.com/user-attachments/assets/4a6e750f-2ddc-4711-b5a4-30c480f75b77" />

<br>



Most LLM applications only convert your PDF simple to txt, nothing more, its like you save your PDF as txt file. Often textblocks are mixed and tables not readable.
Therefore its better to convert it with some help of a <b>parser</b>.<br><br>

# Detailed description
Right-click options:<br>
• You can remove or open the source/converted PDF by right-clicking on it.<br><br>

Status indicators after processing:<br>
If:<br>
[INFO] File completed: TEST.pdf (X pages)!<br>
[INFO] Processing completed<br>
-> This only means all pages were processed; image/drawing/table quality is not guaranteed.<br>
-> If you cannot select and copy the text from the PDF, this program will produce poor results.<br>
-> No OCR or AI-based recognition — pure pymupdf extraction only.<br>
-> No formulas<br><br>

Layout & Content Rules:<br>
• An attempt is made to reproduce page layout in columns (left → right) and blocks (top → bottom).<br>
• Two common types of tables with detectable structure are extracted; headers are assigned and stored as JSON inside the TXT file.<br>
• Adds "Page X of Y" label at the beginning of every processed page.<br><br>

Image/Drawings Extraction:<br>
• Images below 100 px on any side are skipped by default (adjustable via config).<br>
• Full-page images (≥80% of page size) are excluded — likely background/scan artifacts.<br>
• Images overlapping >90% with a similarly-sized text block or table are skipped.<br>
• Max 10 media items per page to prevent cluttered output.<br><br>

Drawing Extraction:<br>
• A "drawing" requires at least 10 drawing rectangles clustered together (configurable via min_items_per_cluster).<br>
• Small text blocks near drawings may be merged into the cluster for context but do NOT count toward the minimum.<br>
• Drawings are saved with padding around their bounding box for visual clarity.<br><br>

Margin & Overlap Protection:<br>
• Content whose center falls within outer margins is skipped (configurable thresholds per side).<br>
• Tables take precedence — text blocks and drawings overlapping a table area by >90% are discarded.<br>
• Images vs Drawings conflict resolution keeps the larger item; smaller one is logged as skipped.<br><br>

Post-Processing Mode:<br>
• First: describe all images and drawings oc with help of Ai (Suggestion: LFM2.5-VL-1.6B)<br>
-> example approach: lfm25_image_describe.py<br>
• This second pass reads existing text files with-in pdf media-folder same name as the PDF and injects a description field alongside each image/drawing JSON block.<br>
example: testfile_page_0003_img_02.png -> testfile_page_0003_img_02.txt<br><br>

Stop function becomes effective only after the currently processed file finishes its page chunk.<br><br>

When processing large amounts of data, the following should be noted:<br>
1. All PDFs are opened once by PDFValidator to determine validity, protection status, and page count.<br>
2. Files with fewer than 32 pages run in parallel — one core per file (up to available cores).<br>
3. Large files (≥32 pages) are split into chunks of ~8 pages per core.<br>
4. Each page runs inside a separate ProcessPoolExecutor worker, fully isolated with its own ConversionConfig copy.<br>
5. Results from all workers are collected and assembled in original page order before writing the final TXT file.<br>
6. Speed: 8 cores  ~50 pages / sec<br><br>

...<br>
<br>
📥 Downloads: <!--download-count-->043<!--/download-count-->


<br>
<br>
This I have created with my brain and the help of Ai, Iam not a coder... sorry so I will not fulfill any wishes unless there are real errors.<br>
It is really hard for me with GUI and the Function and in addition to compile it.<br>
For the python-file oc you need to import missing libraries.<br>
<br><br>

# INSTALL
download exe, no installation, direct working App<br>
python -m venv venv<br>
venv\Scripts\activate  # On Windows<br>
pip install -r requirements.txt<br>
python main.py<br><br>


<b>now have fun and leave a comment if you like  ;)</b><br>
<br>
my raw-txt-snippet extractor<br>
https://github.com/kalle07/raw-txt-snippet-creator<br>
my embedder collection:<br>
https://huggingface.co/kalle07/embedder_collection
<br>
<br>

I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
