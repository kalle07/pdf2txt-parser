# parsing / pdf parser
with GUI<br>
update: 11/25 <b>PDF_Parser-Sevenof9_v7f</b>, cleaner lines (much better handling with hyphen, will be merged on line-end), faster word to text-block algorithm ~20% faster at all.<br>
Check the PDF before converting it to text: go to any page, ideally one at the beginning and one at the end, select the text with the mouse and copy it into an editor (can you see what you copied?)... if that doesn't work, this parser won't work and neither will any other program! To do this, you must remove the copy protection, or the page is just an image and you must use OCR first.<br>

# <b>PDF to TXT converter ready to chunk for your RAG</b>
<b>ONLY WINDOWS</b><br>
<b>EXE and PY available (en)</b><br>
exe files aviable on hugging (or relases -> right side): <br>
https://huggingface.co/kalle07/pdf2txt_parser_converter
<br>

<b>&#x21e8;</b> give me a ❤️, if you like  ;)<br><br>

newest: <b>PDF Parser - Sevenof9_v7f.py</b>
<br>

<img width="1232" height="991" alt="grafik" src="https://github.com/user-attachments/assets/e6596e77-52cb-45c6-9666-3a360d75a38e" />
<br>



Most LLM applications only convert your PDF simple to txt, nothing more, its like you save your PDF as txt file. Often textblocks are mixed and tables not readable.
Therefore its better to convert it with some help of a <b>parser</b>.<br>
I work with "<b>pdfplumber/pdfminer</b>" none OCR(no images) and the PDF must contain copyable text.<br>
<ul style="line-height: 1.05;">
<li>Works with single and multi PDF list, works with folder</li>
<li>Intelligent multiprocessing ~10-30 pages per second</li>
<li>Error tolerant, that means if your PDF is not convertible, it will be skipped, no special handling</li>
<li>Instant view of the result, hit one pdf on top of the list</li>
<li>Removes about 5% of the margins around the page</li>
<li>Converts some common tables as json inside the txt file</li>
<li>Add the absolute PAGE number to each page</li>
<li>Add the tag “chapter” or “important” to large and/or bold font.</li>
<li>All txt files will be created in original folder of PDF, same name as *.txt</li>
<li>All txt files will be overwritten if you start converting with same PDF</li>
<li>If there are many text blocks on a page, it may be that text blocks that you would read first appear further down the page. (It is a compromise between many layout options)</li>
<li>Small blocks of text (such as units or individual numbers), usually near diagrams and sketches, appear at the end of each page</li>
<li>I advise against using a PDF file directly for RAG formatting (embedding), as you never know how it will look, and incorrect input can lead to poor results</li>
<li>tested on 300 PDF files ~30000 pages</li>
</ul>

<br>
This I have created with my brain and the help of Ai, Iam not a coder... sorry so I will not fulfill any wishes unless there are real errors.<br>
It is really hard for me with GUI and the Function and in addition to compile it.<br>
For the python-file oc you need to import missing libraries.<br>
<br><br>
INSTALL:
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python version_xyz.py


<b>now have fun and leave a comment if you like  ;)</b><br>
on discord "sevenof9"
<br>
my raw-txt-snippet extractor<br>
https://github.com/kalle07/raw-txt-snippet-creator<br>
my embedder collection:<br>
https://huggingface.co/kalle07/embedder_collection

<br>
<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
