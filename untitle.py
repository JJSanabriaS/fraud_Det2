from pdfquery import PDFQuery
import timeit
start = timeit.timeit()
pdf = PDFQuery('test3.pdf')
pdf.load()

# Use CSS-like selectors to locate the text elements
text_elements = pdf.pq('LTTextLineHorizontal')

# Extract the text from the elements
text = [t.text for t in text_elements]
print(text)
end = timeit.timeit()
print(end - start)