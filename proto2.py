from pdfreader import PDFDocument, SimplePDFViewer
import timeit

start = timeit.timeit()

# get raw document
fd = open('drive/MyDrive/jonasmotta/test3.pdf', "rb")
doc = PDFDocument(fd)

# there is an iterator for pages
page_one = next(doc.pages())
all_pages = [p for p in doc.pages()]

# and even a viewer
fd = open('drive/MyDrive/jonasmotta/test3.pdf', "rb")
viewer = SimplePDFViewer(fd)

print(all_pages)
end = timeit.timeit()
print(end - start)