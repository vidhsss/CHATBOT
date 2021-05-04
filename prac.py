import urllib
print("https://www.google.com")
print('<a href ="https:google.com">google</a><br>')
from tornado.escape import linkify
post = "Check out http://kite.com!"
print (linkify(post))
