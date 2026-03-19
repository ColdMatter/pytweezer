import  zmq
binding="ipc:///tmp/imagesub"
print("Image server running on "+binding)
context=zmq.Context()
sock_sub = context.socket(zmq.XSUB)
sock_sub.bind(binding)
sock_pub = context.socket(zmq.XPUB)
sock_pub.bind("ipc:///tmp/imagepub")

zmq.proxy(sock_sub,sock_pub)

# clean up
sock_sub.close()
sock_pub.close()
context.term()





