import  zmq
subbind="ipc:///tmp/propertysub"
print("Property server running on "+subbind)
context=zmq.Context()
sock_sub = context.socket(zmq.XSUB)
sock_sub.bind(subbind)
sock_pub = context.socket(zmq.XPUB)
sock_pub.bind("ipc:///tmp/propertypub")

zmq.proxy(sock_sub,sock_pub)

# clean up
sock_sub.close()
sock_pub.close()
context.term()





