import  zmq
print("dataserver up and running")
context=zmq.Context()
sock_sub = context.socket(zmq.XSUB)
sock_sub.bind("ipc:///tmp/datasub")
sock_pub = context.socket(zmq.XPUB)
sock_pub.bind("ipc:///tmp/datapub")

zmq.proxy(sock_sub,sock_pub)

# clean up
sock_sub.close()
sock_pub.close()
context.term()





