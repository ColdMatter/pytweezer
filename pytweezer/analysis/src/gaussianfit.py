

#INPUT:   
#x values
#y values
#fitfunction = f(p,x) p is a list of parameters, x must also be treated as list
#fitstart  starting values
# RETURN
#fit.beta  list of fitted values
#fit.sd_beta  list of corresponding standard deviations
def mfit(x,y,ffunc,fitstart):
  Tx=asarray(x)
  Ty=asarray(y)
  my_data= odr.Data(Tx,Ty);
  mymodel=odr.Model(ffunc)
  myodr=odr.ODR(my_data,mymodel,beta0=fitstart)
  myodr.set_job(fit_type=2)
  fit=myodr.run()
  for i in range(0,len(fit.beta)):
    g=i
#    print "%0.3f" % fit.beta[i], "%0.3f" % fit.sd_beta[i]
  return fit

