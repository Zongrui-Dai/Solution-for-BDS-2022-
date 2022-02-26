
#######################################################
#              Runge_Kutta functions                  #
#######################################################
function_sets<-function(E,S,ES,P,k1,k2,k3){
  dE<--k1*E*S+(k2+k3)*ES
  dS<--k1*E*S+k2*ES
  dES<-k1*E*S-(k2+k3)*ES
  dP<-k3*ES
  
  return(c(dE,dS,dES,dP))
}

Runge_Kutta<-function(E,S,ES,P,k1,k2,k3,h,init){
  K=init
  Kmatrix=K
  
  init2<-0.5*h*K
  K2<-function_sets(E+init2[1],S+init2[2],ES+init2[3],P+init2[4],k1,k2,k3)
    
  init3<-0.5*h*K2
  K3<-function_sets(E+init3[1],S+init3[2],ES+init3[3],P+init3[4],k1,k2,k3)

  init4<-h*K3
  K4<-function_sets(E+init4[1],S+init4[2],ES+init4[3],P+init4[4],k1,k2,k3)
  
  Kmatrix<-rbind(Kmatrix,K2,K3,K4)
  Addition<-h*(Kmatrix[1,]+2*Kmatrix[2,]+2*Kmatrix[3,]+Kmatrix[4,])/6
  
  E<-E+Addition[1]
  S<-S+Addition[2]
  ES<-ES+Addition[3]
  P<-P+Addition[4]
  
  return(c(E,S,ES,P))
}


Solution<-function(E,S,ES,P,k1,k2,k3,t=100,h=0.001){
  init<-function_sets(E,S,ES,P,k1,k2,k3)
  Con<-c(E,S,ES,P)
  tmatrix<-Con
  Speed<-init
  
  for(i in 1:t){
    yt<-Runge_Kutta(Con[1],Con[2],Con[3],Con[4],k1,k2,k3,h,init)
    tmatrix<-rbind(tmatrix,yt)
    
    Con<-yt
    init<-function_sets(Con[1],Con[2],Con[3],Con[4],k1,k2,k3)
    
    Speed<-rbind(Speed,init)
  }
  
  l<-list(tmatrix,Speed)
  
  return(l)
}

#######################################################
#                     Drawing                         #
#######################################################
#ggplot2 for figure.1
w<-Solution(1,10,0,0,100,600,150,h=0.001,t=500)
library(ggplot2)
d<-as.data.frame(w[[1]])
d$time<-1:length(d$V1)
p<-ggplot(d)+
  geom_line(aes(time,V1,color='red'))+
  geom_line(aes(time,V2,color='green'))+
  geom_line(aes(time,V3,color='blue'))+
  geom_line(aes(time,V4,color='black'))
p

#plot for figure.2
S<-1:10
plot(S,150*S/(7.5+S),col='red',type='o',xlim=c(0,10),ylim=c(0,100),xlab = ' ',ylab=' ')
par(new=T)
plot(w[[1]][,2],w[[2]][,4],col='black',
     xlim=c(0,10),ylim=c(0,100),xlab='Concentration of the substrate(um)',ylab='V(um/min)')
legend("top",                                    
        legend=c("Real","Runge-Kutta"),        
        col=c('red','black'),                 
        lty=1,lwd=2,bty="n")    

#Linear regression
lm(as.numeric(1/w[[2]][-1,4])~as.numeric(1/w[[1]][-1,2]))

#plot for figure.3
par(mfrow=c(1,2))
plot(as.numeric(1/w[[1]][-1,2]),as.numeric(1/w[[2]][-1,4]),
     main='y=0.006919+0.048798*x',xlab='1/[S]',ylab='1/Vp')
plot(S,150*S/(7.5+S),col='red',type='o',xlim=c(0,10),ylim=c(0,100),
     xlab = '[S]',ylab='V',main = 'V=150*[S]/(7.5+[S])')

