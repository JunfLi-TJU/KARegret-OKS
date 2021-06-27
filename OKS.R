setwd("E:/experiment/code") # changes the current directory as the working directory
rm(list = ls())

d_index <- 4

# set the path of datasets
dpath          <- file.path("E:/experiment/binary C") 

Dataset        <- c("w8a", "ijcnn1_all","a9a_all","w7a")

# set the path of results

savepath1      <- paste0("F:/experiment/Conference Paper/ECML/ECML2021 Result/",paste0("OKS-c-",Dataset[d_index],".txt"))
savepath2      <- paste0("F:/experiment/Conference Paper/ECML/ECML2021 Result/",paste0("OKS-c-all-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)

para1_setting <- list( 
  delta   = 0.2,
  eta     = 1/sqrt(length_tr),
  lambda  = 5/sqrt(length_tr)
)
x         <- seq(-2,3,1)
sigma     <- 2^(x)
len_sigma <- length(sigma)

para1_setting$eta <- sqrt(2*(1-para1_setting$delta)*log(len_sigma)/(len_sigma*length_tr))

p1 <- (1-para1_setting$delta)
p2 <- para1_setting$delta/len_sigma

reptimes <- 10
runtime  <- c(rep(0, reptimes))
errorrate<- c(rep(0, reptimes))
all_p    <- matrix(0,nrow = reptimes, ncol = len_sigma)
all_N    <- matrix(0,nrow = reptimes, ncol = len_sigma)
all_error<- matrix(0,nrow = reptimes, ncol = len_sigma)

all_infor<- matrix(0,nrow = reptimes, ncol = 3*len_sigma)

for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)   
  
  N        <- c(rep(0, len_sigma))  # store the selected times of each kernel parameter
  error    <- c(rep(0, len_sigma))
  L        <- c(rep(0, len_sigma))
  k        <- c(rep(0, len_sigma))
  p        <- c(rep(1/len_sigma,len_sigma))
  
  sv_coe_list <- list(
    svpara1   = array(0,1),
    svpara2   = array(0,1),
    svpara3   = array(0,1),
    svpara4   = array(0,1),
    svpara5   = array(0,1),
    svpara6   = array(0,1)
  )
  sv_max_list <- list(
    svmat1  = matrix(0,nrow = feature_tr,ncol=1),
    svmat2  = matrix(0,nrow = feature_tr,ncol=1),
    svmat3  = matrix(0,nrow = feature_tr,ncol=1),
    svmat4  = matrix(0,nrow = feature_tr,ncol=1),
    svmat5  = matrix(0,nrow = feature_tr,ncol=1),
    svmat6  = matrix(0,nrow = feature_tr,ncol=1)
  )
  
  t1    <- proc.time()  #proc.time()
  for (i in 1:length_tr)
  {
    err <- 0
    It   <- sample(1:len_sigma, 1, replace = T, prob=p)
    
    tem_svmat    <- sv_max_list[[It]]
    tem_svpara   <- sv_coe_list[[It]]
    
    N[It]<- N[It] + 1
    
    diff_S_i <- tem_svmat - trdata[order[i], ]
    tem      <- colSums(diff_S_i*diff_S_i)
    sum      <- tem_svpara %*% exp(tem/(-2*(sigma[It])^2))
    haty = 1
    if(sum < 0)
      haty = -1
    if(ylabel[order[i]]*sum < 1)
    {
      if(haty != ylabel[order[i]])
        err = 1
      #svpara[It, ] = (1-para1_setting$eta*para1_setting$lambda)*svpara[It, ] # weight
      k[It] <- k[It]+1
      if(k[It] == 1)
      {
        tem_svmat[,1] <- trdata[order[i], ]
      }else{
        tem_svmat <- cbind(tem_svmat,trdata[order[i], ])
      }
      tem_svpara[k[It]]   <- para1_setting$lambda*ylabel[order[i]]
    }
    error[It] <- error[It] + err
    L[It]     <- L[It] + err/p[It]
    p = p1*exp(-1*p2*L)/sum(exp(-1*p2*L)) + p2
    
    sv_max_list[[It]]   <- tem_svmat
    sv_coe_list[[It]]   <- tem_svpara
  }
  t2 <- proc.time()
  runtime[re]    <- (t2 - t1)[3]
  errorrate[re]  <- sum(error)/length_tr
  all_N[re,]     <- N
  all_error[re,] <- error
  all_p[re,]     <- p
}

save_result <- list(
  note     = c("the next term are:alg_name--dataname--sv_num--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_error"),
  alg_name = c("OKS-Class"),
  dataname = paste0(Dataset[d_index], ".train"),
  lambda   = para1_setting$lambda,
  run_time = as.character(runtime),
  err_num  = as.character(errorrate), 
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

all_infor[,1:len_sigma] = all_N[,1:len_sigma]
all_infor[,(len_sigma+1):(2*len_sigma)] = all_p[,1:len_sigma]
all_infor[,(2*len_sigma+1):(3*len_sigma)] = all_error[,1:len_sigma]

write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 
write.table(all_infor,file=savepath2,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the candidate kernel parameter are :")
sprintf("%.5f", sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("total training time is %.4f in dataset", sum(runtime))
sprintf("average training time is %.5f in dataset", sum(runtime)/reptimes)
sprintf("the average error rate is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of average running time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of average error is %.5f in dataset", sd(errorrate))
