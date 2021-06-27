#setwd("E:/experiment/ECML/code")
rm(list = ls())

d_index <- 4

# set the path of datasets
dpath          <- file.path("E:/experiment/binary C") 

Dataset        <- c("w8a", "ijcnn1_all","a9a_all","w7a")

# set the path of results
savepath1      <- paste0("E:/experiment/ECML/ECML2021 Result/",paste0("BOGD-c-",Dataset[d_index],".txt"))
traindatapath  <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1]
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)              

para1_setting <- list( 
  eta    = 5/sqrt(length_tr),
  lambda = 2^(3)/length_tr^2,
  gamma  = 2^0,   ## 2^0, 2^1, 2^2, 2^3,2^4
  B      = 500
)
# -2 -1 0 1 2 3
sigma    <- 2^(-1)

reptimes <- 3
runtime  <- c(rep(0, reptimes))
errorrate<- c(rep(0, reptimes))


for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)   #dis
  svpara   <- array(0,1)    # store the parameter for each support vector
  error    <- 0
  j        <- 0
  k        <- 0
  palpha   <- para1_setting$B/(para1_setting$B-1)
  svmat    <- matrix(0,nrow = feature_tr,ncol=1)
  t1       <- proc.time()  #proc.time()
  for (i in 1:length_tr)
  {
    err <- 0
    if(k>0)
    {
      diff_S_i  <- svmat[,1:k] - trdata[order[i], ]
      if(k>1)
      {
        tem       <- colSums(diff_S_i*diff_S_i)
      }else
      {
        tem       <- sum(diff_S_i*diff_S_i)
      }
    }else{
      diff_S_i  <- svmat[,1] - trdata[order[i], ]
      tem       <- sum(diff_S_i*diff_S_i)
    }
    
    fx        <- (svpara[1:k] %*% exp(tem/(-2*(sigma^2))))[1,1]
    hatyi     <- 1     
    if(fx < 0 )  
      hatyi <- -1
    if(hatyi != ylabel[order[i]])
      err  <- 1
    svpara <- (1-para1_setting$eta*para1_setting$lambda)*svpara
    if(fx*ylabel[order[i]] < 1)
    {
      if(k  >= para1_setting$B)
      {
        dis         <- sample(1:k,1,replace = T)     #uniform sample
        svmat[,dis] <- trdata[order[i],]
        svpara      <- svpara*palpha                 # svpara/(1/para1_setting$B)
        svpara[dis] <- para1_setting$eta*ylabel[order[i]]
        maxsvpara   <- max(abs(svpara))
        svpara      <- svpara*min((para1_setting$eta*para1_setting$gamma)/maxsvpara,1)
      }
      else
      {
        if(k==0)
        {
          svmat[,1] <- trdata[order[i],]
        }else{
          svmat <- cbind(svmat,trdata[order[i],])
        }
        svpara[k+1] <- para1_setting$eta*ylabel[order[i]]
        k  <- k+1
      }
    }
    error <- error + err
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  errorrate[re] <- sum(error)/length_tr
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--sam_num--sigma--sv_num--gamma--eta--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_error"),
  alg_name = c("BOGD"),
  dataname = paste0(Dataset[d_index], ".train"),
  sam_num  = length_tr,
  ker_para = sigma,
  sv_num   = para1_setting$B,
  gamma    = para1_setting$gamma,
  eta      = para1_setting$eta,
  run_time = as.character(runtime),
  err_num  = as.character(errorrate), 
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the candidate kernel parameter are :")
sprintf("the number of sample is %d", length_tr)
sprintf("total training time is %.4f in dataset", sum(runtime))
sprintf("average training time is %.5f in dataset", sum(runtime)/reptimes)
sprintf("the average error rate is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of average running time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of average error is %.5f in dataset", sd(errorrate))

