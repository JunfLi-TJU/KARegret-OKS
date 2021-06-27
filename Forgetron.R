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
  B     = 500,
  U     = 0.25*sqrt((500+1)/log(500+1))
)
# -2 -1 0 1 2 3
sigma     <- 2^(2)
threshold        <- (para1_setting$B+1)^(-0.5/(para1_setting$B+1))

reptimes  <- 10
runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))

for( re in 1:reptimes)
{
  order      <- sample(1:length_tr,length_tr,replace = F)   #dis
  k          <- 0
  norm_ft    <- 0
  error      <- 0
  
  svmat      <- matrix(0,nrow = feature_tr,ncol=1)
  sv_index   <- array(0,1)
  svpara     <- array(0,1)
  t1         <- proc.time()  #proc.time()
  
  ### the first instance
  error      <- 1
  svmat[,1]  <- trdata[order[1], ]
  sv_index[1]<- order[1]
  svpara[1]  <- ylabel[order[1]]
  norm2_fi   <- 1
  phi_i      <- min(threshold, para1_setting$U/sqrt(norm2_fi))
  svpara     <- phi_i*svpara
  k          <- 1
  norm_ft    <- phi_i*norm2_fi
  
  ### from the second instance
  for (i in 2:length_tr)
  {
    err   <- 0
    diff  <- svmat[,1:k]- trdata[order[i], ]
    if(k>1)
    {
      tem   <- colSums(diff*diff)
    }else{
      tem   <- sum((svmat[,1:k]- trdata[order[i], ])*(svmat[,1:k]- trdata[order[i], ]))
    }
    sum   <- crossprod(svpara[1:k],exp(tem/(-2*(sigma)^2)))
    fx <- sum[1,1]
    hatyi <- 1
    if(fx < 0)
      hatyi  <- -1
    if(hatyi != ylabel[order[i]])
    {
      error <- error + 1
    }
    if(fx*ylabel[order[i]] <= 0)
    {
      if(k == para1_setting$B)
      {
        norm2_fi     <- norm_ft^2 + 2*ylabel[order[i]]*fx + 1
        phi_i        <- min(threshold, para1_setting$U/sqrt(norm2_fi))
        
        tem_sv       <- svmat
        tem_sv_index <- sv_index
        tem_svpara   <- svpara
        
        sv1          <- svmat[,1]       #store the first support vector
        sv_index1    <- sv_index[1]
        svpara1      <- svpara[1]
        
        svmat[,1:(k-1)]   <- tem_sv[,2:k]
        sv_index[1:(k-1)] <- tem_sv_index[2:k]
        svpara[1:(k-1)]   <- tem_svpara[2:k]
        
        svmat[,k]         <- trdata[order[i],]
        sv_index[k]       <- order[i]
        svpara[k]         <- ylabel[order[i]]
        
        #update the norm_ft after remove the first sv
        ter_1     <- 1
        ter_2     <- svpara1^2
        ter_3     <- 2*ylabel[order[i]]*fx
        diff      <- tem_sv - sv1
        tem_ft_jt <- colSums(diff*diff)
        fx_jt     <- as.numeric(crossprod(tem_svpara,exp(tem_ft_jt/(-2*(sigma^2)))))
        ter_4     <- -2*svpara1*fx_jt
        tem_jt_t  <- sum((sv1-trdata[order[i],])*(sv1-trdata[order[i],]))
        ter_5     <- -2*ylabel[order[i]]*svpara1*exp(tem_jt_t/(-2*(sigma^2)))
        norm_ft   <- phi_i*sqrt(norm_ft^2+ter_1+ter_2+ter_3+ter_4+ter_5)
        svpara    <- phi_i*svpara
      }
      else
      {
        k           <- k+1
        svmat       <- cbind(svmat,trdata[order[i],])
        sv_index[k] <- order[i]
        svpara[k]   <- ylabel[order[i]]
        norm2_fi    <- norm_ft^2 + 2*ylabel[order[i]]*fx + 1
        phi_i       <- min(threshold, para1_setting$U/sqrt(norm2_fi))
        svpara      <- phi_i*svpara
        norm_ft     <- phi_i*sqrt(norm2_fi)
      }
    }
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  errorrate[re] <- sum(error)/length_tr
}

save_result <- list(
  note     = c("the next term are:alg_name--dataname--sam_num--sigma--sv_num--run_time--err_num--tot_run_time--ave_run_time--ave_err_rate--sd_time--sd_err"),
  alg_name = c("Forgetron-"),
  dataname = paste0(Dataset[d_index], ".train"),
  sam_num  = length_tr,
  ker_para = sigma,
  sv_num   = para1_setting$B,
  run_time = as.character(runtime),
  err_num = errorrate,
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the candidate kernel parameter are :")
sprintf("%.5f", sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("total training time is %.4f in dataset", sum(runtime))
sprintf("average training time is %.5f in dataset", sum(runtime)/reptimes)
sprintf("the average error rate is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of average running time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of average error is %.5f in dataset", sd(errorrate))
