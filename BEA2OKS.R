setwd("E:/experiment/code") # changes the current directory as the working directory
rm(list = ls())


d_index <- 4

# set the path of datasets
dpath          <- file.path("E:/experiment/binary C") 

Dataset        <- c("w8a", "ijcnn1_all","a9a_all","w7a")

# set the path of results

savepath1      <- paste0("E:/experiment/ECML/ECML2021 Result/",paste0("BEA2OKS-c-",Dataset[d_index],".txt"))
savepath2      <- paste0("E:/experiment/ECML/ECML2021 Result/",paste0("BEA2OKS-c-all-",Dataset[d_index],".txt"))

traindatapath <- file.path(dpath, paste0(Dataset[d_index], ".train"))

traindatamatrix <- as.matrix(read.table(traindatapath))
trdata     <- traindatamatrix[ ,-1] 
ylabel     <- traindatamatrix[ ,1]                                             

length_tr  <- nrow(trdata)                                               
feature_tr <- ncol(trdata)              

para1_setting <- list( 
  lambda = 5/sqrt(length_tr),
  M      = round(log(length_tr)), ######### size of reservior, V
  varepsilon = 1,
  beta       = 6^(3/2),
  B          = 400
)

reptimes   <- 10

power      <- seq(-2,3,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)
maxL       <- 0


X          <- 1
U          <- 20
tau        <- 1
L_min      <- 4^(tau)
eta_EA     <- log(1+sqrt(2*log(len_sigma)/L_min))

runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))
restart_time <- c(rep(0, len_sigma))
restart    <- matrix(0, nrow=reptimes ,ncol=len_sigma)
budget     <- matrix(0, nrow=reptimes ,ncol=len_sigma)
all_p      <- matrix(0, nrow=reptimes ,ncol=len_sigma)

for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)   #dis
  Mistake  <- 0
  k        <- c(rep(0, len_sigma))
  p        <- c(rep(1/len_sigma,len_sigma))
  L        <- c(rep(0,len_sigma))
  hatL     <- 0
  
  sv_coe_list <- list(
    svpara1   = array(0,1),
    svpara2   = array(0,1),
    svpara3   = array(0,1),
    svpara4   = array(0,1),
    svpara5   = array(0,1),
    svpara6   = array(0,1)
  )
  sv_index_list <- list(
    sv_index1  = array(0,1),
    sv_index2  = array(0,1),
    sv_index3  = array(0,1),
    sv_index4  = array(0,1),
    sv_index5  = array(0,1),
    sv_index6  = array(0,1)
  )
  sv_max_list <- list(
    svmat1  = matrix(0,nrow = feature_tr,ncol=1),
    svmat2  = matrix(0,nrow = feature_tr,ncol=1),
    svmat3  = matrix(0,nrow = feature_tr,ncol=1),
    svmat4  = matrix(0,nrow = feature_tr,ncol=1),
    svmat5  = matrix(0,nrow = feature_tr,ncol=1),
    svmat6  = matrix(0,nrow = feature_tr,ncol=1)
  )
  V         <- matrix(0,nrow = feature_tr,ncol=1)                   ######## reservoir
  
  Index_BudResSam    <- array(0,1)                                 ######## the index of instance in V
  Grad_BudResSam     <- array(0,1)
  Num_Res            <- 0                                          ######## record the size of reservoir, v
  Norm_ave_grad      <- c(rep(0, len_sigma))                       ######## record the norm of average gradient
  Norm_overlin_f_t   <- c(rep(0, len_sigma))                       ######## record the norm of \vertlone{f}_t
  Norm_overlin_f_t_  <- c(rep(0, len_sigma))                       ######## record the norm of \overline{f'}_t
  Norm_f_t_          <- c(rep(0, len_sigma))                       ######## record the norm of f'_t
  Inner_f_t_avGra    <- c(rep(0, len_sigma))                       ######## inner between f'_t and reservoir estimator
  delta_t            <- 0
  updated_vector     <- matrix(0,nrow=len_sigma,ncol=2)            ## updated_vector[r,1]:= yf(x)<\varepsilon                                                           ## updated_vector[r,2]:= b_{t,r}
  aver_G_nabla_t_    <- matrix(0,nrow=len_sigma,ncol=2)
  pt                 <- c(rep(0, len_sigma))
  
  t1       <- proc.time()                                          ######## proc.time()
  for (i in 1:length_tr)
  {
    if((L_min+para1_setting$varepsilon+X*U) > 4^(tau))
    {
      tau      <- tau+1
      L_min    <- 4^(tau)
      eta_EA   <- log(1+sqrt(2*log(len_sigma)/L_min))
      L        <- c(rep(0,len_sigma))
    }
    
    It          <- sample(1:len_sigma, 1, replace = T, prob = p)
    diff_V_i    <- V - trdata[order[i], ]
    colsum_in_V <- colSums(diff_V_i*diff_V_i)
    
    for(r in 1:len_sigma)
    {
      if(k[r]>para1_setting$B)
      {
        #### restart the sub-algorithm
        sv_index_list[[r]] <- array(0,1)
        sv_coe_list[[r]]   <- array(0,1)
        sv_max_list[[r]]   <- matrix(0,nrow = feature_tr,ncol=1)
        k[r]               <- 0
        Inner_f_t_avGra[r] <- 0
        Norm_f_t_[r]       <- 0
        restart_time[r]    <- restart_time[r]+1
      }
      sv_index    <- sv_index_list[[r]]
      svpara      <- sv_coe_list[[r]]
      svmat       <- sv_max_list[[r]]
      
      diff_S_i    <- svmat - trdata[order[i], ]
      colsum_in_S <- colSums(diff_S_i*diff_S_i)
      kvalue_V    <- exp(colsum_in_V/(-2*(sigma[r]^2)))
      kvalue_S    <- exp(colsum_in_S/(-2*(sigma[r]^2)))
      
      coe_fx_S    <- svpara
      coe_fx_V    <- para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)      ######## the first mirror updating
      fx          <- coe_fx_S %*% kvalue_S + coe_fx_V%*%kvalue_V
      aver_G_nabla_t_[r,1] <- aver_G_nabla_t_[r,2]
      aver_G_nabla_t_[r,2] <- as.numeric(Grad_BudResSam %*%kvalue_V)
      
      
      ######################## updating Inner_f_t_avGra
      if(i>= 2 && i<=(para1_setting$M+1) && sv_index[1]>0)
      {
        sum_tem_fx  <- 0
        for(indx in 1:Num_Res)
        {
          diff_S_V         <- svmat - V[,indx]
          tem_colsum_in_V  <- colSums(diff_S_V*diff_S_V)
          sum_tem_fx       <- sum_tem_fx+Grad_BudResSam[indx]*coe_fx_S %*% exp(tem_colsum_in_V/(-2*(sigma[r]^2)))
        }
        Inner_f_t_avGra[r]    <- -1*sum_tem_fx/Num_Res
      }
      if(i>=(para1_setting$M+2) && sv_index[1]>0)
      {
        if(delta_t==0)
        {
          if((updated_vector[r,1]*updated_vector[r,2])==1)
          {
            Ter_1           <- para1_setting$lambda/pt[r]*ylabel[order[i-1]]*aver_G_nabla_t_[r,1]/Num_Res
            Ter_2           <- para1_setting$lambda*(1-1/pt[r])*Norm_ave_grad[r]^2
            Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]-Ter_1-Ter_2
            if(Norm_overlin_f_t_[r]>U)
            {
              Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]*U/Norm_overlin_f_t_[r]
            }
          }
          if(updated_vector[r,1] == 1 && updated_vector[r,2]==0)
          {
            Inner_f_t_avGra[r] <-Inner_f_t_avGra[r] - para1_setting$lambda*Norm_ave_grad[r]^2
            if(Norm_overlin_f_t_[r]>U)
            {
              Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]*U/Norm_overlin_f_t_[r]
            }
          }
        }else{
          sum_tem_fx  <- 0
          for(indx in 1:Num_Res)
          {
            diff_S_V         <- svmat - V[,indx]
            tem_colsum_in_V  <- colSums(diff_S_V*diff_S_V)
            sum_tem_fx       <- sum_tem_fx+Grad_BudResSam[indx]*coe_fx_S %*% exp(tem_colsum_in_V/(-2*(sigma[r]^2)))
          }
          Inner_f_t_avGra[r]    <- -1*sum_tem_fx/Num_Res
        }
      }
      
      ############ compute the norm of f_t
      ter_1          <- (para1_setting$lambda^2)*(Norm_ave_grad[r]^2)
      ter_2          <- 2*para1_setting$lambda*Inner_f_t_avGra[r]
      Norm_overlin_f_t[r]  <- sqrt(Norm_f_t_[r]^2 + ter_1 - ter_2)
      if(Norm_overlin_f_t[r] > U)
      {
        fx       <- fx*U/Norm_overlin_f_t[r]
      }
      haty  <- 1     
      if(fx < 0)  
        haty <- -1
      if(r==It && haty != ylabel[order[i]])
      {
        Mistake <- Mistake + 1
      }
      updated_vector[r,] <- c(0,0)
      if(fx*ylabel[order[i]] < para1_setting$varepsilon)
      {
        updated_vector[r,1] <- 1
        if(maxL < (para1_setting$varepsilon-fx*ylabel[order[i]]))
        {
          maxL <- para1_setting$varepsilon-fx*ylabel[order[i]]
        }
        L[r]    <- L[r] + (para1_setting$varepsilon-fx*ylabel[order[i]])/(para1_setting$varepsilon+X*U)
        k_xy    <- Norm_ave_grad[r]^2 + 1- 2*ylabel[order[i]]*aver_G_nabla_t_[r,2]/max(Num_Res,1)
        Zt      <- para1_setting$beta*(sqrt(k_xy)+Norm_ave_grad[r])
        pt[r]   <- as.numeric(sqrt(k_xy)/Zt)
        bt      <- rbinom(1,1,pt[r])
        if(bt==1)         #### add the current support vector into S_r
        {
          updated_vector[r,2] <- 1
          if(k[r]==0)
          {
            svmat[,1] <- trdata[order[i],]          ## updating the budget
          }else
          {
            svmat     <- cbind(svmat,trdata[order[i],]) ## updating the budget
          }
          k[r]            <- k[r]+1
          sv_index[k[r]]  <- order[i]
          svpara[k[r]]    <- para1_setting$lambda*ylabel[order[i]]/pt[r]
          Coef_BudResSam  <- (1-1/pt[r])*para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)
          Inser_SvElement <- intersect(Index_BudResSam,sv_index) ## find the same support in S_r and V
          if(length(Inser_SvElement)>0 && Inser_SvElement[1]>0)
          {
            Inser_SvIndex <- match(Inser_SvElement,sv_index) ## the location of the same support vectors in S_r
            for(h in Inser_SvIndex)
            {
              inx        <- match(sv_index[h],Index_BudResSam)
              svpara[h]  <- svpara[h]+Coef_BudResSam[inx]
            }
          }
          ## find the support vector in V but not in S_r, and add them into S_r
          Diff_SvIndex <- setdiff(Index_BudResSam,sv_index)
          if(length(Diff_SvIndex)>0&&Diff_SvIndex[1]>0)
          {
            for(h in 1:length(Diff_SvIndex))
            {
              inx <- match(Diff_SvIndex[h],Index_BudResSam)
              svmat            <- cbind(svmat,V[,inx])
              sv_index[k[r]+h] <- Index_BudResSam[inx]
              svpara[k[r]+h]   <- Coef_BudResSam[inx]
            }
            k[r] <- k[r]+length(Diff_SvIndex)
          }
          ###### compute the norm of f'_t
          Ter_1 <- para1_setting$lambda^2/(pt[r]^2)
          Ter_2 <- 2*para1_setting$lambda/pt[r]*coe_fx_S %*% kvalue_S*ylabel[order[i]]
          Ter_3 <- 2*para1_setting$lambda^2*(1-1/pt[r])/pt[r]*aver_G_nabla_t_[r,2]*ylabel[order[i]]/max(Num_Res,1)
          Ter_4 <- 2*para1_setting$lambda*(1-1/pt[r])*Inner_f_t_avGra[r]
          Ter_5 <- para1_setting$lambda^2*(1-1/pt[r])^2*Norm_ave_grad[r]^2
          Norm_overlin_f_t_[r] <- sqrt(Norm_f_t_[r]^2+Ter_1+Ter_2+Ter_3-Ter_4+Ter_5)
          Norm_f_t_[r]         <- Norm_overlin_f_t_[r]
          if(Norm_overlin_f_t_[r]>U)
          {
            svpara <- svpara*U/Norm_overlin_f_t_[r]
            Norm_f_t_[r]   <- U
          }
          ################################## update S_r
          sv_index_list[[r]]  <- sv_index
          sv_coe_list[[r]]    <- svpara
          sv_max_list[[r]]    <- svmat
        }else{
          updated_vector[r,2]  <- 0
          Coef_BudResSam  <- para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)
          Inser_SvElement <- intersect(Index_BudResSam,sv_index)
          if(length(Inser_SvElement)>0&&Inser_SvElement[1]>0)
          {
            Inser_SvIndex <- match(Inser_SvElement,sv_index)
            for(h in Inser_SvIndex)
            {
              inx <- match(sv_index[h],Index_BudResSam)
              svpara[h] <- svpara[h]+Coef_BudResSam[inx]
            }
          }
          ## incorporating the support vector
          Diff_SvIndex <- setdiff(Index_BudResSam,sv_index) ## find the support vector in V but not in S_r, and add them into S_r
          if(length(Diff_SvIndex)>0&&Diff_SvIndex[1]>0)
          {
            for(h in 1:length(Diff_SvIndex))
            {
              inx <- match(Diff_SvIndex[h],Index_BudResSam)
              if(k[r]==0)
              {
                svmat[,1] <- V[,inx]
              }else{
                svmat <- cbind(svmat,V[,inx])
              }
              k[r] <- k[r]+1
              sv_index[k[r]]  <- Index_BudResSam[inx]
              svpara[k[r]]    <- Coef_BudResSam[inx]
            }
          }
          Norm_overlin_f_t_[r] <- Norm_overlin_f_t[r]
          Norm_f_t_[r]         <- Norm_overlin_f_t_[r]
          if(Norm_overlin_f_t_[r]>U)
          {
            svpara     <- svpara*U/Norm_overlin_f_t_[r]
            Norm_f_t_[r]   <- U
          }
          ################################## update S_r
          sv_index_list[[r]]  <- sv_index
          sv_coe_list[[r]]    <- svpara
          sv_max_list[[r]]    <- svmat
        }
      }
    }
    p     <- exp(-1*eta_EA*L)/sum(exp(-1*eta_EA*L))
    L_min <- min(L)
    ########################################## Updating the Reservoir
    if(i<=para1_setting$M)
    {
      ######################## updating the norm of ave-gradient
      diff_V_V <- V - trdata[order[i], ]
      tem  <- colSums(diff_V_V*diff_V_V)
      for(r in 1:len_sigma)
      {
        kvalue  <- exp(tem/(-2*(sigma[r]^2)))
        Norm_ave_grad[r] <- sqrt(((i-1)^2*Norm_ave_grad[r]^2+1+2*ylabel[order[i]]*Grad_BudResSam%*%kvalue)/(i^2))
      }
      if(Num_Res==0)
      {
        V[,1] <- trdata[order[i],]
      }else{
        V <- cbind(V,trdata[order[i],])
      }
      Index_BudResSam[i] <- order[i]
      Grad_BudResSam[i]  <- ylabel[order[i]]
      Num_Res <- Num_Res+1
    }else{
      delta_t <- rbinom(1,1,para1_setting$M/i)
      if(delta_t==1)
      {
        jt   <-sample(1:para1_setting$M,1,replace = T)   #discared
        ############################# updating the norm of ave-gradient
        diff_i_jt   <- trdata[order[i],]-V[,jt]
        colsum_jt_i <- sum(diff_i_jt*diff_i_jt)
        diff_V_jt   <- V - V[,jt]
        colsum_jt   <- colSums(diff_V_jt*diff_V_jt)
        diff_V_i    <- V - trdata[order[i],]
        colsum_i    <- colSums(diff_V_i*diff_V_i)
        for(r in 1:len_sigma)
        {
          T1        <- 2/(Num_Res^2)*(1-ylabel[order[i]]*Grad_BudResSam[jt]*exp(colsum_jt_i/(-2*(sigma[r]^2))))
          kvalue_jt <- exp(colsum_jt/(-2*(sigma[r]^2)))
          T2        <- -2/(Num_Res^2)*Grad_BudResSam[jt]*Grad_BudResSam%*%kvalue_jt
          kvalue_i  <- exp(colsum_i/(-2*(sigma[r]^2)))
          T3        <- 2/(Num_Res^2)*ylabel[order[i]]*Grad_BudResSam%*%kvalue_i
          Norm_ave_grad[r] <- sqrt(Norm_ave_grad[r]^2 +T1+T2+T3)
        }
        ##########################################################
        Index_BudResSam[jt] <- order[i]
        Grad_BudResSam[jt]  <- ylabel[order[i]]
        V[,jt]              <- trdata[order[i],]
      }
    }
  }
  t2 <- proc.time()
  runtime[re] <- (t2 - t1)[3]
  all_error[re] <- Mistake
  errorrate[re] <- Mistake/length_tr
  restart[re,]  <- restart_time
  budget[re,]   <- k
  all_p[re,]    <- p
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--eta--beta--Norm_f--Norm_X--ave_run_time--all_err_rate--sd_time--sd_err"),
  alg_name = c("BEA2OKS"),
  dataname = paste0(Dataset[d_index], ".train"),
  eta = para1_setting$lambda,
  beta  = para1_setting$beta,
  Norm_f  = U,
  Norm_X  = X,
  run_time = as.character(runtime),
  ave_run_time = sum(runtime)/reptimes,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)

detailed_result <- list(
  Budget   = budget,
  restarttime = restart,
  probability = all_p
)
write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 
write.table(detailed_result,file=savepath2,row.names =TRUE, col.names =FALSE, quote = T) 


sprintf("the kernel parameter is %f", para1_setting$sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("the number of support vectors is %d", k)
sprintf("total running time is %.1f in dataset", sum(runtime))
sprintf("average running time is %.1f in dataset", sum(runtime)/reptimes)
sprintf("the error number is %d", all_error)
sprintf("the average error rate is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of average running time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of error is %.5f in dataset", sd(errorrate))