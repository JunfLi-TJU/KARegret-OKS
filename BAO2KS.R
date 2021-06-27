#setwd("E:/experiment/ECML/code")
rm(list = ls())

d_index <- 4

# set the path of datasets
dpath          <- file.path("E:/experiment/binary C") 

Dataset        <- c("w8a", "ijcnn1_all","a9a_all","w7a")

# set the path of results
savepath1      <- paste0("E:/experiment/ECML/ECML2021 Result/",paste0("BAO2KS-c-",Dataset[d_index],".txt"))
savepath2      <- paste0("E:/experiment/ECML/ECML2021 Result/",paste0("BAO2KS-c-all-",Dataset[d_index],".txt"))

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
  delta      = 0.01   # fail probability in GREEN-IX algorithm
)

reptimes   <- 10

power      <- seq(-2,3,1)
sigma      <- 2^(power)
len_sigma  <- length(sigma)

X          <- 1
U          <- 20
Freeze_p   <- c(rep(0, len_sigma))
Psi_delta  <- len_sigma*log(len_sigma/para1_setting$delta)

runtime    <- c(rep(0, reptimes))
errorrate  <- c(rep(0, reptimes))
all_error  <- c(rep(0, reptimes))
maxL       <- 0


for(re in 1:reptimes)
{
  order    <- sample(1:length_tr,length_tr,replace = F)   #dis
  Mistake  <- 0
  k        <- c(rep(0,len_sigma))
  p        <- c(rep(1/len_sigma,len_sigma))
  L        <- c(rep(0,len_sigma))
  hatL     <- 0
  tau      <- 1
  epsilon_GREEN_IX  <- 2^{-tau}
  epsilon_GREEN_IX_ <- epsilon_GREEN_IX/2
  eta_GREEN_IX      <- epsilon_GREEN_IX_/(2*len_sigma)
  zeta_GREEN_IX     <- eta_GREEN_IX
  gamma_GREEN_ix    <- epsilon_GREEN_IX_/len_sigma
  
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
  V         <- matrix(0,nrow = feature_tr,ncol=1)                  ######## reservoir
  
  Index_BudResSam    <- array(0,1)                                 ######## the index of instance in V
  Grad_BudResSam     <- array(0,1)
  Num_Res            <- 0                                          ######## record the size of reservoir, v
  Norm_ave_grad      <- c(rep(0, len_sigma))                       ######## record the norm of average gradient
  Norm_overlin_f_t   <- c(rep(0, len_sigma))                       ######## record the norm of \vertlone{f}_t
  Norm_overlin_f_t_  <- c(rep(0, len_sigma))                       ######## record the norm of \overline{f'}_t
  Norm_f_t_          <- c(rep(0, len_sigma))                       ######## record the norm of f'_t
  Inner_f_t_avGra    <- c(rep(0, len_sigma))                       ######## inner between f'_t and reservoir estimator
  delta_t            <- 0
  updated_vector     <- c(rep(0, len_sigma))                       ## updated_vector[r,1]:= yf(x)<\varepsilon
  aver_G_nabla_t_    <- matrix(0,nrow=len_sigma,ncol=2)
  Jt_                <- 1
  Jt                 <- 1
  
  t1       <- proc.time()                                          ######## proc.time()
  for (i in 1:length_tr)
  {
    if(epsilon_GREEN_IX^2*hatL>Psi_delta)
    {
      tau               <- tau+1
      epsilon_GREEN_IX  <- 2^{-tau}
      epsilon_GREEN_IX_ <- epsilon_GREEN_IX/2
      p                 <- c(rep(1/len_sigma,len_sigma))
      eta_GREEN_IX      <- epsilon_GREEN_IX_/(2*len_sigma)
      zeta_GREEN_IX     <- eta_GREEN_IX
      gamma_GREEN_ix    <- epsilon_GREEN_IX_/len_sigma
      L                 <- c(rep(0,len_sigma))
    }
    p[p<gamma_GREEN_ix] <- 0
    Freeze_p            <- p/sum(p)
    It                  <- sample(1:len_sigma, 1, replace = T, prob = Freeze_p)
    q                   <- c(rep(1/len_sigma,len_sigma))
    Jt_                 <- Jt
    Jt                  <- sample(1:len_sigma, 1, replace = T, prob = q)
    
    diff_V_i      <- V - trdata[order[i], ]
    colsum_in_V   <- colSums(diff_V_i*diff_V_i)
    for(r in 1:len_sigma)
    {
      sv_index    <- sv_index_list[[r]]
      svpara      <- sv_coe_list[[r]]
      svmat       <- sv_max_list[[r]]
      
      coe_fx_S    <- svpara
      kvalue_V    <- exp(colsum_in_V/(-2*(sigma[r]^2)))
      aver_G_nabla_t_[r,1] <- aver_G_nabla_t_[r,2]
      aver_G_nabla_t_[r,2] <- as.numeric(Grad_BudResSam %*%kvalue_V)
      
      if(r==Jt || r==It)
      {
        ############ compute the norm of f_t
        diff_S_i    <- svmat - trdata[order[i], ]
        colsum_in_S <- colSums(diff_S_i*diff_S_i)
        kvalue_S    <- exp(colsum_in_S/(-2*(sigma[r]^2)))
        fx          <- coe_fx_S %*% kvalue_S + para1_setting$lambda*aver_G_nabla_t_[r,2]/max(Num_Res,1)
      }
      
      ######################## updating Inner_f_t_avGra
      if(i>= 2 && i<=(para1_setting$M+1) && sv_index[1]>0)
      {
        sum_tem_fx  <- 0
        for(indx in 1:Num_Res)
        {
          diff_V_V  <- svmat - V[,indx]
          tem_colsum_in_V  <- colSums(diff_V_V*diff_V_V)
          sum_tem_fx       <- sum_tem_fx+Grad_BudResSam[indx]*coe_fx_S %*% exp(tem_colsum_in_V/(-2*(sigma[r]^2)))
        }
        Inner_f_t_avGra[r]    <- -1*sum_tem_fx/Num_Res
      }
      if(i>=(para1_setting$M+2) && sv_index[1]>0)
      {
        if(delta_t==0)
        {
          if(r == Jt_ && updated_vector[r]==1)
          {
            Ter_1           <- para1_setting$lambda/q[r]*ylabel[order[i-1]]*aver_G_nabla_t_[r,1]/Num_Res
            Ter_2           <- para1_setting$lambda*(1-1/q[r])*Norm_ave_grad[r]^2
            Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]-Ter_1-Ter_2
            if(Norm_overlin_f_t_[r]>U)
            {
              Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]*U/Norm_overlin_f_t_[r]
            }
          }
          else if(r == Jt_ && updated_vector[r] == 0)
          {
            Inner_f_t_avGra[r] <-Inner_f_t_avGra[r] - para1_setting$lambda*(1-1/q[r])*Norm_ave_grad[r]^2
            if(Norm_overlin_f_t_[r]>U)
            {
              Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]*U/Norm_overlin_f_t_[r]
            }
          }
          else  
          {
            Inner_f_t_avGra[r] <- Inner_f_t_avGra[r] - para1_setting$lambda*Norm_ave_grad[r]^2
            if(Norm_overlin_f_t_[r]>U)
            {
              Inner_f_t_avGra[r] <- Inner_f_t_avGra[r]*U/Norm_overlin_f_t_[r]
            }
          }
        }
        else
        {
          sum_tem_fx  <- 0
          for(indx in 1:Num_Res)
          {
            diff_V_V <- svmat - V[,indx]
            tem_colsum_in_V  <- colSums(diff_V_V*diff_V_V)
            sum_tem_fx       <- sum_tem_fx+Grad_BudResSam[indx]*coe_fx_S %*% exp(tem_colsum_in_V/(-2*(sigma[r]^2)))
          }
          Inner_f_t_avGra[r]    <- -1*sum_tem_fx/Num_Res
        }
      }
      updated_vector[r] <- 0
      if(r==Jt)
      {
        #################################### compute nabla_{t,i} and output prediction
        ter_1          <- para1_setting$lambda^2*(Norm_ave_grad[r]^2)
        ter_2          <- 2*para1_setting$lambda*Inner_f_t_avGra[r]
        Norm_overlin_f_t[r]  <- sqrt(Norm_f_t_[r]^2 + ter_1 - ter_2)
        if(Norm_overlin_f_t[r] > U)
        {
          fx       <- fx*U/Norm_overlin_f_t[r]
        }
        if(fx*ylabel[order[i]] < para1_setting$varepsilon)
        {
          updated_vector[r] <- 1
          if(k[r]==0)
          {
            svmat[,1] <- trdata[order[i],]          ## updating the budget
          }else{
            svmat     <- cbind(svmat,trdata[order[i],]) ## updating the budget
          }
          k[r]            <- k[r]+1
          sv_index[k[r]]  <- order[i]
          svpara[k[r]]    <- para1_setting$lambda*ylabel[order[i]]/q[r]
          Coef_BudResSam  <- (1-1/q[r])*para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)
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
          Diff_SvIndex   <- setdiff(Index_BudResSam,sv_index)
          if(length(Diff_SvIndex)>0 && Diff_SvIndex[1]>0)
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
          Ter_1 <- para1_setting$lambda^2/(q[r]^2)
          Ter_2 <- 2*para1_setting$lambda/q[r]*coe_fx_S %*% kvalue_S*ylabel[order[i]]
          Ter_3 <- 2*para1_setting$lambda^2*(1-1/q[r])/q[r]*aver_G_nabla_t_[r,2]*ylabel[order[i]]/max(Num_Res,1)
          Ter_4 <- 2*para1_setting$lambda*(1-1/q[r])*Inner_f_t_avGra[r]
          Ter_5 <- para1_setting$lambda^2*(1-1/q[r])^2*Norm_ave_grad[r]^2
          #tenmm <- Norm_f_t_[r]
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
          updated_vector[r] <- 0
          Coef_BudResSam    <- (1-1/q[r])*para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)
          Inser_SvElement   <- intersect(Index_BudResSam,sv_index)
          if(length(Inser_SvElement)>0 && Inser_SvElement[1]>0)
          {
            Inser_SvIndex   <- match(Inser_SvElement,sv_index)
            for(h in Inser_SvIndex)
            {
              inx <- match(sv_index[h],Index_BudResSam)
              svpara[h] <- svpara[h]+Coef_BudResSam[inx]
            }
          }
          ## incorporating the support vector
          Diff_SvIndex <- setdiff(Index_BudResSam,sv_index) ## find the support vector in V but not in S_r, and add them into S_r
          if(length(Diff_SvIndex)>0 && Diff_SvIndex[1]>0)
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
              k[r]    <- k[r]+1
              sv_index[k[r]]  <- Index_BudResSam[inx]
              svpara[k[r]]    <- Coef_BudResSam[inx]
            }
          }
          ter_1 <- (para1_setting$lambda^2)*(1-1/q[r])^2*(Norm_ave_grad[r]^2)
          ter_2 <- 2*para1_setting$lambda*(1-1/q[r])*Inner_f_t_avGra[r]
          Norm_overlin_f_t_[r] <- sqrt(Norm_f_t_[r]^2 +ter_1-ter_2)
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
      else
      {
        if(r== It)
        {
          ter_1          <- para1_setting$lambda^2*(Norm_ave_grad[r]^2)
          ter_2          <- 2*para1_setting$lambda*Inner_f_t_avGra[r]
          Norm_overlin_f_t[r]  <- sqrt(Norm_f_t_[r]^2 + ter_1 - ter_2)
          if(Norm_overlin_f_t[r] > U)
          {
            fx       <- fx*U/Norm_overlin_f_t[r]
          }
          haty  <- 1     
          if(fx < 0)  
            haty <- -1
          ell_It<- (para1_setting$varepsilon - fx*ylabel[order[i]])/(para1_setting$varepsilon+X*U)
          L[r]  <- L[r] + ell_It/(zeta_GREEN_IX+Freeze_p[It])
          hatL  <- hatL+ ell_It/(zeta_GREEN_IX+Freeze_p[It])
          if(maxL< (para1_setting$varepsilon - fx*ylabel[order[i]]))
            maxL<-(para1_setting$varepsilon - fx*ylabel[order[i]])
          if(haty != ylabel[order[i]])
          {
            Mistake <- Mistake + 1
          }
        }
        Coef_BudResSam  <- para1_setting$lambda*Grad_BudResSam/max(Num_Res,1)
        Inser_SvElement <- intersect(Index_BudResSam,sv_index)
        if(length(Inser_SvElement)>0 && Inser_SvElement[1]>0)
        {
          Inser_SvIndex <- match(Inser_SvElement,sv_index)
          for(h in Inser_SvIndex)
          {
            inx         <- match(sv_index[h],Index_BudResSam)
            svpara[h]   <- svpara[h]+Coef_BudResSam[inx]
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
        ter_1   <- para1_setting$lambda^2*(Norm_ave_grad[r]^2)
        ter_2   <- 2*para1_setting$lambda*Inner_f_t_avGra[r]
        Norm_overlin_f_t_[r] <- sqrt(Norm_f_t_[r]^2 + ter_1 - ter_2)
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
    p     <- exp(-1*eta_GREEN_IX*L)/sum(exp(-1*eta_GREEN_IX*L))
    ########################################## Updating the Reservoir
    if(i<=para1_setting$M)
    {
      ######################## updating the norm of ave-gradient
      diff_V_i <- V - trdata[order[i], ]
      tem      <- colSums(diff_V_i*diff_V_i)
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
        colsum_jt_i <- sum((trdata[order[i],]-V[,jt])*(trdata[order[i],]-V[,jt]))
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
}

save_result <- list(
  note     = c(" the next term are:alg_name--datanames--eta--sam_num--run_time--tot_run_time--ave_run_time--err_num--all_err_rate--ave_err_rate--sd_time--sd_err"),
  alg_name = c("BAO2KS"),
  dataname = paste0(Dataset[d_index], ".train"),
  eta = para1_setting$lambda,
  sam_num  = length_tr,
  run_time = as.character(runtime),
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  err_num  = errorrate,
  all_err_rate = as.character(all_error),
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time  <- sd(runtime),
  sd_err    <-sd(errorrate)
)
write.table(save_result,file=savepath1,row.names =TRUE, col.names =FALSE, quote = T) 

sprintf("the kernel parameter is %f", para1_setting$sigma)
sprintf("the number of sample is %d", length_tr)
sprintf("the number of support vectors is %d", k)
sprintf("total running time is %.1f in dataset", sum(runtime))
sprintf("average running time is %.1f in dataset", sum(runtime)/reptimes)
sprintf("the error number is %d", all_error)
sprintf("the average error rate is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of average running time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of average error is %.5f in dataset", sd(errorrate))
