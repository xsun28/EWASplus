files.GenoCanyon10K = c("WholeGenome_GenoCanyon10K_1_1.desc",
                        "WholeGenome_GenoCanyon10K_1_2.desc",
                        "WholeGenome_GenoCanyon10K_1_3.desc",
                        "WholeGenome_GenoCanyon10K_2_1.desc",
                        "WholeGenome_GenoCanyon10K_2_2.desc",
                        "WholeGenome_GenoCanyon10K_2_3.desc",
                        "WholeGenome_GenoCanyon10K_3_1.desc",
                        "WholeGenome_GenoCanyon10K_3_2.desc",
                        "WholeGenome_GenoCanyon10K_3_3.desc",
                        "WholeGenome_GenoCanyon10K_4_1.desc",
                        "WholeGenome_GenoCanyon10K_4_2.desc",
                        "WholeGenome_GenoCanyon10K_4_3.desc",
                        "WholeGenome_GenoCanyon10K_5_1.desc",
                        "WholeGenome_GenoCanyon10K_5_2.desc",
                        "WholeGenome_GenoCanyon10K_5_3.desc",
                        "WholeGenome_GenoCanyon10K_6_1.desc",
                        "WholeGenome_GenoCanyon10K_6_2.desc",
                        "WholeGenome_GenoCanyon10K_6_3.desc",
                        "WholeGenome_GenoCanyon10K_7_1.desc",
                        "WholeGenome_GenoCanyon10K_7_2.desc",
                        "WholeGenome_GenoCanyon10K_8_1.desc",
                        "WholeGenome_GenoCanyon10K_8_2.desc",
                        "WholeGenome_GenoCanyon10K_9_1.desc",
                        "WholeGenome_GenoCanyon10K_9_2.desc",
                        "WholeGenome_GenoCanyon10K_10_1.desc",
                        "WholeGenome_GenoCanyon10K_10_2.desc",
                        "WholeGenome_GenoCanyon10K_11_1.desc",
                        "WholeGenome_GenoCanyon10K_11_2.desc",
                        "WholeGenome_GenoCanyon10K_12_1.desc",
                        "WholeGenome_GenoCanyon10K_12_2.desc",
                        "WholeGenome_GenoCanyon10K_13_1.desc",
                        "WholeGenome_GenoCanyon10K_13_2.desc",
                        "WholeGenome_GenoCanyon10K_14_1.desc",
                        "WholeGenome_GenoCanyon10K_14_2.desc",
                        "WholeGenome_GenoCanyon10K_15.desc",
                        "WholeGenome_GenoCanyon10K_16.desc",
                        "WholeGenome_GenoCanyon10K_17.desc",
                        "WholeGenome_GenoCanyon10K_18.desc",
                        "WholeGenome_GenoCanyon10K_19.desc",
                        "WholeGenome_GenoCanyon10K_20.desc",
                        "WholeGenome_GenoCanyon10K_21.desc",
                        "WholeGenome_GenoCanyon10K_22.desc",
                        "WholeGenome_GenoCanyon10K_23_1.desc",
                        "WholeGenome_GenoCanyon10K_23_2.desc",
                        "WholeGenome_GenoCanyon10K_24.desc")

Chr.GenoCanyon10K = c(1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,16,17,18,19,20,21,22,23,23,24)

PosStart.GenoCanyon10K = c(1,80000001,160000001,1,80000001,160000001,1,60000001,120000001,1,60000001,120000001,1,60000001,120000001,1,60000001,120000001,
                           1,80000001,1,80000001,1,60000001,1,60000001,1,60000001,1,60000001,1,60000001,1,60000001,1,1,1,1,1,1,1,1,1,80000001,1)

PosStop.GenoCanyon10K = c(80000000,160000000,249250621,80000000,160000000,243199373,60000000,120000000,198022430,60000000,120000000,191154276,60000000,120000000,180915260,
                          60000000,120000000,171115067,80000000,159138663,80000000,146364022,60000000,141213431,60000000,135534747,60000000,135006516,60000000,133851895,
                          60000000,115169878,60000000,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,80000000,155270560,59373566)



###   Chr in Character -> Chr in Number   ###
CharToNum <- function(chr)
{
    result = rep(NA, length(chr))
    name = c(paste0('chr',1:22),"chrX","chrY")
    for(i in 1:24) 
    {
        L = sum(chr == name[i])
        result[chr == name[i]] = rep(i, L)
    }
    return(result)
}
