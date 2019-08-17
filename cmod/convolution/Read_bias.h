
double* Read_bias(FILE* fp2, double* conv1_bias, double* conv2_bias, double* ip1_bias, double* ip2_bias, int sel)
{
    int i;
    char dump[10];  // Dump string
    char cha;       // Dump char

    int conv1_bias_w, conv2_bias_w, ip1_bias_w, ip2_bias_w; 
    
    if(sel == 0){
    fscanf(fp2,"%s",dump);
    fscanf(fp2,"%s",dump);
    fscanf(fp2," [%d],", &conv1_bias_w);

    conv1_bias = (double*)malloc(conv1_bias_w * sizeof(double));

    fscanf(fp2,"%s",dump);  // "data"
    for(i=0; i < conv1_bias_w; i++){
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%lf ", &conv1_bias[i]);
    }
    return conv1_bias;
    }
    
    else if(sel == 1){
    fscanf(fp2,"%s",dump);
    fscanf(fp2," %s",dump);
    fscanf(fp2,"%s",dump);
    fscanf(fp2," [%d],", &conv2_bias_w);

    conv2_bias = (double*)malloc(conv2_bias_w * sizeof(double));

    fscanf(fp2,"%s",dump);  // "data"
    for(i=0; i < conv2_bias_w; i++){
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%lf ", &conv2_bias[i]);
    }
    return conv2_bias;
    }

    else if(sel == 2){
    fscanf(fp2,"%s",dump);
    fscanf(fp2," %s",dump);
    fscanf(fp2,"%s",dump);
    fscanf(fp2," [%d],", &ip1_bias_w);

    ip1_bias = (double*)malloc(ip1_bias_w * sizeof(double));

    fscanf(fp2,"%s",dump);  // "data"
    for(i=0; i < ip1_bias_w; i++){
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%lf ", &ip1_bias[i]);
    }
    return ip1_bias;
    }

    else if(sel == 3){
    fscanf(fp2,"%s",dump);
    fscanf(fp2," %s",dump);
    fscanf(fp2,"%s",dump);
    fscanf(fp2," [%d],", &ip2_bias_w);

    ip2_bias = (double*)malloc(ip2_bias_w * sizeof(double));

    fscanf(fp2,"%s",dump);  // "data"
    for(i=0; i < ip2_bias_w; i++){
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%c",&cha);
        fscanf(fp2,"%lf ", &ip2_bias[i]);
    }
    return ip2_bias;
    }

}